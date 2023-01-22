"""Test the obstacle avoidance BarrierNetLayer for a dubins vehicle"""
from typing import Callable, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
import dill
import cvxpy as cp
import warnings

sys.path.append(os.path.abspath('..'))

from mpc.dynamics_constraints import car_2d_dynamics as dubins_car_dynamics
from mpc.simulator import simulate_barriernet

from cbf.mlpbarriernet import MLPBarrierNet
from cvxpylayers.torch import CvxpyLayer

# -------------------------------------------
# PROBLEM PARAMETERS
n_states = 4
n_controls = 2
horizon = 15
dt = 0.1

# -------- Define Vehicle Dynamics --------
dynamics_fn = dubins_car_dynamics
# Define limits for state space
state_space = [(0.0, 50.0),
               (0.0, 50.0),
               (0, 2),
               (-np.pi, np.pi)]
# Define control bounds
control_bounds = [1, 0.5]
torch.set_default_dtype(torch.float64)
# -------- Define Number of cbf and clf constraints --------
n_clf = 2  # Number of CLF constraints [V_speed, V_angle]
n_cbf = 3  # Number of CBF constraints [b_radius, b_v_min, b_v_max]
n_clf_slack = n_clf  # Number of CLF slack variables
n_cbf_slack = 1  # Number of CBF slack variables
slack_weights = [1., 1., 1000.]
rel_degree = [2, 1, 1]  # Relative degree of the CBFs [distance, v_min, v_max]
num_penalty_terms = n_controls+n_clf + sum(rel_degree)
# -------- Define obstacles --------
radius = 7.0
margin = 0.0
center = [32.0, 25.0]
# --------- Define Goal and Initial State ---------
# Define initial states
x0s = [
    np.array([5.0, 25.0, 0.0, 0.0]),
]
# Define goal state
x_goal = np.array([45.0, 25.001, 2])

# -------------------------------------------
# DEFINE DEVICE
# Define device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


# -------------------------------------------
def define_hocbf_clf_filter(n_controls, n_clf, n_cbf, n_cbf_slack, cntrl_bounds=[], slack_weights=[]):
    """ Define a General Differentiable Control Lyapunov function and High Order Control Barrier function """
    # -------- Define Optimization Variables --------
    u = cp.Variable((n_controls, 1))  # control input
    # Define the control Lyapunov function variables
    slack_clf = cp.Variable((n_clf, 1))  # slack variables
    # Define the control Barrier function variables
    if n_cbf > 1 and n_cbf_slack > 0:
        slack_cbf = cp.Variable((n_cbf_slack, 1))
    else:
        slack_cbf = torch.zeros(n_cbf,1)
    # Create a list of slack variables
    if n_cbf_slack > 0:
        slack = cp.vstack([slack_clf, slack_cbf])
    else:
        slack = slack_clf
    if slack_cbf.shape[0] != n_cbf:
        delta_cbf = cp.vstack([slack_cbf, torch.zeros(n_cbf - n_cbf_slack, 1)])
    else:
        delta_cbf = slack_cbf
    X = cp.vstack([u, slack])
    # -------- Define Differentiable Variables --------
    # Define Constraint Parameters CLF
    A_clf = cp.Parameter((n_clf, n_controls))  # CLF matrix
    b_clf = cp.Parameter((n_clf,1))  # CLF vector
    # Define Constraint Parameters CBF
    A_cbf = cp.Parameter((n_cbf, n_controls))  # CBF matrix
    b_cbf = cp.Parameter((n_cbf,1))  # CBF vector
    # Define Objective Parameters
    cntrl_weights = cp.Parameter((n_controls, 1), nonneg=True)  # Q_tunning matrix for the control input
    # Add weights to the slack variables CLF
    n_slack = n_clf + n_cbf_slack
    if slack_weights:
        n_weights = len(slack_weights)
        if n_weights != n_slack:
            raise "Error: the number of slack weights must be equal to the number of slack variables"
        weights = cp.vstack([cntrl_weights, torch.tensor(slack_weights).reshape(n_slack, 1)])
    else:
        weights = cp.vstack([cntrl_weights, torch.ones(n_clf, 1).requires_grad_(False)])
    # Compute the objective function
    objective = 0.5*cp.Minimize(cp.sum(cp.multiply(weights, cp.square(X))))
    # -------- Define Constraints --------
    contraints = []
    # Define the constraints CLF
    contraints += [A_clf @ u <= b_clf+slack_clf]
    # Define the constraints CBF
    contraints += [A_cbf @ u <= b_cbf+delta_cbf]
    # Define the constraints on the control input
    if cntrl_bounds:
        for control_idx, bound in enumerate(control_bounds):
            contraints += [u[control_idx] <= bound]
            contraints += [u[control_idx] >= -bound]
    # -------- Define the Problem --------
    problem = cp.Problem(objective, contraints)
    if n_cbf_slack>0:
        return CvxpyLayer(problem, parameters=[A_clf, b_clf, A_cbf, b_cbf, cntrl_weights], variables=[u,slack_clf, slack_cbf])
    else:
        return CvxpyLayer(problem, parameters=[A_clf, b_clf, A_cbf, b_cbf, cntrl_weights], variables=[u,slack_clf])


def compute_parameters_hocbf(parameters: torch.Tensor, x: torch.Tensor, x_goal: torch.Tensor, x_obst: torch.Tensor,
                             r_obst: torch.Tensor, hocbf_clf_layer, n_controls: int, n_clf: int, n_cbf: int,
                             rel_degree: List[int], device: torch.device = torch.device("cpu")):
    """ Compute constraint parameters for the CLF and CBF """
    # Housekeeping
    # Make sure that the list of relative degrees is correct
    if len(rel_degree) != n_cbf:
        raise "Error: the number of relative degrees must be equal to the number of CBFs"
    # Extract vehicle state
    x = torch.from_numpy(x).to(device)
    px,py,v,theta = x
    # -------- Extract Parameters from NN --------
    # Compute the total number of cbf rates corresponding to the relative degree
    n_cbf_rates = sum(rel_degree)
    # Extract control weights from NN
    cntrl_weights = torch.reshape(parameters[:n_controls], (n_controls, 1))
    # Extract the CLF rates from the NN
    clf_rates = torch.reshape(parameters[n_controls:n_controls + n_clf], (n_clf, 1))
    # Extract the CBF rates from the NN
    cbf_rates = torch.reshape(parameters[n_controls + n_clf:n_controls + n_clf + n_cbf_rates], (n_cbf_rates, 1))
    # -------- Compute CLF and CBF Constraint Parameters --------
    # Compute CLF parameters
    # TODO this can be called for automatic differentiation given a vector of Lyapunov functions
    LfV = torch.zeros(n_clf, 1).to(device).requires_grad_(False)
    LgV = torch.vstack([2 * (v - x_goal[2]),
                        -2 * (torch.cos(theta) * (px - x_goal[0]) + torch.sin(theta) *
                              (py - x_goal[1])) * (torch.cos(theta) * (py - x_goal[1]) -
                               torch.sin(theta) * (px - x_goal[0]))])
    V_speed = torch.square(x[2] - x_goal[2])
    V_angle = torch.square(torch.cos(theta) * (py - x_goal[1]) - torch.sin(theta) * (px - x_goal[0]))
    V = torch.vstack([V_speed, V_angle])
    A_clf = torch.diag(LgV.squeeze())
    b_clf = -LfV - clf_rates*V
    # Compute CBF parameters
    A_cbf = torch.zeros(n_cbf, n_controls).to(device)
    b_cbf = torch.zeros(n_cbf, 1).to(device)
    # Distance from Obstacle
    for i in range(len(x_obst)):
        p1,p2 = cbf_rates[i*2:i*2+2]
        b_dist = (px - x_obst[i,0])**2 + (py - x_obst[i,1]) ** 2 - r_obst[i] ** 2
        LgLfb_dist = torch.hstack([2 * torch.cos(theta) * (px - x_obst[i,0]) + 2 * torch.sin(theta) * (py - x_obst[i,1]),
                                   2 * v * torch.cos(theta) * (py - x_obst[i,1]) - 2 * v * torch.sin(theta) *
                                   (px - x_obst[i,0])])
        Lfb_dist = 2 * (px - x_obst[i,0]) * v * torch.cos(theta) + \
                   2 * (py - x_obst[i,1]) * v * torch.sin(theta)
        Lf2b_dist = 2 * v**2
        A_cbf[i, :n_controls] = -LgLfb_dist
        b_cbf[i] = Lf2b_dist + (p1 + p2) * Lfb_dist + p1 * p2 * b_dist
    # Velocity Bounds Barrier
    b_v = torch.vstack([v - 0.2, 2. - v])
    Lgb_v = torch.tensor([1,-1]).to(device)
    n_dist = len(x_obst)
    cbf_rates_v = cbf_rates[sum(rel_degree[:n_dist]):]

    A_cbf[n_dist:, 0] = -Lgb_v
    b_cbf[n_dist:] = cbf_rates_v * b_v

    # Compute control input by passing the parameters to the CBF-CLF layer
    try:
        u = hocbf_clf_layer(A_clf, b_clf, A_cbf, b_cbf, cntrl_weights, solver_args={"solve_method": "ECOS"})[0]
    except Exception as e:
        warnings.warn("Error: {}".format(e))
        u = torch.zeros(n_controls).to(device)

    return u.squeeze().detach().cpu()


def simulate_and_plot(policy):
    # -------------------------------------------
    # Simulate the cloned policy
    # -------------------------------------------
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([], [], "ro", label="Start")

    n_steps = 500
    for x0 in x0s:
        # Run the cloned policy
        _, x, u = simulate_barriernet(
            policy,
            x0,
            n_states,
            n_controls,
            dt,
            dynamics_fn,
            n_steps,
            substeps=10,
        )

        # Plot it
        ax.plot(x0[0], x0[1], "ro")
        ax.plot(x[:, 0], x[:, 1], "r-")

    # Plot obstacle
    theta = np.linspace(0, 2 * np.pi, 100)
    obs_x = radius * np.cos(theta) + center[0]
    obs_y = radius * np.sin(theta) + center[1]
    margin_x = (radius + margin) * np.cos(theta) + center[0]
    margin_y = (radius + margin) * np.sin(theta) + center[1]
    ax.plot(obs_x, obs_y, "k-")
    ax.plot(margin_x, margin_y, "k:")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    lim_min = min(min(x[:,0]), min(x[:,1]))
    lim_max = max(max(x[:,0]), max(x[:,1]))
    lim_min = min([lim_min, center[0]-radius, center[1] - radius]);
    lim_max = max([lim_max, center[0]+radius, center[1] + radius]);
    ax.set_xlim([lim_min, lim_max])
    ax.set_ylim([lim_min, lim_max])
    ax.title.set_text("Test BarrierNetLayer Policy")

    ax.set_aspect("equal")

    ax.legend()

    plt.show()


if __name__ == "__main__":
    barrier_fn = define_hocbf_clf_filter(n_controls, n_clf, n_cbf, n_cbf_slack, cntrl_bounds=control_bounds,
                                         slack_weights=slack_weights)
    x_obstacle = torch.tensor([center]).to(device).requires_grad_(False)
    # x_obstacle = torch.vstack([x_obstacle, torch.tensor([-10., -10.])])
    x_g = torch.tensor(x_goal).to(device).requires_grad_(False)
    r_obstacle = torch.tensor([radius + margin]).to(device).requires_grad_(False)
    # r_obstacle = torch.vstack([r_obstacle, torch.tensor([7.])])

    # p_weights = torch.tensor([1.,1.,1., 1., 1.,50.,1., 50., 1., 1.]).to(device)
    p_weights = torch.tensor([1., 1., 1., 1., 1., 50., 1., 1.]).to(device)
    policy = lambda x_state: compute_parameters_hocbf(p_weights, x_state, x_g, x_obstacle,
                                                                            r_obstacle, barrier_fn, n_controls, n_clf,
                                                                            n_cbf, rel_degree, device)
    simulate_and_plot(policy)
