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
from functorch import jacrev
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
# -------------------------------------------
# DEFINE DEVICE
# Define device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# -------------------------------------------
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
# ___________________________________________
# Define control affine dynamics
_f = lambda x: torch.vstack([x[2] * torch.cos(x[3]), x[2] * torch.sin(x[3]), torch.zeros(2, 1).to(device)])
_g = torch.vstack([torch.zeros(2, 2), torch.eye(2, 2)]).to(device)
# -------- Define Number of cbf and clf constraints --------
n_clf = 2  # Number of CLF constraints [V_speed, V_angle]
V_speed = lambda x, x_goal: torch.square(x[2] - x_g[2])
V_angle = lambda x, x_goal: torch.square(torch.cos(x[3]) * (x[1] - x_goal[1]) - torch.sin(x[3]) * (x[0] - x_goal[0]))
clf = [V_speed, V_angle]
# -------- Define the Control Barrier Function --------
n_cbf = 3  # Number of CBF constraints [b_radius, b_v_min, b_v_max]
n_clf_slack = n_clf  # Number of CLF slack variables
n_cbf_slack = 1  # Number of CBF slack variables
distance_cbf = lambda x, x_obst, radius: (x[0] - x_obst[0])**2 + (x[1] - x_obst[1]) ** 2 - radius ** 2
velocity_min_cbf = lambda x, v_min: x[2] - v_min
velocity_max_cbf = lambda x, v_max: v_max - x[2]
cbf = [distance_cbf, velocity_min_cbf, velocity_max_cbf]
slack_weights = [1., 1., 1000.]
rel_degree = [2, 1, 1]  # Relative degree of the CBFs [distance, v_min, v_max]
num_penalty_terms = n_controls + n_clf + sum(rel_degree)
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
        slack_cbf = torch.zeros(n_cbf, 1)
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
    b_clf = cp.Parameter((n_clf, 1))  # CLF vector
    # Define Constraint Parameters CBF
    A_cbf = cp.Parameter((n_cbf, n_controls))  # CBF matrix
    b_cbf = cp.Parameter((n_cbf, 1))  # CBF vector
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
    objective = 0.5 * cp.Minimize(cp.sum(cp.multiply(weights, cp.square(X))))
    # -------- Define Constraints --------
    contraints = []
    # Define the constraints CLF
    contraints += [A_clf @ u <= b_clf + slack_clf]
    # Define the constraints CBF
    contraints += [A_cbf @ u <= b_cbf + delta_cbf]
    # Define the constraints on the control input
    if cntrl_bounds:
        for control_idx, bound in enumerate(control_bounds):
            contraints += [u[control_idx] <= bound]
            contraints += [u[control_idx] >= -bound]
    # -------- Define the Problem --------
    problem = cp.Problem(objective, contraints)
    if n_cbf_slack > 0:
        return CvxpyLayer(problem, parameters=[A_clf, b_clf, A_cbf, b_cbf, cntrl_weights],
                          variables=[u, slack_clf, slack_cbf])
    else:
        return CvxpyLayer(problem, parameters=[A_clf, b_clf, A_cbf, b_cbf, cntrl_weights], variables=[u, slack_clf])


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
    px, py, v, theta = x
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
    LgV = torch.zeros(n_clf, n_controls).to(device).requires_grad_(False)
    V = torch.zeros(n_clf, 1).to(device).requires_grad_(False)
    x.requires_grad_(True)
    for clf_idx in range(n_clf):
        V_i = clf[clf_idx](x, x_goal)
        V_jac_i = torch.autograd.grad(V_i, x, create_graph=True)[0]
        LfV[clf_idx, 0] = V_jac_i@_f(x)
        LgV[clf_idx, :] = V_jac_i@_g
        V[clf_idx, 0] = V_i

    A_clf = LgV
    b_clf = -LfV - clf_rates * V

    A_cbf = torch.zeros(n_cbf, n_controls).to(device)
    b_cbf = torch.zeros(n_cbf, 1).to(device)
    # Distance from Obstacle
    alpha = lambda psi, p: p * psi
    for i in range(len(x_obst)):
        p1, p2 = cbf_rates[i * 2:i * 2 + 2]
        psi0 = distance_cbf(x, x_obst[0], r_obst[0])
        db_dx = torch.autograd.grad(psi0, x, create_graph=True, retain_graph=True)[0]
        Lfb = db_dx @ _f(x)
        db2_dx = torch.autograd.grad(Lfb, x, retain_graph=True)[0]
        Lf2b = db2_dx @ _f(x)
        LgLfb = db2_dx @ _g
        psi1 = Lfb + alpha(psi0, p1)
        psi1_dot = torch.autograd.grad(psi1, x, create_graph=True)[0] @ _f(x)
        psi2 = psi1_dot + alpha(psi1, p2)
        A_cbf[i, :n_controls] = -LgLfb.detach()
        b_cbf[i] = Lf2b + psi2
    # Velocity Bounds Barrier
    b_v = torch.vstack([v - 0.2, 2. - v])
    Lgb_v = torch.tensor([1, -1]).to(device)
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
    lim_min = min(min(x[:, 0]), min(x[:, 1]))
    lim_max = max(max(x[:, 0]), max(x[:, 1]))
    lim_min = min([lim_min, center[0] - radius, center[1] - radius]);
    lim_max = max([lim_max, center[0] + radius, center[1] + radius]);
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
    p_weights = torch.tensor([1., 1., 1, 1, 1., 50., 1., 1.]).to(device)
    policy = lambda x_state: compute_parameters_hocbf(p_weights, x_state, x_g, x_obstacle,
                                                      r_obstacle, barrier_fn, n_controls, n_clf,
                                                      n_cbf, rel_degree, device)
    simulate_and_plot(policy)
