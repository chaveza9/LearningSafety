"""Test the obstacle avoidance BarrierNet for a dubins vehicle"""
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

from mpc.costs import lqr_running_cost, squared_error_terminal_cost
from mpc.dynamics_constraints import car_2d_dynamics as dubins_car_dynamics
from mpc.mpc import construct_MPC_problem, solve_MPC_problem
from mpc.obstacle_constraints import hypersphere_sdf
from mpc.simulator import simulate_nn

from cbf.barriernet import BarrierNet
from cvxpylayers.torch import CvxpyLayer

# -------------------------------------------
# PROBLEM PARAMETERS
n_states = 4
n_controls = 2
horizon = 20
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
num_penalty_terms = n_controls+n_clf+sum(rel_degree)
# -------- Define obstacles --------
radius = 0.2
margin = 0.1
center = [-1.0, 0.0]
# --------- Define Goal and Initial State ---------
# Define initial states
x0s = [
    np.array([-2.0, 0.0, 0.0, 0.0]),
    np.array([-2.0, 0.1, 0.0, 0.0]),
    np.array([-2.0, 0.2, 0.0, 0.0]),
    np.array([-2.0, 0.5, 0.0, 0.0]),
    np.array([-2.0, -0.1, 0.0, 0.0]),
    np.array([-2.0, -0.2, 0.0, 0.0]),
    np.array([-2.0, -0.5, 0.0, 0.0]),
]
# Define goal state
x_goal = np.array([0.0, 0.001, 0.5, 0.0])

# -------------------------------------------
# DEFINE DEVICE
# Define device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def define_dubins_mpc_expert() -> Callable[[torch.Tensor], torch.Tensor]:
    """Run a test of obstacle avoidance MPC with a dubins car and return the results"""
    # -------------------------------------------
    # Define the problem
    # -------------------------------------------
    # Define obstacles by defining a signed distance function
    obstacle_fns = [(lambda x: hypersphere_sdf(x, radius, [0, 1], center), margin)]

    # Define costs
    running_cost_fn = lambda x, u: lqr_running_cost(
        x, u, x_goal, dt * np.diag([1.0, 1.0, 0.1, 0.0]), 0.05 * np.eye(2)
    )
    terminal_cost_fn = lambda x: squared_error_terminal_cost(x, x_goal)

    # Define MPC problem
    opti, x0_variables, u0_variables, x_variables, u_variables = construct_MPC_problem(
        n_states,
        n_controls,
        horizon,
        dt,
        dynamics_fn,
        obstacle_fns,
        running_cost_fn,
        terminal_cost_fn,
        control_bounds,
    )

    # -------------------------------------------
    # Wrap the MPC problem to accept a tensor input and tensor output
    # -------------------------------------------
    max_tries = 15
    def mpc_expert(current_state: torch.Tensor) -> torch.Tensor:
        # Initialize counters and variables
        tries = 0
        success = False
        x_guess = None
        u_guess = None

        while not success and tries < max_tries:
            success, control_out, _, _ = solve_MPC_problem(
                opti.copy(),
                x0_variables,
                u0_variables,
                current_state.detach().numpy(),
                x_variables=x_variables,
                u_variables=u_variables,
                x_guess=x_guess,
                u_guess=u_guess,
            )
            tries += 1

            if not success:
                print(f"failed after {tries} tries")

        return torch.from_numpy(control_out)

    return mpc_expert


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


    return u.squeeze()


def clone_dubins_barrier_preferences(train=True, load=False):
    # Define Barrier Function
    barrier_fn = define_hocbf_clf_filter(n_controls, n_clf, n_cbf, n_cbf_slack, cntrl_bounds=control_bounds,
                                         slack_weights=slack_weights)
    x_obstacle = torch.tensor([center]).to(device).requires_grad_(False)
    x_g = torch.tensor(x_goal).to(device).requires_grad_(False)
    r_obstacle = torch.tensor([radius + margin]).to(device).requires_grad_(False)

    barrier_policy_fn = lambda x_state, p_weights: compute_parameters_hocbf(p_weights, x_state, x_g, x_obstacle,
                                                                            r_obstacle, barrier_fn, n_controls, n_clf,
                                                                            n_cbf, rel_degree, device)
    # -------------------------------------------
    # Clone the MPC policy
    # -------------------------------------------
    mpc_expert = define_dubins_mpc_expert()
    hidden_layers = 4
    hidden_layer_width = 32
    cloned_policy = BarrierNet(
        hidden_layers= hidden_layers,
        hidden_layer_width= hidden_layer_width,
        n_state_dims= n_states,
        n_control_dims= n_controls,
        n_input_dims= n_states,
        n_output_dims= num_penalty_terms,
        state_space=state_space,
        barrier_net_fn=barrier_policy_fn,
        preprocess_input_fn=None,
        # load_from_file="mpc/tests/data/cloned_quad_policy_weight_decay.pth",
    )

    n_pts = int(2e4)
    n_epochs = 500
    learning_rate = 1e-3
    # Define Training optimizer
    if train and not load:
        cloned_policy.clone(
            mpc_expert,
            n_pts,
            n_epochs,
            learning_rate,
            batch_size=32,
            save_path="./data/cloned_dubins_barrier_policy_weight_decay.pt",
        )
    elif train and load:
        checkpoint = "./data/cloned_dubins_barrier_policy_weight_decay.pt"
        cloned_policy.clone(
            mpc_expert,
            n_pts,
            n_epochs,
            learning_rate,
            save_path="./data/cloned_dubins_barrier_policy_weight_decay.pt",
            load_checkpoint=checkpoint,
        )
    else:
        load_checkpoint = "./data/cloned_dubins_barrier_policy_weight_decay.pt"
        checkpoint = torch.load(load_checkpoint, map_location=device)
        cloned_policy.load_state_dict(checkpoint["model_state_dict"])

    return cloned_policy


def simulate_and_plot(policy):
    # -------------------------------------------
    # Simulate the cloned policy
    # -------------------------------------------
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([], [], "ro", label="Start")

    n_steps = 100
    for x0 in x0s:
        # Run the cloned policy
        _, x, u = simulate_nn(
            policy,
            x0,
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

    ax.set_xlim([-2.5, 0.5])
    ax.set_ylim([-1.0, 1.0])
    ax.title.set_text("Cloned Dubins Car Policy")

    ax.set_aspect("equal")

    ax.legend()

    plt.show()

if __name__ == "__main__":
    policy = clone_dubins_barrier_preferences(train= True, load=False)
    simulate_and_plot(policy)
