"""Test the obstacle avoidance BarrierNet for a dubins vehicle"""
from typing import Callable, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
import cvxpy as cp


sys.path.append(os.path.abspath('..'))

from NNet.converters.onnx2nnet import onnx2nnet

from mpc.costs import lqr_running_cost, squared_error_terminal_cost
from mpc.dynamics_constraints import dubins_car_dynamics
from mpc.mpc import construct_MPC_problem, solve_MPC_problem
from mpc.obstacle_constraints import hypersphere_sdf
from mpc.simulator import simulate_nn

from mpc.network_utils import pytorch_to_nnet
from cbf.barriernet import BarrierNet
from cvxpylayers.torch import CvxpyLayer

n_states = 4
n_controls = 2
horizon = 15
n_penalty_params = 2
dt = 0.1

# Define dynamics
dynamics_fn = dubins_car_dynamics

# Define obstacles
radius = 0.2
margin = 0.1
center = [-1.0, 0.0]

# Define limits for state space
state_space = [
    (-2.0, 2.0),
    (-2.0, 2.0),
    (-3.0, 3.0),
    (-np.pi, np.pi),
]

def define_dubins_mpc_expert() -> Callable[[torch.Tensor], torch.Tensor]:
    """Run a test of obstacle avoidance MPC with a dubins car and return the results"""
    # -------------------------------------------
    # Define the problem
    # -------------------------------------------
    # Define obstacles by defining a signed distance function
    obstacle_fns = [(lambda x: hypersphere_sdf(x, radius, [0, 1], center), margin)]

    # Define costs
    x_goal = np.array([0.0, 0.0, 0.5, 0.0])
    running_cost_fn = lambda x, u: lqr_running_cost(
        x, u, x_goal, dt * np.diag([1.0, 1.0, 0.0]), 0 * np.eye(2)
    )
    terminal_cost_fn = lambda x: squared_error_terminal_cost(x, x_goal)

    # Define control bounds
    control_bounds = [0.5, np.pi / 2]

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
    max_tries = 10

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

def define_cbf_filter():
    """ Define a differentiable barrier function for the dubins car """
    # Define the barrier function
    u = cp.Variable(n_controls) # control input (velocity, steering angle) [m/s, rad]
    # Define State Parameters
    x_obstacle = cp.Parameter(2) # Obstacle position (x, y) in the world frame [m]
    x_goal = cp.Parameter(n_states) # Goal state (x, y, v, theta) [m, m, m/s, rad]
    x = cp.Parameter(n_states+1) # Current state, augmented with sin and cos of theta (x, y, v , sin(theta), cos(theta))
                                # [m, m, m/s, rad]
    # Define slack variables
    delta = cp.Variable(2) # slack variable for the barrier function
    # Define CBF Parameters
    R_vals = cp.Parameter(n_controls) # Diagonal matrix of the control cost
    R =  cp.diag(R_vals) # Control cost matrix
    p = cp.Parameter(n_penalty_params) # Penalty parameters
    # Define CBF functions
    LgLfb = cp.hstack([-2*(x[0]-x_obstacle[0])*x[2]*x[3]+2*(x[1]-x_obstacle[1])*x[2]*x[4],
             2*(x[0]-x_obstacle[0])*x[4]+2*(x[1]-x_obstacle[1])*x[3]])
    Lf2b = 2*cp.square(x[2])
    Lfb = 2*(x[0]-x_obstacle[0])*x[2]*x[4]+2*(x[1]-x_obstacle[1])*x[2]*x[3]
    b = cp.sum_squares(x[0:2]-x_obstacle)+radius**2
    # Declare CBF constraint
    cbf_constraint = -LgLfb@u <= Lf2b + (p[0]+p[1])*Lfb+p[1]*p[0]*b
    # Define CBF objective
    cbf_objective = cp.quad_form(u, R)
    # Define the CLF objectives
    V_speed = cp.square(x[2]-x_goal[2])
    V_angle = cp.square(x[3]-x_goal[3])

    LgV = cp.vstack([2*(x[2]-x_goal[2]), 2*(x[3]-x_goal[3])])
    V = cp.vstack([V_speed, V_angle])
    # Define the CLF constraint
    clf_constraint = LgV*u + V <= delta
    # CLF objective
    Q = cp.diag([p[3],p[4]])
    clf_objective = cp.quad_form(delta, Q)

    # Define the CBF and CLF problems
    cbf_problem = cp.Problem(cp.Minimize(cbf_objective+clf_objective), [cbf_constraint, clf_constraint])

    return CvxpyLayer(cbf_problem, parameters=[x, x_obstacle, x_goal, R_vals, p], variables=[u, delta])

def cbf_policy

def nn_input(x):
    nn_input = torch.zeros(x.shape[0], n_states + 1)
    nn_input[:, :2] = x[:, :2]
    nn_input[:, 2] = torch.sin(x[:, 2])
    nn_input[:, 3] = torch.cos(x[:, 2])
    return nn_input

def clone_dubins_barrier_preferences(train = True):
    # -------------------------------------------
    # Clone the MPC policy
    # -------------------------------------------
    mpc_expert = define_dubins_mpc_expert()
    hidden_layers = 4
    hidden_layer_width = 32
    cloned_policy = BarrierNet(
        hidden_layers,
        hidden_layer_width,
        n_states,
        n_controls,
        state_space,
        # load_from_file="mpc/tests/data/cloned_quad_policy_weight_decay.pth",
    )

    n_pts = int(2e4)
    n_epochs = 1000
    learning_rate = 1e-3
    # Define Training optimizer
    if train:
        cloned_policy.clone(
            mpc_expert,
            n_pts,
            n_epochs,
            learning_rate,
            save_path="./data/cloned_dubins_policy_weight_decay.pth",
        )

    return cloned_policy

def simulate_and_plot(policy):
    # -------------------------------------------
    # Simulate the cloned policy
    # -------------------------------------------

    # Define initial states
    x0s = [
        np.array([-2.0, 0.0, 0.0]),
        np.array([-2.0, 0.1, 0.0]),
        np.array([-2.0, 0.2, 0.0]),
        np.array([-2.0, 0.5, 0.0]),
        np.array([-2.0, -0.1, 0.0]),
        np.array([-2.0, -0.2, 0.0]),
        np.array([-2.0, -0.5, 0.0]),
    ]

    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([], [], "ro", label="Start")

    n_steps = 50
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
    radius = 0.2
    margin = 0.01
    center = [-1.0, 0.0]
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

def save_to_onnx(policy):
    """Save to an onnx file"""
    save_path = os.path.abspath('./data')+"/cloned_dubins_policy_weight_decay.onnx"
    pytorch_to_nnet(policy, n_states, n_controls, save_path)

    input_mins = [state_range[0] for state_range in state_space]
    input_maxes = [state_range[1] for state_range in state_space]
    means = [0.5 * (state_range[0] + state_range[1]) for state_range in state_space]
    means += [0.0]
    ranges = [state_range[1] - state_range[0] for state_range in state_space]
    ranges += [1.0]
    onnx2nnet(save_path, input_mins, input_maxes, means, ranges)

if __name__ == "__main__":
    policy = clone_dubins_mpc(train=True)
    save_to_onnx(policy)
    simulate_and_plot(policy)
