"""Test the obstacle avoidance BarrierNetLayer for a dubins vehicle"""
from typing import Callable, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath('..'))

from mpc.costs import lqr_running_cost, squared_error_terminal_cost
from mpc.dynamics_constraints import car_2d_dynamics as dubins_car_dynamics
from mpc.mpc import construct_MPC_problem, solve_MPC_problem
from mpc.obstacle_constraints import hypersphere_sdf
from mpc.simulator import simulate_barriernet

from cbf.classknn import PolicyCloningModel
from cvxpylayers.torch import CvxpyLayer

# -------------------------------------------
# PROBLEM PARAMETERS
n_states = 4
n_controls = 2
horizon = 10
dt = 0.1

# -------- Define Vehicle Dynamics --------
dynamics_fn = dubins_car_dynamics
# Define limits for state space
state_space = [(-3, 3),
               (-2, 2),
               (0, 2),
               (-np.pi, np.pi)]
# Define control bounds
control_bounds = [(-1, 1),
                  (-0.5, 0.5)]  # [m/s^2, rad/s]
torch.set_default_dtype(torch.float64)
# -------- Define Number of cbf and clf constraints --------
n_cbf = 1  # Number of CBF constraints [b_radius, b_v_min, b_v_max]
distance_cbf = lambda x, x_obst, radius: (x[0] - x_obst[0])**2 + (x[1] - x_obst[1]) ** 2 - radius ** 2
cbf = [distance_cbf]
n_cbf_slack = 1  # Number of CBF slack variables
cbf_slack_weight = [1000.]
rel_degree = [2]  # Relative degree of the CBFs [distance, v_min, v_max]
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

    # --------------- ----------------------------
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


def clone_dubins_barrier_preferences(train=True, load=False):
    # Define Barrier Function
    x_obstacle = torch.tensor([center]).to(device)
    x_g = torch.tensor(x_goal).to(device)
    r_obstacle = torch.tensor([radius + margin]).to(device)

    # -------------------------------------------
    # Clone the MPC policy
    # -------------------------------------------
    mpc_expert = define_dubins_mpc_expert()
    hidden_layers = 4
    hidden_layer_width = 64

    cloned_policy = PolicyCloningModel(
        hidden_layers=hidden_layers,
        hidden_layer_width=hidden_layer_width,
        n_state_dims=n_states,
        n_control_dims=n_controls,
        n_input_dims=n_states,
        cbf= cbf,  # Ordered list of Barrier functions
        n_cbf_slack=n_cbf_slack,
        cbf_slack_weight=cbf_slack_weight,
        cbf_rel_degree=rel_degree,
        state_space=state_space,
        control_bounds=control_bounds,
        x_obst= x_obstacle,
        r_obst= r_obstacle,
    )

    n_pts = int(1e4)
    n_epochs = 400
    learning_rate = 0.001
    path = "/workspaces/learning_safety/cloned_monotonic_barrier.pt"
    # Define Training optimizer
    if train and not load:
        cloned_policy.clone(
            mpc_expert,
            n_pts,
            n_epochs,
            learning_rate,
            batch_size=128,
            save_path=path,
            x_des=x_g,
            x_obs=x_obstacle,
        )
    elif train and load:
        checkpoint = path
        cloned_policy.clone(
            mpc_expert,
            n_pts,
            n_epochs,
            learning_rate,
            batch_size=128,
            save_path=path,
            load_checkpoint=checkpoint,
            x_des=x_g,
            x_obs=x_obstacle,
        )
    else:
        load_checkpoint = path
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
    # Create a lambda function to pass pass the policy together with obstacles and goal
    x_obstacle = torch.tensor([center]).to(device)
    x_g = torch.tensor(x_goal).to(device)

    policy_fn = policy.eval_np
    n_steps = 100
    for x0 in x0s:
        # Run the cloned policy
        _, x, u = simulate_barriernet(
            policy_fn,
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

    ax.set_xlim([-2.5, 0.5])
    ax.set_ylim([-1.0, 1.0])
    ax.title.set_text("Cloned Dubins Car Policy")

    ax.set_aspect("equal")

    ax.legend()

    plt.savefig('comparison_run.png')


if __name__ == "__main__":
    policy = clone_dubins_barrier_preferences(train=False, load=True)
    simulate_and_plot(policy)
