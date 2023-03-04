"""Test the obstacle avoidance BarrierNetLayer for a dubins vehicle"""
from typing import Callable, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
import datetime

sys.path.append(os.path.abspath('..'))

from mpc.costs import lqr_running_cost, squared_error_terminal_cost
from mpc.dynamics_constraints import car_2d_dynamics as dubins_car_dynamics
from mpc.mpc import construct_MPC_problem, solve_MPC_problem
from mpc.obstacle_constraints import hypersphere_sdf
from mpc.simulator import simulate_barriernet, simulate_mpc

from models.clone.classknn import ClassKNN as PolicyCloningModel


# -------------------------------------------
# PROBLEM PARAMETERS
n_states = 4
n_controls = 2
horizon = 10
dt = 0.1

# -------- Define Vehicle Dynamics --------
dynamics_fn = dubins_car_dynamics
# Define limits for state space
state_space = [(-3, -1),
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
v_min_cbf = lambda x: x[2] - 0.01
v_max_cbf = lambda x: 2 - x[2]
cbf = [distance_cbf, v_min_cbf, v_max_cbf]
n_cbf_slack = 0#1 # Number of CBF slack variables
cbf_slack_weight = [] #[1000.]
rel_degree = [2,1,1]  # Relative degree of the CBFs [distance, v_min, v_max]
# -------- Define obstacles --------
radius = 0.3
margin = 0.1
center = [0.0, 0.0]
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
    np.array([-1.0, 0, 0.0, 0.0]),
]
# Define goal state
x_goal = np.array([1.5, 0.001, 0.5, 0.0])

# -------------------------------------------
# DEFINE DEVICE
# Define device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def define_dubins_expert(x0, n_steps) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a test of obstacle avoidance MPC with a dubins car and return the results"""
    # -------------------------------------------
    # Define the problem
    # -------------------------------------------
    n_states = 4
    n_controls = 2
    horizon = 80
    dt = 0.1

    # Define dynamics
    dynamics_fn = dubins_car_dynamics

    # Define obstacles
    radius = 0.3
    margin = 0.1
    center = [0.0, 0.0]
    obstacle_fns = [(lambda x: hypersphere_sdf(x, radius, [0, 1], center), margin)]

    # Define costs
    x_goal = np.array([1.5, 0.001, 0.5, 0.0])
    running_cost_fn = lambda x, u: lqr_running_cost(
        x, u, x_goal, dt * np.diag([1.0, 1.0, 0.1, 0.0]), 0.01 * np.eye(2)
    )
    terminal_cost_fn = lambda x: squared_error_terminal_cost(x, x_goal)

    # Define control bounds
    control_bounds = [(-1, 1),
                      (-0.5, 0.5)]  # [m/s^2, rad/s]

    # Make sure that the initial state is np array
    if isinstance(x0, torch.Tensor):
        x0 = x0.numpy()
    elif isinstance(x0, np.ndarray):
        pass
    else:
        raise ValueError("x0 must be either a torch.Tensor or a np.ndarray")


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
    # Simulate and return the results
    # -------------------------------------------
    _,x,u = simulate_mpc(
        opti,
        x0_variables,
        u0_variables,
        x0,
        dt,
        dynamics_fn,
        n_steps,
        verbose=False,
        x_variables=x_variables,
        u_variables=u_variables,
    )
    return torch.from_numpy(x[:-1,:]), torch.from_numpy(u)



def clone_dubins_barrier_preferences(train=True, path = None):
    # Define Barrier Function
    x_obstacle = torch.tensor([center]).to(device)
    x_g = torch.tensor(x_goal).to(device)
    r_obstacle = torch.tensor([radius + margin]).to(device)

    # -------------------------------------------
    # Clone the MPC policy
    # -------------------------------------------
    mpc_expert = lambda x0, hor: define_dubins_expert(x0, n_steps=hor)
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
        load_from_file=path,
    )

    n_pts = int(0.7e4)
    n_epochs = 500
    learning_rate = 0.001

    # Define Training optimizer
    if train:
        checkpoint = path
        cloned_policy.clone(
            mpc_expert,
            n_pts,
            n_epochs,
            learning_rate,
            batch_size=64,
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
            verbose=True,
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

    ax.set_xlim([-3., 2.5])
    ax.set_ylim([-1.0, 1.0])
    ax.title.set_text("Cloned Dubins Car Policy with UMNN BarrierNet")
    ax.grid()
    ax.set_aspect("equal")

    ax.legend()
    # Save the figure in vector format using time stamp as name
    dir = os.path.dirname(__file__)
    name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file = "..\\figures\\" + name + "_dubins_cloned_barriernet_monotonic_policy.pdf"
    path = os.path.join(dir, file)
    plt.savefig(path)
    plt.show()

if __name__ == "__main__":
    # Extract current folder
    dir = os.path.dirname(__file__)
    # Define a file with the current date
    name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file = "..\\data\\" + name + "_dubins_cloned_barriernet_monotonic_policy.pt"
    path = os.path.join(dir, file)

    path = "G:\\My Drive\\PhD\\Research\\CODES\\GameTheory\\restructured\\data\\2023-02-09_11-15-11_dubins_cloned_barriernet_monotonic_policy.pt"
    # Define the policy
    policy = clone_dubins_barrier_preferences(train=False, path= path)
    simulate_and_plot(policy)