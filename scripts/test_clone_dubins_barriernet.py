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
from mpc.simulator import simulate_nn

from Models.mlpbarriernet import MLPBarrierNet
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
control_bounds = [(-1, 1), # [m/s^2]
                  (-0.5, 0.5) ] # [rad/s]
torch.set_default_dtype(torch.float64)
# -------- Define Number of cbf and clf constraints --------
n_clf = 2  # Number of CLF constraints [V_speed, V_angle]
n_cbf = 3  # Number of CBF constraints [b_radius, b_v_min, b_v_max]
n_clf_slack = n_clf  # Number of CLF slack variables
n_cbf_slack = 0  # Number of CBF slack variables
clf_slack_weight = [1., 1.]
cbf_slack_weight = [1000.]
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
    max_tries = 100
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

def compute_parameters_hocbf(cbf_rates:torch.Tensor, clf_rates:torch.Tensor, x: torch.Tensor, x_goal: torch.Tensor, x_obst: torch.Tensor,
                             r_obst: torch.Tensor, n_controls: int, n_clf: int, n_cbf: int,
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
    cntrl_weights = torch.ones((n_controls, 1))
    # Extract the CLF rates from the NN
    # clf_rates = torch.reshape(parameters[n_clf], (n_clf, 1))
    clf_rates = torch.reshape(clf_rates, (n_clf, 1))
    clf_rates = torch.ones((n_clf, 1))
    # Extract the CBF rates from the NN
    # cbf_rates = torch.reshape(parameters[n_clf:n_clf + n_cbf_rates], (n_cbf_rates, 1))
    cbf_rates = torch.reshape(cbf_rates, (n_cbf_rates, 1))
    # -------- Compute CLF and CBF Constraint Parameters --------
    # Compute CLF parameters
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


    return A_clf, b_clf, A_cbf, b_cbf, cntrl_weights


def clone_dubins_barrier_preferences(train=True, load=False):
    # Define Barrier Function
    x_obstacle = torch.tensor([center]).to(device)
    x_g = torch.tensor(x_goal).to(device)
    r_obstacle = torch.tensor([radius + margin]).to(device)

    process_barrier_inputs = lambda x_state, cbf_rates, clf_rates: compute_parameters_hocbf(cbf_rates, clf_rates,
                                                                                        x_state, x_g, x_obstacle,
                                                                                        r_obstacle, n_controls, n_clf,
                                                                                        n_cbf, rel_degree, device)
    # -------------------------------------------
    # Clone the MPC policy
    # -------------------------------------------
    mpc_expert = define_dubins_mpc_expert()
    hidden_layers = 4
    hidden_layer_width = 32
    
    cloned_policy = MLPBarrierNet(
        hidden_layers= hidden_layers,
        hidden_layer_width= hidden_layer_width,
        n_state_dims= n_states,
        n_control_dims= n_controls,
        n_input_dims= n_states+len(x_obstacle[0])+len(x_g),
        n_clf= n_clf,
        clf_slack_weight= clf_slack_weight,
        n_cbf= n_cbf, # Ordered list of Barrier functions
        n_cbf_slack= n_cbf_slack,
        cbf_slack_weight= cbf_slack_weight,
        cbf_rel_degree= rel_degree,
        state_space= state_space,
        control_bounds= control_bounds,
        preprocess_barrier_input_fn= process_barrier_inputs
    )

    n_pts = int(0.1e4)
    n_epochs = 300
    learning_rate = 0.01
    path = "./data/cloned_dubins_barrier_policy_weight_decay_no_mlp_constant_clf.pt"
    # Define Training optimizer
    if train and not load:
        cloned_policy.clone(
            mpc_expert,
            n_pts,
            n_epochs,
            learning_rate,
            batch_size=32,
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