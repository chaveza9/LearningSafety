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

from models.hocbf.barriernet import CLFBarrierNetLayer


# -------------------------------------------
# DEFINE DEVICE
# Define device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# -------------------------------------------
torch.set_default_dtype(torch.float64)
# -------------------------------------------
# PROBLEM PARAMETERS
n_states = 4
n_controls = 2
horizon = 15
dt = 0.1
# -------- Define Vehicle Dynamics --------
dynamics_fn = dubins_car_dynamics
# Define limits for state space
state_space = [(-3, -0.5),
               (-3, 3),
               (0, 2),
               (-1.0472, 1.0472)]
# Define control bounds
control_bounds = [(-2.0, 2.0),
                  (-1, 1)]
# ___________________________________________
# Define control affine dynamics
_f = lambda x: torch.vstack([x[2] * torch.cos(x[3]), x[2] * torch.sin(x[3]), torch.zeros(2, 1).to(device)])
_g = torch.vstack([torch.zeros(2, 2), torch.eye(2, 2)]).to(device)
# -------------------------------------------
def compute_parameters_clf_hocbf(parameters: torch.Tensor, x: torch.Tensor, x_goal: torch.Tensor, x_obst: torch.Tensor,
                                 r_obst: torch.Tensor, n_controls: int, n_clf: int, n_cbf: int,
                                 rel_degree: List[int], alpha: List[Callable], device: torch.device = torch.device("cpu")):
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
    LfV = torch.zeros(n_clf, 1).to(device).requires_grad_(False)
    LgV = torch.zeros(n_clf, n_controls).to(device).requires_grad_(False)
    V = torch.zeros(n_clf, 1).to(device).requires_grad_(False)
    x.requires_grad_(True)
    for clf_idx in range(n_clf):
        V_i = clf[clf_idx](x, x_goal)
        V_jac_i = torch.autograd.grad(V_i, x, create_graph=True)[0]
        LfV[clf_idx, 0] = V_jac_i @ _f(x)
        LgV[clf_idx, :] = V_jac_i @ _g
        V[clf_idx, 0] = V_i

    A_clf = LgV
    b_clf = -LfV - clf_rates * V

    A_cbf = torch.zeros(n_cbf, n_controls).to(device)
    b_cbf = torch.zeros(n_cbf, 1).to(device)
    # Distance from Obstacle
    alpha1 = alpha[0]
    alpha2 = alpha[1]
    for i in range(len(x_obst)):
        p1, p2 = cbf_rates[i * 2:i * 2 + 2]
        psi0 = distance_cbf(x, x_obst[0], r_obst[0])
        db_dx = torch.autograd.grad(psi0, x, create_graph=True, retain_graph=True)[0]
        Lfb = db_dx @ _f(x)
        db2_dx = torch.autograd.grad(Lfb, x, retain_graph=True)[0]
        Lf2b = db2_dx @ _f(x)
        LgLfb = db2_dx @ _g
        psi1 = Lfb + alpha1(psi0, p1)
        psi1_dot = torch.autograd.grad(psi1, x, create_graph=True)[0] @ _f(x)
        psi2 = psi1_dot + alpha2(psi1, p2)
        A_cbf[i, :n_controls] = -LgLfb.detach()
        b_cbf[i] = Lf2b + psi2
    # Velocity Bounds Barrier
    b_v = torch.vstack([v - 0.2, 2. - v])
    Lgb_v = torch.tensor([1, -1]).to(device)
    n_dist = len(x_obst)
    cbf_rates_v = cbf_rates[sum(rel_degree[:n_dist]):]

    A_cbf[n_dist:, 0] = -Lgb_v
    b_cbf[n_dist:] = cbf_rates_v * b_v

    return A_clf, b_clf, A_cbf, b_cbf, cntrl_weights


def simulate_and_plot(policies, x_goal, center, radius, margin):
    # -------------------------------------------
    # Simulate the cloned policy
    # -------------------------------------------
    fig = plt.figure(figsize=plt.figaspect(1.0))
    fig_u, (ax_u, ax_psi) = plt.subplots(2, 1, sharey=True)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot([], [], "ro", label="Start")

    # Generate random initial states
    x0s = [
        np.array([-2.0, 0.1, 0.0, 0.0]),
        np.array([-2.0, 0.1, 0.0, 0.0]),
        np.array([-2.0, 0.2, 0.0, 0.0]),
        np.array([-2.0, 0.5, 0.0, 0.0]),
        np.array([-2.0, -0.1, 0.0, 0.0]),
        np.array([-2.0, -0.2, 0.0, 0.0]),
        np.array([-2.0, -0.5, 0.0, 0.0]),

    ]
    x0s = []
    for i in range(20):
        x0s.append(np.zeros(n_states))
        for dim in range(n_states - 1):
            x0s[i][dim] = np.random.uniform(state_space[dim][0], state_space[dim][1])

    n_steps = 100
    count = 0
    colors_with_type = ["r", "g", "b", "c", "m", "y", "k", "w"]
    for policy in policies:
        for x0 in x0s:
            # Run the cloned policy
            t, x, u = simulate_barriernet(
                policy,
                x0,
                n_states,
                n_controls,
                dt,
                dynamics_fn,
                n_steps,
                substeps=10,
                x_goal=x_goal,
            )

            # Plot it
            ax.plot(x0[0], x0[1], colors_with_type[count] + "o")
            ax.plot(x[:, 0], x[:, 1], color=colors_with_type[count], linestyle="dotted")
            ax_u.plot(t[:-1], u[:, 0], color=colors_with_type[count], linestyle="dotted")
            ax_psi.plot(t[:-1], u[:, 1], color=colors_with_type[count], linestyle="dotted")
        count += 1

    # Plot obstacle
    theta = np.linspace(0, 2 * np.pi, 100)
    obs_x = radius * np.cos(theta) + center[0]
    obs_y = radius * np.sin(theta) + center[1]
    margin_x = (radius + margin) * np.cos(theta) + center[0]
    margin_y = (radius + margin) * np.sin(theta) + center[1]
    ax.plot(obs_x, obs_y, "k-", label="Obstacle")
    ax.plot(margin_x, margin_y, color="k", linestyle="dashdot")
    ax.plot(x_goal[0], x_goal[1], "go", label="Goal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_xlim([-3, 10])
    ax.set_ylim([-3, 10])
    ax.grid(True, which="both")
    ax.title.set_text("HOCBF-CLF Policy Comparison")

    ax.set_aspect("equal")

    ax.legend()

    ax_psi.set_xlabel("Time [s]")
    ax_u.set_ylabel("u1 [m/s]")
    ax_psi.set_ylabel("u2 [rad/s]")
    ax_u.grid(True, which="both")
    ax_psi.grid(True, which="both")
    ax_u.title.set_text("Control Input Comparison")

    plt.show()


if __name__ == "__main__":
    # -------------------------------------------
    # Define the problem
    # -------------------------------------------
    # Define goal state
    x_goal = torch.tensor([2.5, 0.001, 0.5, 0.0]).to(device)
    # Define obstacle
    radius = 0.5
    margin = 0.1
    center = [0.0, 0.0]
    x_obstacle = torch.tensor([center]).to(device).requires_grad_(False)
    r_obstacle = torch.tensor([radius + margin]).to(device).requires_grad_(False)
    # -------------------------------------------
    # Define Number of cbf and clf constraints
    # CLF
    n_clf = 2  # Number of CLF constraints [V_speed, V_angle]
    V_speed = lambda x, x_goal: torch.square(x[2] - x_goal[2])
    V_angle = lambda x, x_goal: torch.square(
        torch.cos(x[3]) * (x[1] - x_goal[1]) - torch.sin(x[3]) * (x[0] - x_goal[0]))
    clf = [V_speed, V_angle]
    clf_slack_weight = [1., 1.]
    # CBF
    n_cbf = 3  # Number of CBF constraints [b_radius, b_v_min, b_v_max]
    n_clf_slack = n_clf  # Number of CLF slack variables
    n_cbf_slack = 1  # Number of CBF slack variables
    distance_cbf = lambda x, x_obst, radius: (x[0] - x_obst[0]) ** 2 + (x[1] - x_obst[1]) ** 2 - radius ** 2
    velocity_min_cbf = lambda x, v_min: x[2] - v_min
    velocity_max_cbf = lambda x, v_max: v_max - x[2]
    cbf = [distance_cbf, velocity_min_cbf, velocity_max_cbf]
    cbf_slack_weight = [1000.]  # Slack weights for the CBF constraints that contain slack variables
    cbf_rel_degree = [2, 1, 1]  # Relative degree of the CBFs [distance, v_min, v_max]
    # -------------------------------------------
    # Define CLFBarrierNetLayer solver
    barrier_layer = CLFBarrierNetLayer(
        n_states,
        n_controls,
        n_clf,
        clf_slack_weight=clf_slack_weight,
        n_cbf=n_cbf,
        n_cbf_slack=n_cbf_slack,
        cbf_rel_degree=cbf_rel_degree,
        control_bounds=control_bounds,
        cbf_slack_weight=cbf_slack_weight,
    )
    # -------------------------------------------
    # Define Different expert policies
    policies = []
    # Policy 1 (Conservative)
    p_weights_1 = torch.tensor([1., 1., 1, 1, 0.01, 50., 1., 1.]).to(device)
    # Linear class k functions and tanh class k functions
    alpha1 = [lambda psi, p: p * psi+p*torch.tanh(psi), lambda psi, p: p * psi]
    policy1 = lambda x_state: barrier_layer(*compute_parameters_clf_hocbf(
        p_weights_1, x_state, x_goal, x_obstacle,
        r_obstacle, n_controls, n_clf,
        n_cbf, cbf_rel_degree, alpha1, device)
                                            ).detach().squeeze()
    policies.append(policy1)
    # Policy 2 (Aggressive)
    p_weights_2 = torch.tensor([1., 1., 1, 1, 0.5, 10, 1., 1.]).to(device)
    # Linear class k functions
    alpha2 = [lambda psi, p: p * psi, lambda psi, p: p * psi]
    policy2 = lambda x_state: barrier_layer(*compute_parameters_clf_hocbf(
        p_weights_2, x_state, x_goal, x_obstacle,
        r_obstacle, n_controls, n_clf,
        n_cbf, cbf_rel_degree, alpha2, device)
                                            ).detach().squeeze()
    policies.append(policy2)
    # -------------------------------------------
    # Simulate and plot
    simulate_and_plot(policies, x_goal.numpy(), center, radius, margin)
