"""Test the obstacle avoidance BarrierNetLayer for a dubins vehicle"""
from typing import Callable, Tuple, List, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
import copy


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
# Define limits for state space
state_space = [(-3, -0.5),
               (-3, 3),
               (0, 2),
               (-1.0472, 1.0472)]
# Define control bounds
control_bounds = [(-2.0, 2.0),
                  (-1, 1)]
# ___________________________________________
dynamics_fn = dubins_car_dynamics
# Define control affine dynamics
_f = lambda x: torch.vstack([x[2] * torch.cos(x[3]), x[2] * torch.sin(x[3]), torch.zeros(2, 1).to(device)])
_g = torch.vstack([torch.zeros(2, 2), torch.eye(2, 2)]).to(device)
# -------------------------------------------

# Define a general CBF data structure        
def compute_parameters_clf_hocbf(x: torch.Tensor, 
                                 cntrl_weights: torch.Tensor, 
                                 x_goal: torch.Tensor, n_controls: int, 
                                 clfs: List[Dict], cbfs: List[Dict], 
                                 device: torch.device = torch.device("cpu")):
    """ Compute constraint parameters for the CLF and CBF """
    # -----------------Housekeeping -------------
    # Make sure that vector is in the correct device
    x = torch.from_numpy(x).to(device)
    # Compute the total number of CLF and CBF constraints
    n_clf = len(clfs)
    n_cbf = len(cbfs)
    # -------- Compute CLF and CBF Constraint Parameters --------
    
    # Compute CLF parameters
    A_clf = torch.zeros(n_clf, n_controls).to(device)
    b_clf = torch.zeros(n_clf, 1).to(device)
    for idx, clf in enumerate(clfs):
        V = lambda x: clf['f'](x, x_goal)
        alpha1 = lambda x: clf['alpha'][0](x, clf['rates'][0])
        # Compute CLF constraint
        G, F, _ = _compute_lie_derivative_1st_order(x, V, alpha1)
        # Populate constraint matrices
        A_clf[idx] = G
        b_clf[idx] = -F
        
    # Compute CBF parameters
    A_cbf = torch.zeros(n_cbf, n_controls).to(device)
    b_cbf = torch.zeros(n_cbf, 1).to(device)
    for idx, cbf in enumerate (cbfs):
        barrier_function = lambda x: cbf['f'] (x, *cbf['args'])
        if cbf["type"] == "distance" and cbf['rel_degree'] == 2:
            # Extract Rates
            p1, p2 = cbf['rates']
            # Create alpha functions for each relative degree
            alpha1 = lambda x: cbf['alpha'][0](x, p1)
            alpha2 = lambda x: cbf['alpha'][1](x, p2)
            # Compute CBF constraint
            G, F, _ = _compute_lie_derivative_2nd_order(x, barrier_fun= barrier_function, alpha_fun_1= alpha1, alpha_fun_2= alpha2)
        elif cbf["type"] == "state_constraint" and cbf['rel_degree'] == 1:
            p = cbf['rates']
            # Create alpha functions for each relative degree
            alpha1 = lambda x: cbf['alpha'][0](x, p)
            # Compute CBF constraint
            G, F, _ = _compute_lie_derivative_1st_order(x, barrier_fun= barrier_function, alpha_fun_1= alpha1)
        else :
            raise "Error: CBF type not recognized"
        # Populate CBF constraint parameters
        A_cbf[idx, :n_controls] = -G.detach()
        b_cbf[idx] = F.detach()
        # Make sure that control weights have the correct dimension
        cntrl_weights = cntrl_weights.reshape(n_controls,1)

    return A_clf, b_clf, A_cbf, b_cbf, cntrl_weights

def _compute_lie_derivative_2nd_order(x: torch.Tensor, barrier_fun: Callable, alpha_fun_1: Callable, alpha_fun_2: Callable):
        """Compute the Lie derivative of the CBF wrt the dynamics"""
        # Make sure the input requires gradient
        x.requires_grad_(True)
        # Compute the CBF
        psi0 = barrier_fun(x)
        db_dx = torch.autograd.grad(psi0, x, create_graph=True, retain_graph=True)[0]
        Lfb = db_dx @ _f(x)
        db2_dx = torch.autograd.grad(Lfb, x, retain_graph=True)[0]
        Lf2b = db2_dx @ _f(x)
        LgLfb = db2_dx @ _g
        psi1 = Lfb + alpha_fun_1(psi0)
        psi1_dot = torch.autograd.grad(psi1, x, retain_graph=True)[0] @ _f(x)
        psi2 = psi1_dot + alpha_fun_2(psi1)
        # Compute the Lie derivative
        return LgLfb, Lf2b + psi2, psi1
    
def _compute_lie_derivative_1st_order(x: torch.Tensor, barrier_fun: Callable, alpha_fun_1: Callable):
        """Compute the Lie derivative of the CBF wrt the dynamics"""
        # Make sure the input requires gradient
        x.requires_grad_(True)
        # Compute the CBF
        psi0 = barrier_fun(x)
        db_dx = torch.autograd.grad(psi0, x, create_graph=True, retain_graph=True)[0]
        Lfb = db_dx @ _f(x)
        Lgb = db_dx @ _g
        psi1 = Lfb + alpha_fun_1(psi0)
        
        # Compute the Lie derivative
        return Lgb, psi1, psi0

def simulate_and_plot(policies, x_goal, center, radius, margin):
    # -------------------------------------------
    # Simulate the cloned policy
    # -------------------------------------------
    fig = plt.figure(figsize=plt.figaspect(1.0))
    fig_u, (ax_u, ax_psi) = plt.subplots(2, 1, sharey=True)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot([], [], "ro", label="Start")

    # Generate random initial states
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
                verbose=True,
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
    n_clf_slack = n_clf  # Number of CLF slack variables
    clf_slack_weight = [1., 1.]
    
    V_speed = {'type': 'des_speed', 
               'f': lambda x, x_goal: torch.square(x[2] - x_goal[2]),
               'rel_degree': 1,
               'rates': [1.0],
               'alpha': [lambda x, p: p*x]}
    V_angle = {'type': 'des_angle',
               'f': lambda x, x_goal:torch.square(
                    torch.cos(x[3]) * (x[1] - x_goal[1]) - torch.sin(x[3]) * (x[0] - x_goal[0])),
               'rel_degree': 1,
               'rates': [1.0],
               'alpha': [lambda x, p: p*x]}
    clf = [V_speed, V_angle]
    
    # CBF
    n_cbf = 3  # Number of CBF constraints [b_radius, b_v_min, b_v_max]
    n_cbf_slack = 1  # Number of CBF slack variables
    cbf_rel_degree = [2, 1, 1]
    cbf_slack_weight = [1000]
    
    distance_cbf = {'type': 'distance',
                    'f': lambda x, x_obst, radius: (x[0] - x_obst[0, 0]) ** 2 + (x[1] - x_obst[0,1]) ** 2 - radius ** 2,
                    'args': [x_obstacle, r_obstacle], 
                    'rel_degree': 2,
                    'slack': True,
                    'slack_weight': 1000}
    vel_min_cbf = {'type': 'state_constraint',
                   'f': lambda x, v_min: x[2] - v_min,
                   'args': [0.1],
                   'rel_degree': 1,
                   'slack': False,
                   'rates': 1.0,
                   'alpha': [lambda psi, p: p * psi]}
    vel_max_cbf = {'type': 'state_constraint',
                   'f': lambda x, v_max: v_max - x[2],
                   'args': [1.0],
                   'rel_degree': 1,
                   'slack': False,
                   'rates': 1.0,
                   'alpha': [lambda psi, p: p * psi]}
    cbf = [distance_cbf, vel_min_cbf, vel_max_cbf]
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
    cntrl_weights = torch.tensor([1., 1.]).to(device) # Same weights for all policies
    # Policy 1 (Conservative) 
    # Linear class k functions and tanh class k functions
    clf_1 = copy.deepcopy(clf)
    cbf_1 = copy.deepcopy(cbf)
    # distance cbf
    cbf_1[0]['alpha'] = [lambda psi, p: p * psi+p*torch.tanh(psi), lambda psi, p: p * psi]
    cbf_1[0]['rates'] = [0.1, 10]
    policies.append(lambda x_state: barrier_layer(*compute_parameters_clf_hocbf(x_state, 
                                 cntrl_weights= cntrl_weights, 
                                 x_goal = x_goal, 
                                 n_controls= n_controls,
                                 clfs= clf_1, cbfs= cbf_1,  
                                 device= device)).detach().squeeze().cpu()
                    )
    
    # Policy 2 (Aggressive)
    # Linear class k fuctions only
    p_weights_2 = torch.tensor([1., 1., 1, 1, 0.5, 10, 1., 1.]).to(device)
    # Linear class k functions
    clf_2 = copy.deepcopy(clf)
    cbf_2 = copy.deepcopy(cbf)
    # distance cbf
    cbf_2[0]['alpha'] = [lambda psi, p: p * psi, lambda psi, p: p * psi]
    cbf_2[0]['rates'] = [0.5, 10]
    policies.append(lambda x_state: barrier_layer(*compute_parameters_clf_hocbf(x_state,
                                 cntrl_weights= cntrl_weights,
                                 x_goal = x_goal,
                                 n_controls= n_controls,
                                 clfs= clf_2, cbfs= cbf_2,
                                 device= device)).detach().squeeze().cpu()
                    )
    # -------------------------------------------
    # Simulate and plot
    simulate_and_plot(policies, x_goal.cpu().numpy(), center, radius, margin)
