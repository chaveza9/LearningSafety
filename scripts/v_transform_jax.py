"""Test the obstacle avoidance BarrierNetLayer for a dubins vehicle"""
from typing import Callable, Tuple, List, Dict, Optional
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpc.dynamics_constraints import car_2d_dynamics as dubins_car_dynamics
from mpc.simulator import simulate_barriernet
from functools import partial
import os, sys
# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.BarrierNet import CLFBarrierNetLayer

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
@jax.jit
def f(x):
    dxdt = jnp.zeros_like(x)
    dxdt = dxdt.at[0].set(x[2] * jnp.cos(x[3]))
    dxdt = dxdt.at[1].set(x[2] * jnp.sin(x[3]))
    return dxdt

@jax.jit
def g(x):
    return jnp.vstack([jnp.zeros((2,2)), jnp.eye(2)])


def dynamics(x, u):
    return f(x) + g(x) @ u


# -------------------------------------------
# CONSTRUCT HOCBF CONSTRAINTS DIFFERENTIABLE FUNCTIONS

def compute_parameters_clf_cbf(x: jnp.ndarray,
                               cntrl_weights: jnp.ndarray,
                               x_goal: jnp.ndarray, n_controls: int,
                               clfs: List[Dict], cbfs: List[Dict],
                               ):
    """ Compute constraint parameters for the CLF and CBF """
    # -----------------Housekeeping -------------
    # Compute the total number of CLF and CBF constraints
    n_clf = len(clfs)
    n_cbf = len(cbfs)
    # -------- Compute CLF and CBF Constraint Parameters --------

    # Compute CLF parameters
    A_clf = jnp.zeros((n_clf, n_controls))
    b_clf = jnp.zeros((n_clf, 1))
    for idx, clf in enumerate(clfs):
        V = partial(clf['f'], *clf['args'])
        alpha1 = partial(clf['alpha'][0], p=clf['rates'][0])
        # Compute CLF constraint
        G, F, _ = _compute_lie_derivative_1st_order(x, V, alpha1)
        # Populate constraint matrices
        A_clf= A_clf.at[idx, :].set(G)
        b_clf = b_clf.at[idx, :].set(-F)

    # Compute CBF parameters
    A_cbf = jnp.zeros((n_cbf, n_controls))
    b_cbf = jnp.zeros((n_cbf, 1))
    for idx, cbf in enumerate(cbfs):
        barrier_function = partial(cbf['f'], *cbf['args'])
        if (cbf["type"] == "state_constraint" or cbf["type"] == "distance_trans") and cbf['rel_degree'] == 1:
            # Create alpha functions for each relative degree
            alpha1 = partial(cbf['alpha'][0], p=cbf['rates'])
            # Compute CBF constraint
            G, F, _ = _compute_lie_derivative_1st_order(x, barrier_fun=barrier_function, alpha_fun_1=alpha1)
        else:
            raise "Error: CBF type not recognized"
        # Populate CBF constraint parameters
        A_cbf = A_cbf.at[idx, :n_controls].set(-G)
        b_cbf = b_cbf.at[idx].set(F)
    # Make sure that control weights have the correct dimension
    cntrl_weights = cntrl_weights.reshape(n_controls, 1)

    return A_clf, b_clf, A_cbf, b_cbf, cntrl_weights


# -------------------------------------------
# DEFINE LIE DERIVATIVE AUTODIFF FUNCTIONS
@partial(jax.jit, static_argnums=(0,))
def _compute_lie_derivative_1st_order(x: jnp.ndarray, barrier_fun: Callable, alpha_fun_1: Callable):
    """Compute the Lie derivative of the CBF wrt the dynamics"""
    # Compute the CBF and its gradient
    psi0, db_dx = jax.value_and_grad(barrier_fun)(x)
    # Compute the Lie derivative
    Lfb = db_dx @ f(x)
    Lgb = db_dx @ g(x)
    # Compute the CBF constraint
    psi1 = Lfb + alpha_fun_1(psi0)
    # Compute the Lie derivative
    return Lgb, psi1, psi0


def simulate_and_plot(policies, x_goal, center, radius, margin, x0s: Optional[List[jnp.ndarray]] = None):
    # -------------------------------------------
    # Simulate the cloned policy
    # -------------------------------------------
    fig = plt.figure(figsize=plt.figaspect(1.0))
    fig_u, (ax_u, ax_psi) = plt.subplots(2, 1, sharey=True)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot([], [], "ro", label="Start")
    key = jax.random.PRNGKey(0)
    if x0s is None:
        # Generate random initial states
        x0s = []
        for i in range(20):
            x0s.append(jnp.zeros((n_states,)))
            for dim in range(n_states - 1):
                x0s[i][dim] = jax.random.uniform(key, state_space[dim][0], state_space[dim][1])

    n_steps = 300
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
    theta = jnp.linspace(0, 2 * jnp.pi, 100)
    obs_x = radius * jnp.cos(theta) + center[0]
    obs_y = radius * jnp.sin(theta) + center[1]
    margin_x = (radius + margin) * jnp.cos(theta) + center[0]
    margin_y = (radius + margin) * jnp.sin(theta) + center[1]
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
    x_goal = jnp.array([2.5, 0.001, 0.5, 0.0])
    # Define obstacle
    radius = 0.5
    margin = 0.1
    center = [0.0, 0.0]
    x_obstacle = jnp.array([center])
    r_obstacle = jnp.array([radius + margin])
    # Define Different expert policies buffer
    policies = []
    # -------------------------------------------
    # -------------------------------------------
    # Define HOCBF-CLF policy
    # -------------------------------------------
    # Define Number of cbf and clf constraints
    # CLF
    n_clf = 2  # Number of CLF constraints [V_speed, V_angle]
    n_clf_slack = n_clf  # Number of CLF slack variables
    clf_slack_weight = [1., 1.]

    V_speed = {'type': 'des_speed',
               'f': lambda x_goal, x: jnp.square(x[2] - x_goal[2]),
               'args': [x_goal],
               'rel_degree': 1,
               'rates': [1.0],
               'alpha': [lambda x, p: p * x]}
    V_angle = {'type': 'des_angle',
               'f': lambda x_goal, x: jnp.square(
                   jnp.cos(x[3]) * (x[1] - x_goal[1]) - jnp.sin(x[3]) * (x[0] - x_goal[0])),
               'args': [x_goal],
               'rel_degree': 1,
               'rates': [1.0],
               'alpha': [lambda x, p: p * x]}
    clf = [V_speed, V_angle]

    # CBF
    n_cbf = 3  # Number of CBF constraints [b_radius, b_v_min, b_v_max]
    cbf_rel_degree = [1, 1, 1]
    cbf_slack_weight = [1000, 0, 0]

    cv = 1 / 6 * jnp.pi
    av_p = jnp.array([0.21, 3.5])
    aV = lambda x, p: p[0] * jnp.square(x[3]) + p[1] * jnp.square(x[3] + cv)
    distance_cbf = {'type': 'distance_trans',
                    'f': lambda x_obst, radius, x: jnp.squeeze(jnp.square(x[0] - x_obst[0, 0]) + jnp.square(
                            x[1] - x_obst[0, 1]) - jnp.square(radius) - aV(x, av_p)),
                    'args': [x_obstacle, r_obstacle],
                    'rel_degree': 1,
                    'slack': True,
                    'slack_weight': 1000,
                    'alpha': [lambda psi, p: p[0] * psi],
                    'rates': [0.5]}
    vel_min_cbf = {'type': 'state_constraint',
                   'f': lambda v_min, x: x[2] - v_min,
                   'args': [0.1],
                   'rel_degree': 1,
                   'slack': False,
                   'rates': 1.0,
                   'alpha': [lambda psi, p: p * psi]}
    vel_max_cbf = {'type': 'state_constraint',
                   'f': lambda v_max, x: v_max - x[2],
                   'args': [1.0],
                   'rel_degree': 1,
                   'slack': False,
                   'rates': 1.0,
                   'alpha': [lambda psi, p: p * psi]}
    cbf = [distance_cbf, vel_min_cbf, vel_max_cbf]
    # -------------------------------------------
    # Define HOCBF-CLFBarrierNetLayer solver
    av_cbf_barrier_layer = CLFBarrierNetLayer(
        n_state_dims=n_states,
        n_control_dims=n_controls,
        n_clf_constraints=n_clf,
        clf_slack_penalty=clf_slack_weight,
        n_cbf_constraints=n_cbf,
        cbf_slack_penalty=cbf_slack_weight,
        cbf_relative_degree=cbf_rel_degree,
        control_bounds=control_bounds,
        state_bounds=state_space,
    )
    # -------------------------------------------
    # Policy (Aggressive)
    # Linear class k fuctions only
    cntrl_weights = jnp.array([1., 1.])  # Same weights for all policies
    parameters_func = partial(compute_parameters_clf_cbf, cntrl_weights=cntrl_weights, x_goal=x_goal,
                              n_controls=n_controls, clfs=clf, cbfs=cbf)

    # Initialize the policy
    params = av_cbf_barrier_layer.init(jax.random.PRNGKey(0), *parameters_func(jnp.zeros(n_states)))

    # Append to the list of policies
    policies.append(lambda x_state: av_cbf_barrier_layer.apply(params, *parameters_func(x_state)))
    # -------------------------------------------
    # -------------------------------------------
    # Plot Policies
    # -------------------------------------------
    # Define initial conditions
    x0s = [
        jnp.array([-2.0, 0.1, 0.0, 0.0]),
        jnp.array([-2.0, 0.1, 0.0, 0.0]),
        jnp.array([-2.0, 0.2, 0.0, 0.0]),
        jnp.array([-2.0, 0.5, 0.0, 0.0]),
        jnp.array([-2.0, -0.1, 0.0, 0.0]),
        jnp.array([-2.0, -0.2, 0.0, 0.0]),
        jnp.array([-2.0, -0.5, 0.0, 0.0]),

    ]
    # Simulate and plot
    simulate_and_plot(policies, x_goal, center, radius, margin, x0s)
