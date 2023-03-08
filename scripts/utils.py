import torch
import numpy as np

from typing import Callable, Tuple, List, Optional


# -------------------------------------------
# DEFINE LIE DERIVATIVE AUTODIFF FUNCTIONS
def compute_lie_derivative_2nd_order(x: torch.Tensor, barrier_fun: Callable, alpha_fun_1: Callable, alpha_fun_2: Callable, f, g):
        """Compute the Lie derivative of the CBF wrt the dynamics"""
        # Make sure the input requires gradient
        x.requires_grad_(True)
        # Compute the CBF
        psi0 = barrier_fun(x)
        db_dx = torch.autograd.grad(psi0, x, create_graph=True, retain_graph=True)[0]
        Lfb = db_dx @ f(x)
        db2_dx = torch.autograd.grad(Lfb, x, retain_graph=True)[0]
        Lf2b = db2_dx @ f(x)
        LgLfb = db2_dx @ g
        psi1 = Lfb + alpha_fun_1(psi0)
        psi1_dot = torch.autograd.grad(psi1, x, retain_graph=True)[0] @ f(x)
        psi2 = psi1_dot + alpha_fun_2(psi1)
        # Compute the Lie derivative
        return LgLfb, Lf2b + psi2, psi1
    
def compute_lie_derivative_1st_order(x: torch.Tensor, barrier_fun: Callable, alpha_fun_1: Callable, f, g):
        """Compute the Lie derivative of the CBF wrt the dynamics"""
        # Make sure the input requires gradient
        x.requires_grad_(True)
        # Compute the CBF
        psi0 = barrier_fun(x)
        db_dx = torch.autograd.grad(psi0, x, create_graph=True, retain_graph=True)[0]
        Lfb = db_dx @ f(x)
        Lgb = db_dx @ g
        psi1 = Lfb + alpha_fun_1(psi0)
        
        # Compute the Lie derivative
        return Lgb, psi1, psi0

def simulate_and_plot(policies, x_goal, center, radius, margin, n_states, n_controls, state_space, dynamics_fn, dt:Optional[float]=0.1 , x0s: Optional[List[np.ndarray]]=None):
    # -------------------------------------------
    # Simulate the cloned policy
    # -------------------------------------------
    fig = plt.figure(figsize=plt.figaspect(1.0))
    fig_u, (ax_u, ax_psi) = plt.subplots(2, 1, sharey=True)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot([], [], "ro", label="Start")

    if x0s is None:
        # Generate random initial states
        x0s = []    
        for i in range(20):
            x0s.append(np.zeros(n_states))
            for dim in range(n_states - 1):
                x0s[i][dim] = np.random.uniform(state_space[dim][0], state_space[dim][1])

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
    
# ___________________________________________
# DEFINE EXPERT POLICY FOR DATA GENERATION
# ___________________________________________
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
import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, Tuple, List, Optional
from src.mpc import simulate_barriernet, simulate_mpc
from src.mpc.dynamics_constraints import car_2d_dynamics as dubins_car_dynamics
from src.mpc import lqr_running_cost, squared_error_terminal_cost
from src.mpc import construct_MPC_problem
from src.mpc import hypersphere_sdf
# -------------------------------------------
# DEFINE LIE DERIVATIVE AUTODIFF FUNCTIONS
def compute_lie_derivative_2nd_order(x: torch.Tensor, barrier_fun: Callable, alpha_fun_1: Callable, alpha_fun_2: Callable, f, g):
        """Compute the Lie derivative of the CBF wrt the dynamics"""
        # Make sure the input requires gradient
        x.requires_grad_(True)
        # Compute the CBF
        psi0 = barrier_fun(x)
        db_dx = torch.autograd.grad(psi0, x, create_graph=True, retain_graph=True)[0]
        Lfb = db_dx @ f(x)
        db2_dx = torch.autograd.grad(Lfb, x, retain_graph=True)[0]
        Lf2b = db2_dx @ f(x)
        LgLfb = db2_dx @ g
        psi1 = Lfb + alpha_fun_1(psi0)
        psi1_dot = torch.autograd.grad(psi1, x, retain_graph=True)[0] @ f(x)
        psi2 = psi1_dot + alpha_fun_2(psi1)
        # Compute the Lie derivative
        return LgLfb, Lf2b + psi2, psi1
    
def compute_lie_derivative_1st_order(x: torch.Tensor, barrier_fun: Callable, alpha_fun_1: Callable, f, g):
        """Compute the Lie derivative of the CBF wrt the dynamics"""
        # Make sure the input requires gradient
        x.requires_grad_(True)
        # Compute the CBF
        psi0 = barrier_fun(x)
        db_dx = torch.autograd.grad(psi0, x, create_graph=True, retain_graph=True)[0]
        Lfb = db_dx @ f(x)
        Lgb = db_dx @ g
        psi1 = Lfb + alpha_fun_1(psi0)
        
        # Compute the Lie derivative
        return Lgb, psi1, psi0

def simulate_and_plot(policies, x_goal, center, radius, margin, n_states, n_controls, state_space, dynamics_fn, dt:Optional[float]=0.1 , x0s: Optional[List[np.ndarray]]=None):
    # -------------------------------------------
    # Simulate the cloned policy
    # -------------------------------------------
    fig = plt.figure(figsize=plt.figaspect(1.0))
    fig_u, (ax_u, ax_psi) = plt.subplots(2, 1, sharey=True)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot([], [], "ro", label="Start")

    if x0s is None:
        # Generate random initial states
        x0s = []    
        for i in range(20):
            x0s.append(np.zeros(n_states))
            for dim in range(n_states - 1):
                x0s[i][dim] = np.random.uniform(state_space[dim][0], state_space[dim][1])

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
    
# ___________________________________________
# DEFINE EXPERT POLICY FOR DATA GENERATION
# ___________________________________________
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