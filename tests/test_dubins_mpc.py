"""Test the obstacle avoidance MPC for a dubins vehicle"""
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import datetime
sys.path.append(os.path.abspath('..'))


from src.mpc import lqr_running_cost, squared_error_terminal_cost
from src.mpc.dynamics_constraints import car_2d_dynamics as dubins_car_dynamics
from src.mpc import construct_MPC_problem
from src.mpc import hypersphere_sdf
from src.mpc import simulate_mpc


def test_dubins_mpc(x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    margin = 0.2
    center = [0.0, 0.0]
    obstacle_fns = [(lambda x: hypersphere_sdf(x, radius, [0, 1], center), margin)]

    # Define costs
    x_goal = np.array([1.5, 0.0, 0.5, 0.0])
    running_cost_fn = lambda x, u: lqr_running_cost(
        x, u, x_goal, dt * np.diag([1.0, 1.0, 0.1, 0.0]), 0.01 * np.eye(2)
    )
    terminal_cost_fn = lambda x: squared_error_terminal_cost(x, x_goal)

    # Define control bounds
    control_bounds = [(-1, 1),
                      (-1, 1)]  # [m/s^2, rad/s]

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
    n_steps = 100
    return simulate_mpc(
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


def run_and_plot_dubins_mpc():
    x0s = []
    n_states = 4
    state_space = [(-3, -0.5),
                   (-1, 1),
                   (0, 2),
                   (-1.0472, 1.0472)]
    for i in range(20):
        x0s.append(np.zeros(n_states))
        for dim in range(n_states - 1):
            x0s[i][dim] = np.random.uniform(state_space[dim][0], state_space[dim][1])

    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([], [], "ro", label="Start")

    for x0 in x0s:
        # Run the MPC
        _, x, u = test_dubins_mpc(x0)

        # Plot it
        ax.plot(x0[0], x0[1], "ro")
        ax.plot(x[:, 0], x[:, 1], "r-")

    # Plot obstacle
    radius = 0.3
    margin = 0.2
    center = [0.0, 0.0]
    theta = np.linspace(0, 2 * np.pi, 100)
    obs_x = radius * np.cos(theta) + center[0]
    obs_y = radius * np.sin(theta) + center[1]
    margin_x = (radius + margin) * np.cos(theta) + center[0]
    margin_y = (radius + margin) * np.sin(theta) + center[1]
    ax.plot(obs_x, obs_y, "k-")
    ax.plot(margin_x, margin_y, "k:")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xlim([-3.5, 2])
    ax.set_ylim([-1.0, 1.0])
    ax.grid(True, which="both")
    ax.title.set_text("MPC Expert Policy Comparison")

    ax.set_aspect("equal")

    ax.legend()
    # Save the figure in vector format using time stamp as name
    dir = os.path.dirname(__file__)
    name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file = "..\\figures\\" + name + "_mpc_policy_Expert.pdf"
    path = os.path.join(dir, file)
    plt.savefig(path)

    plt.show()


if __name__ == "__main__":
    run_and_plot_dubins_mpc()


