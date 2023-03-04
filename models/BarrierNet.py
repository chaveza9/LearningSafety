import jax
from typing import Any, Callable, Sequence, Dict, List, Tuple, Optional
from jax import lax, random, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn
from collections import OrderedDict

from cvxpylayers.jax import CvxpyLayer

import cvxpy as cp
import jax.numpy as jnp
from tqdm import tqdm
import warnings


class CBFBarrierNetLayer(nn.Module):
    """
    A simple Differential QP barrier function neural network.
    """
    # State and Control Dimensions
    n_state_dims: int
    n_control_dims: int
    # State and Control Bounds
    state_bounds: List[float]
    control_bounds: List[float]
    # Barrier Parameters
    n_cbf_constraints: int
    cbf_slack_penalty: List[float]
    cbf_relative_degree: List[int]

    # cbf_layer: CvxpyLayer

    # Define the control Barrier function variables
    def setup(self):
        # Check if parameters are valid and properly sized
        if len(self.cbf_relative_degree) != self.n_cbf_constraints:
            raise ValueError("cbf_relative_degree lenght must be the same length as n_cbf_constraints. \
                It should be a list of integers.")
        if len(self.cbf_slack_penalty) != self.n_cbf_constraints:
            raise ValueError("cbf_slack_penalty lenght must be the same length as n_cbf_constraints. \
                    It should be a list of floats. Set to 0.0 if you don't want to penalize the slack.")
        # Extract weights for the slack variables based on penalties greater than 0
        penalty_weights = jnp.asarray(self.cbf_slack_penalty)
        weights = penalty_weights.at[penalty_weights > 0].get()
        # Define the cbf cvxpy layer
        self.cbf_layer = self._define_cbf_diff_qp(weights)

    def __call__(self, u_ref: jnp.ndarray, A_cbf: jnp.ndarray, b_cbf: jnp.ndarray, **kwargs):
        """ Computes the control input using a differential QP."""
        # Compute the safety filter
        solver_opts = {"solve_method": "ECOS", "verbose": False, "max_iters": 50000000}
        try:
            u, _ = self.cbf_layer(u_ref, A_cbf, b_cbf, solver_args=solver_opts)
        except Exception as e:
            warnings.warn("Error: {}".format(e))
            u = jnp.zeros_like(u_ref)
        return u

    def _define_cbf_diff_qp(self, slack_weights: Optional[jnp.ndarray] = None) -> CvxpyLayer:
        """ Defines a differential cvxpy layer that contains a HOCBF filter."""
        # -------- Define Optimization Variables --------
        u = cp.Variable((self.n_control_dims, 1))  # control input
        # Define reference input
        u_ref = cp.Parameter((self.n_control_dims, 1))
        # Define the control Barrier function variables
        n_cbf_slack = (jnp.asarray(self.cbf_slack_penalty) > 0).sum()
        if self.n_cbf_constraints > 1 and n_cbf_slack > 0:
            slack_cbf = cp.Variable((n_cbf_slack, 1))
        else:
            slack_cbf = jnp.zeros((self.n_cbf_constraints, 1))
        # Create a list of slack variables
        if slack_cbf.shape[0] != self.n_cbf_constraints:
            delta_cbf = cp.vstack([slack_cbf, jnp.zeros((self.n_cbf_constraints - n_cbf_slack, 1))])
        else:
            delta_cbf = slack_cbf
        # -------- Define Differentiable Variables --------
        # Define Constraint Parameters CBF
        A_cbf = cp.Parameter((self.n_cbf_constraints, self.n_control_dims))  # CBF matrix
        b_cbf = cp.Parameter((self.n_cbf_constraints, 1))  # CBF vector
        # Define Objective Parameters
        # Add weights to the slack variables CLF
        if slack_weights is not None:
            n_weights = len(slack_weights)
            if n_weights != slack_weights:
                raise "Error: the number of slack weights must be equal to the number of slack variables"
            weights = cp.vstack([slack_weights, jnp.array(slack_weights).reshape(n_cbf_slack, 1)])
            objective_slack = cp.sum(cp.multiply(weights, slack_cbf))
        else:
            objective_slack = jnp.zeros((1, 1))
        # Compute the objective function
        objective = 0.5 * cp.Minimize(cp.sum(cp.square(u - u_ref)) + objective_slack)
        # -------- Define Constraints --------
        contraints = []
        # Define the constraints CBF
        contraints += [A_cbf @ u <= b_cbf + delta_cbf]
        # Define the constraints on the control input
        for control_idx, bound in enumerate(self.control_bounds):
            contraints += [u[control_idx] <= bound[1]]
            contraints += [u[control_idx] >= bound[0]]
        # -------- Define the Problem --------
        problem = cp.Problem(objective, contraints)
        if self.n_cbf_slack > 0:
            return CvxpyLayer(problem, parameters=[u_ref, A_cbf, b_cbf], variables=[u, slack_cbf])
        else:
            return CvxpyLayer(problem, parameters=[u_ref, A_cbf, b_cbf], variables=[u])


class CLFBarrierNetLayer(CBFBarrierNetLayer):
    """
    A simple Differential QP barrier function neural network.
    """
    # CLF Parameters
    n_clf_constraints: int
    clf_slack_penalty: List[float]
    # Solver Parameters
    cbf_clf_struc: str = "single"  # single QP or double QP

    # clf_layer: CvxpyLayer
    # clf_cbf_layer: CvxpyLayer

    def setup(self):
        # Extract weights for the slack variables based on penalties greater than 0
        cbf_penalty_weights = jnp.array(self.cbf_slack_penalty)
        cbf_penalty_weights = cbf_penalty_weights.at[cbf_penalty_weights > 0].get()
        clf_penalty_weights = jnp.array(self.clf_slack_penalty)
        weights = jnp.hstack((clf_penalty_weights, cbf_penalty_weights))
        # Define the cbf cvxpy layer based on user defined structure
        self.clf_layer = self._define_clf_diff_qp()
        self.clf_cbf_layer = self._define_clf_cbf_diff_qp(weights)

    def __call__(self, A_clf: jnp.ndarray, b_clf: jnp.ndarray, A_cbf: jnp.ndarray, b_cbf: jnp.ndarray,
                 cntrl_weights: jnp.ndarray, **kwargs):
        solver_opts = {"solve_method": "ECOS", "verbose": False, "max_iters": 50000000}
        try:
            if self.cbf_clf_struc == "single":
                u = self.clf_cbf_layer(A_clf, b_clf,
                                       A_cbf, b_cbf,
                                       cntrl_weights,
                                       solver_args=solver_opts)[0]
            elif self.cbf_clf_struc == "double":
                u_ref = self.clf_layer(A_clf, b_clf, cntrl_weights, solver_args=solver_opts)
                u = super().__call__(u_ref, A_cbf, b_cbf)
        except Exception as e:
            warnings.warn("Error: {}".format(e))
            u = jnp.zeros(self.n_control_dims)
        return u

    def _define_clf_diff_qp(self) -> CvxpyLayer:
        """ Defines a differential cvxpy layer that contains a CLF controller."""
        # -------- Define Optimization Variables --------
        u = cp.Variable((self.n_control_dims, 1))  # control input
        # -------- Define Differentiable Variables --------
        # Define Constraint Parameters CLF
        A_clf = cp.Parameter((self.n_clf_constraints, self.n_control_dims))  # CLF matrix
        b_clf = cp.Parameter((self.n_clf_constraints, 1))  # CLF vector
        # Define Objective Parameters
        cntrl_weights = cp.Parameter((self.n_control_dims, 1), nonneg=True)  # Q_tunning matrix for the control input
        # Compute the objective function
        objective = 0.5 * cp.Minimize(cp.sum(cp.multiply(cntrl_weights, cp.square(u))))
        # -------- Define Constraints --------
        contraints = []
        # Define the constraints CLF
        contraints += [A_clf @ u <= b_clf]
        # Define the constraints on the control input
        for control_idx, bound in enumerate(self.control_bounds):
            contraints += [u[control_idx] <= bound[1]]
            contraints += [u[control_idx] >= bound[0]]
        # -------- Define the Problem --------
        problem = cp.Problem(objective, contraints)
        return CvxpyLayer(problem, parameters=[A_clf, b_clf, cntrl_weights], variables=[u])

    def _define_clf_cbf_diff_qp(self, slack_weights: Optional[jnp.ndarray] = None) -> CvxpyLayer:
        """ Defines a differential cvxpy layer that contains a HOCBF filter together with a CLF controller."""
        # -------- Define Optimization Variables --------
        u = cp.Variable((self.n_control_dims, 1))  # control input
        # Define the control Lyapunov function variables
        slack_clf = cp.Variable((self.n_clf_constraints, 1))  # slack variables
        # Define the control Barrier function variables
        n_cbf_slack = int((jnp.asarray(self.cbf_slack_penalty) > 0).sum())
        if self.n_cbf_constraints > 1 and n_cbf_slack > 0:
            slack_cbf = cp.Variable((n_cbf_slack, 1))
        else:
            slack_cbf = jnp.zeros((self.n_cbf_constraints, 1))
        # Create a list of slack variables
        if n_cbf_slack > 0:
            slack = cp.vstack([slack_clf, slack_cbf])
        else:
            slack = slack_clf
        if slack_cbf.shape[0] != self.n_cbf_constraints:
            delta_cbf = cp.vstack([slack_cbf, jnp.zeros((self.n_cbf_constraints - n_cbf_slack, 1))])
        else:
            delta_cbf = slack_cbf
        X = cp.vstack([u, slack])
        # -------- Define Differentiable Variables --------
        # Define Constraint Parameters CLF
        A_clf = cp.Parameter((self.n_clf_constraints, self.n_control_dims))  # CLF matrix
        b_clf = cp.Parameter((self.n_clf_constraints, 1))  # CLF vector
        # Define Constraint Parameters CBF
        A_cbf = cp.Parameter((self.n_cbf_constraints, self.n_control_dims))  # CBF matrix
        b_cbf = cp.Parameter((self.n_cbf_constraints, 1))  # CBF vector
        # Define Objective Parameters
        cntrl_weights = cp.Parameter((self.n_control_dims, 1), nonneg=True)  # Q_tunning matrix for the control input
        # Add weights to the slack variables CLF
        n_slack = self.n_clf_constraints + n_cbf_slack
        if slack_weights is not None:
            n_weights = len(slack_weights)
            if n_weights != n_slack:
                raise "Error: the number of slack weights must be equal to the number of slack variables"
            weights = cp.vstack([cntrl_weights, jnp.array(slack_weights).reshape(n_slack, 1)])
        else:
            weights = cp.vstack([cntrl_weights, jnp.ones(self.n_clf_constraints, 1)])
        # Compute the objective function
        objective = 0.5 * cp.Minimize(cp.sum(cp.multiply(weights, cp.square(X))))
        # -------- Define Constraints --------
        contraints = []
        # Define the constraints CLF
        contraints += [A_clf @ u <= b_clf + slack_clf]
        # Define the constraints CBF
        contraints += [A_cbf @ u <= b_cbf + delta_cbf]
        # Define the constraints on the control input
        for control_idx, bound in enumerate(self.control_bounds):
            contraints += [u[control_idx] <= bound[1]]
            contraints += [u[control_idx] >= bound[0]]
        # -------- Define the Problem --------
        problem = cp.Problem(objective, contraints)
        if n_cbf_slack > 0 and self.n_cbf_constraints > 1:
            return CvxpyLayer(problem, parameters=[A_clf, b_clf, A_cbf, b_cbf, cntrl_weights],
                              variables=[u, slack_clf, slack_cbf])
        else:
            return CvxpyLayer(problem, parameters=[A_clf, b_clf, A_cbf, b_cbf, cntrl_weights], variables=[u, slack_clf])
