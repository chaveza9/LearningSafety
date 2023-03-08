"""A class for cloning an MPC policy using a neural network"""
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Tuple, Optional

from cvxpylayers.torch import CvxpyLayer
import torch
import cvxpy as cp
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import warnings


class CLFBarrierNetLayer(torch.nn.Module):
    def __init__(
        self,
        n_state_dims: int,
        n_control_dims: int,
        n_clf:int,
        clf_slack_weight: List[float],
        n_cbf:int, # Ordered list of Barrier functions
        n_cbf_slack: int,
        cbf_rel_degree: List[int],
        control_bounds: List[Tuple[float, float]],
        cbf_slack_weight: Optional[List[float]]= None,
        device: torch.device = torch.device("cpu"),
        solver_opts: Dict[str, Any] = {"solve_method": "ECOS", "verbose": False, "max_iters": 50000000},
        verbose: bool = True,
    ):
        """
        A model for cloning a policy.

        args:
            n_state_dims: how many input state dimensions
            n_control_dims: how many output control dimensions
            clf: Lyapunov functions (must be a list of length n_clf)
            clf_slack_weight: weight for the slack variables for CLF constraints (must be a list of length n_clf)
            cbf: barrier functions (must be a list of length n_cbf)
            n_cbf_slack: number of CBF slack variables
            weight_cbf_slack: weight for the slack variables for CBF constraints (must be a list of length n_cbf_slack)
            cbf_rel_degree: relative degree of the CBF constraints (must be a list of length n_cbf)
            control_bounds: control limits (must be a list of tuples of length n_control_dims)
            
            device: device to run the model on      
        """
        super(CLFBarrierNetLayer, self).__init__()

        # ------------------- Populate class paramaters -------------------#
        # otherwise, use the provided parameters
        self.verbose = verbose
        self.n_state_dims = n_state_dims
        self.n_control_dims = n_control_dims
        # Output control limits
        self.control_bounds = control_bounds
        # If the input dimensions are not provided, use the state dimensions

        self.n_input_dims = n_state_dims
        # If the output dimensions are not provided, use the control dimensions
        self.n_output_dims = n_control_dims
        # Define CLF Parameters
        # n_clf = len(clf) # Number of CLF constraints
        self.n_clf = n_clf # Number of CLF constraints
        # Check if the slack weights are provided
        if not clf_slack_weight:
            self.clf_slack_weight = [1.0]*n_clf
        else:
            self.clf_slack_weight = clf_slack_weight
        
        # Define CBF Parameters
        # n_cbf = len(cbf) # Number of CBF constraints
        self.n_cbf = n_cbf # Number of CBF constraints
        self.n_cbf_slack = n_cbf_slack # Number of slack variables for CBF constraints
        # Check if the slack weights are provided (can be less than the total number of CBF constraints)
        if not cbf_slack_weight:
            self.cbf_slack_weight = [1.0]*n_cbf_slack
        else:
            self.cbf_slack_weight = cbf_slack_weight
        # Check if the total number of slack weights is equal to the number of CBF constraints
        if len(self.cbf_slack_weight) != n_cbf_slack:
            raise ValueError("The number of CBF slack weights should be equal to the number of CBF constraints with slack")
        # Check if the relative degree of the CBF constraints is provided
        if len(cbf_rel_degree) != n_cbf:
            raise ValueError("The number of relative degrees should be equal to the number of CBF constraints")
        
        self.cbf_rel_degree = cbf_rel_degree
        # Define the function to preprocess the parameters for the HOCBF constraints
        # Define the device
        self.device = device
        
        # ------------------- Construct Policy Network -------------------#
        # Create nn parameters
        # self.cbf_rates = nn.Parameter(1e-3*torch.zeros(sum(cbf_rel_degree), 1))
        # self.clf_rates = nn.Parameter(1e-3*torch.zeros(n_clf, 1))
        # Define slack weights for the CLF and CBF objectives
        weights = self.clf_slack_weight + self.cbf_slack_weight
        self.cbf_layer = self._define_hocbf_filter_layer(weights)
        self.solver_opts = solver_opts
        
    def forward(self, A_clf, b_clf, A_cbf, b_cbf, cntrl_weights) -> torch.Tensor:
        """ Forward pass of the policy network."""
        try:
            u = self.cbf_layer(A_clf, b_clf,
                               A_cbf, b_cbf,
                               cntrl_weights,
                               solver_args=self.solver_opts)[0]
        except Exception as e:
            if self.verbose:
                warnings.warn("Error: {}".format(e))
            u = torch.zeros(self.n_control_dims)
        return u
        
    def _define_hocbf_filter_layer(self, slack_weights: List=[])->CvxpyLayer:
        """ Defines a differential cvxpy layer that contains a HOCBF filter together with a CLF controller."""
        # -------- Define Optimization Variables --------
        u = cp.Variable((self.n_control_dims, 1))  # control input
        # Define the control Lyapunov function variables
        slack_clf = cp.Variable((self.n_clf, 1))  # slack variables
        # Define the control Barrier function variables
        if self.n_cbf > 1 and self.n_cbf_slack > 0:
            slack_cbf = cp.Variable((self.n_cbf_slack, 1))
        else:
            slack_cbf = torch.zeros(self.n_cbf,1)
        # Create a list of slack variables
        if self.n_cbf_slack > 0:
            slack = cp.vstack([slack_clf, slack_cbf])
        else:
            slack = slack_clf
        if slack_cbf.shape[0] != self.n_cbf:
            delta_cbf = cp.vstack([slack_cbf, torch.zeros(self.n_cbf - self.n_cbf_slack, 1)])
        else:
            delta_cbf = slack_cbf
        X = cp.vstack([u, slack])
        # -------- Define Differentiable Variables --------
        # Define Constraint Parameters CLF
        A_clf = cp.Parameter((self.n_clf, self.n_control_dims))  # CLF matrix
        b_clf = cp.Parameter((self.n_clf,1))  # CLF vector
        # Define Constraint Parameters CBF
        A_cbf = cp.Parameter((self.n_cbf, self.n_control_dims))  # CBF matrix
        b_cbf = cp.Parameter((self.n_cbf,1))  # CBF vector
        # Define Objective Parameters
        cntrl_weights = cp.Parameter((self.n_control_dims, 1), nonneg=True)  # Q_tunning matrix for the control input
        # Add weights to the slack variables CLF
        n_slack = self.n_clf + self.n_cbf_slack
        if slack_weights:
            n_weights = len(slack_weights)
            if n_weights != n_slack:
                raise "Error: the number of slack weights must be equal to the number of slack variables"
            weights = cp.vstack([cntrl_weights, torch.tensor(slack_weights).reshape(n_slack, 1)])
        else:
            weights = cp.vstack([cntrl_weights, torch.ones(self.n_clf, 1).requires_grad_(False)])
        # Compute the objective function
        objective = 0.5*cp.Minimize(cp.sum(cp.multiply(weights, cp.square(X))))
        # -------- Define Constraints --------
        contraints = []
        # Define the constraints CLF
        contraints += [A_clf @ u <= b_clf+slack_clf]
        # Define the constraints CBF
        contraints += [A_cbf @ u <= b_cbf+delta_cbf]
        # Define the constraints on the control input
        for control_idx, bound in enumerate(self.control_bounds):
            contraints += [u[control_idx] <= bound[1]]
            contraints += [u[control_idx] >= bound[0]]
        # -------- Define the Problem --------
        problem = cp.Problem(objective, contraints)
        if self.n_cbf_slack>0 and self.n_cbf>1:
            return CvxpyLayer(problem, parameters=[A_clf, b_clf, A_cbf, b_cbf, cntrl_weights], variables=[u,slack_clf, slack_cbf])
        else:
            return CvxpyLayer(problem, parameters=[A_clf, b_clf, A_cbf, b_cbf, cntrl_weights], variables=[u,slack_clf])


class BarrierNetLayer(nn.Module):
    def __init__(
        self,
        n_state_dims: int,
        n_control_dims: int,
        n_cbf: int,  # Ordered list of Barrier functions
        n_cbf_slack: int,
        cbf_rel_degree: List[int],
        control_bounds: List[Tuple[float, float]],
        cbf_slack_weight: Optional[List[float]] = None,
        device: torch.device = torch.device("cpu"),
        solver_opts: Dict[str, Any] = {"solve_method": "ECOS", "verbose": False, "max_iters": 50000000},
        verbose: bool = True,
    ):
        super(BarrierNetLayer, self).__init__()

        # ------------------- Populate class paramaters -------------------#
        # otherwise, use the provided parameters
        self.verbose = verbose
        self.n_state_dims = n_state_dims
        self.n_control_dims = n_control_dims
        # Output control limits
        self.control_bounds = control_bounds
        # If the input dimensions are not provided, use the state dimensions

        self.n_input_dims = n_state_dims
        # If the output dimensions are not provided, use the control dimensions
        self.n_output_dims = n_control_dims
        # Define CBF Parameters
        self.n_cbf = n_cbf  # Number of CBF constraints
        self.n_cbf_slack = n_cbf_slack  # Number of slack variables for CBF constraints
        # Check if the slack weights are provided (can be less than the total number of CBF constraints)
        if not cbf_slack_weight:
            self.cbf_slack_weight = [1.0] * n_cbf_slack
        else:
            self.cbf_slack_weight = cbf_slack_weight
        # Check if the total number of slack weights is equal to the number of CBF constraints
        if len(self.cbf_slack_weight) != n_cbf_slack:
            raise ValueError(
                "The number of CBF slack weights should be equal to the number of CBF constraints with slack")
        # Check if the relative degree of the CBF constraints is provided
        if len(cbf_rel_degree) != n_cbf:
            raise ValueError("The number of relative degrees should be equal to the number of CBF constraints")

        self.cbf_rel_degree = cbf_rel_degree
        # Define the function to preprocess the parameters for the HOCBF constraints
        # Define the device
        self.device = device
        # ------------------- Construct Policy Network -------------------#
        # Create nn parameters
        # self.cbf_rates = nn.Parameter(torch.randn(sum(cbf_rel_degree), 1)).to(self.device)
        # Define slack weights for the CLF and CBF objectives
        weights = self.cbf_slack_weight
        self.cbf_layer = self._define_hocbf_filter(weights)
        self.solver_opts = solver_opts

    def forward(self, u_ref, A_cbf, b_cbf) -> torch.Tensor:
        """ Forward pass of the policy network."""
        # Get the control input from the CvxpyLayer
        # Deconstruct the weights (CLF rates always come first)
        try:
            u = self.cbf_layer(u_ref, A_cbf, b_cbf, solver_args=self.solver_opts)[0]
        except Exception as e:
            if self.verbose:
                print("Error: {}\n".format(e))
            u = u_ref
        return u

    def _define_hocbf_filter(self, slack_weights: Optional[List] = None) -> CvxpyLayer:
        """ Defines a differential cvxpy layer that contains a HOCBF filter together with a CLF controller."""
        # -------- Define Optimization Variables --------
        u = cp.Variable((self.n_control_dims, 1))  # control input
        # Define reference input
        u_ref = cp.Parameter((self.n_control_dims, 1))
        # Define the control Barrier function variables
        if self.n_cbf_slack > 0:
            slack_cbf = cp.Variable((self.n_cbf_slack, 1))
        else:
            slack_cbf = torch.zeros(self.n_cbf, 1)
        # Create a list of slack variables
        if slack_cbf.shape[0] != self.n_cbf:
            delta_cbf = cp.vstack([slack_cbf, torch.zeros(self.n_cbf - self.n_cbf_slack, 1)])
        else:
            delta_cbf = slack_cbf

        # -------- Define Differentiable Variables --------
        # Define Constraint Parameters CBF
        A_cbf = cp.Parameter((self.n_cbf, self.n_control_dims))  # CBF matrix
        b_cbf = cp.Parameter((self.n_cbf, 1))  # CBF vector
        # Define Objective Parameters
        # Add weights to the slack variables CBF
        n_slack = self.n_cbf_slack
        if slack_weights:
            n_weights = len(slack_weights)
            if n_weights != n_slack:
                raise "Error: the number of slack weights must be equal to the number of slack variables"
            weights = torch.tensor(slack_weights).reshape(n_slack, 1)
            objective_slack = cp.sum(cp.multiply(weights, slack_cbf))
        else:
            objective_slack = torch.zeros(1, 1)
        # Compute the objective function
        objective = 0.5 * cp.Minimize(cp.sum(cp.square(u-u_ref)) + objective_slack)
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


