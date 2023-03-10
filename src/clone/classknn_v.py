"""A class for cloning an MPC policy using a neural network"""
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Tuple, Optional
import warnings

import torch
import torch.nn as nn
import numpy as np
# Import local modules
from src.BarrierNet.barriernet import BarrierNetLayer
from src.models import MonotonicNN
from functorch import vmap, jacrev
from tqdm import tqdm



class ClassKNN(torch.nn.Module):
    def __init__(
            self,
            hidden_layers: int,
            hidden_layer_width: int,
            n_state_dims: int,
            n_control_dims: int,
            n_input_dims: int,
            cbf: List[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]],
            # Ordered list of Barrier functions
            n_cbf_slack: int,
            cbf_rel_degree: List[int],
            state_space: List[Tuple[float, float]],
            control_bounds: List[Tuple[float, float]],
            x_obst: torch.Tensor,
            r_obst: torch.Tensor,
            cbf_slack_weight: Optional[List[float]] = None,
            load_from_file: Optional[str] = None,
            use_barrier_net: bool = True,
    ):
        """
        A model for cloning a policy.

        args:
            hidden_layers: number of hidden layers
            hidden_layer_width: width of hidden layers (num neurons per layer)
            n_state_dims: how many input state dimensions
            n_control_dims: how many output control dimensions
            n_input_dims: how many input dimensions to the barrier net (can be different from state dims with at least n_state_dims)
            cbf: List of barrier functions (in order)
            n_cbf_slack: number of CBF slack variables (can be 0)
            cbf_slack_weight: weight for the slack variables for CBF constraints (must be a list of length n_cbf_slack)
            cbf_rel_degree: relative degree of the barrier functions (must be a list of length n_cbfs)
            state_space: list of tuples of (min, max) for each state dimension
            control_bounds: list of tuples of the form (min, max) for each control dimension
            preprocess_barrier_input_fn: function to preprocess the input to the barrier net (must construct matrices for clf and cbf constraints)
            load_from_file: path to a file to load the model from

        """
        super(ClassKNN, self).__init__()

        # ----------------- Propagate Class Properties -----------------
        self.hidden_layers = hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.n_state_dims = n_state_dims
        self.n_control_dims = n_control_dims
        self.n_input_dims = n_input_dims
        self.load_from_file = load_from_file
        self.state_space = state_space
        # Define device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)

        # ----------------- Construct MLP Network -----------------
        # Compute the output dimension of the MLP
        self.n_output_dims = n_control_dims
        self.policy_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.policy_layers["input_linear"] = nn.Linear(
            self.n_input_dims,
            self.hidden_layer_width,
        )
        self.policy_layers["input_activation"] = nn.Tanh()
        for i in range(self.hidden_layers):
            self.policy_layers[f"layer_{i}_linear"] = nn.Linear(
                self.hidden_layer_width, self.hidden_layer_width
            )
            self.policy_layers[f"layer_{i}_activation"] = nn.Tanh()
        # Output the penalty parameters for cbf
        self.policy_layers["output_linear"] = nn.Linear(
            self.hidden_layer_width, self.n_control_dims
        )
        # Convert to sequential model
        self.policy_nn = nn.Sequential(self.policy_layers).to(self.device)

        # ----------------- Construct CBF Network -----------------
        # Monotonically increasing neural network for relative degree 2
        #Includes obstacle position encoding
        self.mono1 = MonotonicNN(self.n_state_dims+2+1, [hidden_layer_width]*hidden_layers, nb_steps=50, dev=self.device).to(self.device)
        # self.mono2 = MonotonicNN(self.n_state_dims+2+1, [hidden_layer_width]*hidden_layers, nb_steps=50, dev=self.device).to(self.device)
        self.use_barrier_net = use_barrier_net
        self.n_cbf = len(cbf)
        self.cbf_rel_degree = cbf_rel_degree
        self.cbf = cbf
        self.x_obst = x_obst.to(self.device)
        self.r_obst = r_obst.to(self.device)
        # ----------------- Construct Barrier Network -----------------
        self.barrier_layer = BarrierNetLayer(
            n_state_dims=self.n_state_dims,
            n_control_dims=self.n_control_dims,
            n_cbf=self.n_cbf,
            n_cbf_slack=n_cbf_slack,
            cbf_slack_weight=cbf_slack_weight,
            cbf_rel_degree=self.cbf_rel_degree,
            control_bounds=control_bounds,
            device=self.device,
        )

        # Define a vmap function for computing the lie derivative
        self.vmap_lie_derivative = \
            vmap(self._compute_lie_derivative_1st_order, in_dims=(0, None, None), out_dims=(0,0,0))

        # Load the weights and biases if provided
        try:
            if load_from_file is not None:
                checkpoint = torch.load(load_from_file, map_location=self.device)
                self.load_state_dict(checkpoint["model_state_dict"])
        except:
            warnings.warn("Failed to load model from file")

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        # Construct the input to the barrier net
        x = torch.atleast_2d(x).to(self.device)
        u = torch.atleast_2d(u).to(self.device)
        # pass state through policy network
        u_hat = self.policy_nn(x)
        if self.use_barrier_net:
            u_hat =  self.barrier_layer(*self.compute_batched_hocbf_params(x, u_hat))
        # pass state through cbf network
        return u_hat, self.compute_batched_hocbf_params(x, u)

    def eval_np(self, x: np.ndarray):
        # Construct the input to the barrier net

        x = torch.atleast_2d(torch.from_numpy(x)).to(self.device)
        # pass state through policy network
        u_hat = self.policy_nn(x)
        # pass state through cbf network
        return self.barrier_layer(*self.compute_hocbf_params(x, u_hat)).detach().cpu().squeeze()

    def _f(self, x):
        """Open Loop Dynamics"""
        return torch.vstack([x[2] * torch.cos(x[3]), x[2] * torch.sin(x[3]), torch.zeros(2, 1).to(self.device)])

    def _g(self):
        """ Control Matrix"""
        return torch.vstack([torch.zeros(2, 2).to(self.device), torch.eye(2).to(self.device)])

    def compute_hocbf_params(self, x: torch.Tensor, u_ref: torch.Tensor):
        # Compute CBF parameters
        A_cbf = torch.zeros(self.n_cbf, self.n_control_dims).repeat(x.shape[0], 1, 1).to(self.device)
        b_cbf = torch.zeros(self.n_cbf, 1).repeat(x.shape[0], 1, 1).to(self.device)
        # Iterate over each cbf over each state
        for i in range(x.shape[0]):
            for idx, cbf in enumerate(self.cbf):
                # check relative degree of cbf (Obstacle type)
                if idx == 0:
                    barrier_fun = lambda x: cbf(x, self.x_obst, self.r_obst)
                    h = torch.cat((x[i], self.x_obst.reshape(-1)), dim=0)
                    alpha_1 = lambda psi: self.mono1(torch.atleast_2d(psi), torch.atleast_2d(h))
                    # alpha_2 = lambda psi: self.mono2(torch.atleast_2d(psi), torch.atleast_2d(h))
                    A_i, b_i, _ = self._compute_lie_derivative_1st_order(x[i], barrier_fun, alpha_1)
                    A_cbf[i, idx] = -A_i
                    b_cbf[i, idx] = b_i
                elif self.cbf_rel_degree[idx] == 1:
                    alpha_1 = lambda psi: 1.0 * psi  # force gain of 1.0
                    A_i, b_i, _ = self._compute_lie_derivative_1st_order(x[i], cbf, alpha_1)
                    A_cbf[i, idx] = -A_i
                    b_cbf[i, idx] = b_i
        return u_ref.reshape((x.shape[0], self.n_control_dims, 1)), A_cbf, b_cbf

    def compute_batched_hocbf_params(self, x: torch.Tensor, u_ref: torch.Tensor):
        # Compute CBF parameters
        A_cbf = torch.zeros(self.n_cbf, self.n_control_dims).repeat(x.shape[0], 1, 1).to(self.device)
        b_cbf = torch.zeros(self.n_cbf, 1).repeat(x.shape[0], 1, 1).to(self.device)
        # Iterate over each cbf over each state

        # Make sure that the input is a batch
        x = torch.atleast_2d(x).to(self.device)
        batch_size = x.shape[0]
        for idx, cbf in enumerate(self.cbf):
            # check relative degree of cbf (Obstacle type)
            if idx == 0:
                barrier_fun = lambda x: cbf(x, self.x_obst, self.r_obst)
                alpha_mock = lambda psi: psi*0.0
                A_i, Lfb, psi0 = self.vmap_lie_derivative(x, barrier_fun, alpha_mock)
                # Compute the class kappa function and barrier constraint
                h = torch.cat((x, self.x_obst.repeat((x.shape[0], 1))), dim=1)
                b_i = Lfb + self.mono1(torch.atleast_2d(psi0).T, h)
                A_cbf[:, idx] = -A_i
                b_cbf[:, idx] = b_i
            else:
                alpha_1 = lambda psi: 1.0 * psi  # force gain of 1.0
                A_i, b_i, _ = self.vmap_lie_derivative(x, cbf, alpha_1)
                A_cbf[:, idx] = -A_i
                b_cbf[:, idx] = b_i
        return u_ref.reshape((x.shape[0], self.n_control_dims, 1)), A_cbf, b_cbf

    def _compute_lie_derivative_1st_order(self, x: torch.Tensor, barrier_fun: Callable, alpha_fun_1: Callable):
        """Compute the Lie derivative of the CBF wrt the dynamics"""
        # Compute the CBF
        psi0 = barrier_fun(x)
        db_dx = jacrev(barrier_fun)(x)
        Lfb = db_dx @ self._f(x)
        Lgb = db_dx @ self._g()
        psi1 = Lfb + alpha_fun_1(psi0)
        # Compute the Lie derivative
        return Lgb, psi1, psi0

    def _compute_lie_derivative_2nd_order_jacrev(self, x: torch.Tensor, barrier_fun: Callable, alpha_fun_1: Callable,
                                          alpha_fun_2: Callable):
        psi0 = barrier_fun(x)
        Lfb = lambda x: jacrev(barrier_fun)(x) @ self._f(x)
        db2_dx = jacrev(Lfb)(x)
        Lf2b = db2_dx @ self._f(x)
        LgLfb = db2_dx @ self._g()

        psi1 = lambda x: Lfb(x) + alpha_fun_1(psi0)
        psi1_dot = jacrev(psi1)(x) @ self._f(x)

        psi2 = psi1_dot + alpha_fun_2(psi1(x))

        return LgLfb, Lf2b + psi2, psi1(x)

    def _compute_lie_derivative_2nd_order(self, x: torch.Tensor, barrier_fun: Callable, alpha_fun_1: Callable,
                                          alpha_fun_2: Callable):
        """Compute the Lie derivative of the CBF wrt the dynamics"""
        # Make sure the input requires gradient
        x.requires_grad_(True)
        # Compute the CBF
        psi0 = barrier_fun(x)
        db_dx = torch.autograd.grad(psi0, x, create_graph=True, retain_graph=True)[0]
        Lfb = db_dx @ self._f(x)
        db2_dx = torch.autograd.grad(Lfb, x, retain_graph=True, create_graph=True)[0]
        Lf2b = db2_dx @ self._f(x)
        LgLfb = db2_dx @ self._g()
        psi1 = Lfb + alpha_fun_1(psi0)
        psi1_dot = torch.autograd.grad(psi1, x, retain_graph=True, create_graph=True)[0] @ self._f(x)
        psi2 = psi1_dot + alpha_fun_2(psi1)
        # Compute the Lie derivative
        return LgLfb, Lf2b + psi2, psi1

    def _barrier_loss(self, u_train, cbf_params):
        """Compute the barrier loss"""
        # Compute the CBF parameters
        _, A_cbf, b_cbf = cbf_params
        LgLfb = -A_cbf
        # Compute the barrier loss
        loss = 0
        beta1 = 1
        beta2 = 0.001
        # Reshape u to match the shape of the CBF parameters
        batch_size = u_train.shape[0]
        u_train = u_train.reshape((batch_size, self.n_control_dims, 1))
        # constraint violation
        loss += beta1*torch.sum(torch.relu(-(torch.bmm(LgLfb, u_train) + b_cbf)))
        # constraint satisfaction
        loss += beta2*torch.sum(torch.relu(torch.tanh(torch.bmm(LgLfb, u_train) + b_cbf)))
        # # violation
        # loss += beta1*torch.sum(torch.relu(-psi_n_1))
        # # saturation
        # loss += beta2*torch.sum(torch.relu(torch.tanh(psi_n_1)))

        return loss/batch_size

    def clone(
            self,
            expert: Callable[[torch.Tensor, int], torch.Tensor],
            n_pts: int,
            n_epochs: int,
            learning_rate: float,
            batch_size: int = 64,
            n_steps: int = 100,
            save_path: Optional[str] = None,
            load_checkpoint: Optional[str] = None,
            x_obs: Optional[torch.Tensor] = None,
            x_des: Optional[torch.Tensor] = None,
    ):
        """Clone the provided expert policy. Uses dead-simple supervised regression
        to clone the policy (no DAgger currently).

        args:
            expert: the policy to clone
            n_pts: the number of points in the cloning dataset
            n_epochs: the number of epochs to train for
            learning_rate: step size
            batch_size: size of mini-batches
            save_path: path to save the file (if none, will not save the model)
        """
        # Generate some training data
        # Start by sampling points uniformly from the state space


        # Create buffer for training data
        x_train = torch.zeros((n_pts, self.n_state_dims))
        u_expert = torch.zeros((n_pts, self.n_control_dims))
        # Create random initial conditions
        n_x0s = n_pts // n_steps
        x0s = []
        data_gen_range = tqdm(range(n_x0s), ascii=True, desc="Generating data")
        data_gen_range.set_description("Generating training data...")
        for i in data_gen_range:
            x0s.append(np.zeros(self.n_state_dims))
            for dim in range(self.n_state_dims - 1):
                x0s[i][dim] = np.random.uniform(self.state_space[dim][0], self.state_space[dim][1])
            # Run expert policy for n_steps
            x_train[i * n_steps:(i + 1) * n_steps, :], u_expert[i * n_steps:(i + 1) * n_steps, :] \
                = expert(x0s[i], n_steps+1)

        # Move inputs and outputs to the GPU
        x_train = x_train.to(self.device)
        u_expert = u_expert.to(self.device)

        # Make a loss function and optimizer
        mse_loss_fn = torch.nn.MSELoss(reduction="mean")
        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate
        )
        # Load checkpoint if provided
        if load_checkpoint:
            try:
                checkpoint = torch.load(load_checkpoint, map_location=self.device)
                self.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                epoch = checkpoint["epoch"]
                loss = checkpoint["loss"]
                n_epochs -= epoch
                print(f"Loaded checkpoint from {load_checkpoint} at epoch {epoch} with loss {loss}")
            except FileNotFoundError:
                print(f"Checkpoint {load_checkpoint} not found. Starting from scratch.")

        curr_min_loss = np.inf
        # Optimize in mini-batches
        for epoch in range(n_epochs):
            permutation = torch.randperm(n_pts)

            loss_accumulated = 0.0
            epoch_range = tqdm(range(0, n_pts, batch_size), ascii=True, desc="Epoch")
            epoch_range.set_description(f"Epoch {epoch} training...")
            for i in epoch_range:
                batch_indices = permutation[i: i + batch_size]
                x_batch = x_train[batch_indices]
                u_expert_batch = u_expert[batch_indices]

                # Forward pass: predict the control input
                u_hat, cbf_params = self(x_batch, u_expert_batch)
                # Compute the loss and backpropagate
                # MSE Loss
                # Clone Loss
                loss = 0.1*mse_loss_fn(u_hat.squeeze(), u_expert_batch)
                # CBF Loss
                loss += self._barrier_loss(u_expert_batch, cbf_params)
                # Add L1 regularization
                for layer in self.policy_nn:
                    if not hasattr(layer, "weight"):
                        continue
                    loss += 0.001 * learning_rate * torch.norm(layer.weight, p=2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_accumulated += loss.detach()

            print(f" \n Epoch {epoch}: {loss_accumulated / (n_pts / batch_size)} \n")

            if loss_accumulated < curr_min_loss:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_accumulated,
                }, save_path)
                curr_min_loss = loss_accumulated

