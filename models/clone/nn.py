"""A class for cloning an MPC policy using a neural network"""
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Tuple, Optional
import warnings

from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
import numpy as np
from .barriernet import BarrierNetLayer
from tqdm import tqdm


class PosLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PosLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn((in_dim, out_dim)))
        self.bias = nn.Parameter(torch.zeros((out_dim,)))

    def forward(self, x):
        return torch.matmul(x, self.weight) + torch.abs(self.bias)


class PolicyCloningModel(torch.nn.Module):
    def __init__(
            self,
            hidden_layers: int,
            hidden_layer_width: int,
            n_state_dims: int,
            n_control_dims: int,
            n_input_dims: int,
            n_cbf: int,  # Ordered list of Barrier functions
            n_cbf_slack: int,
            cbf_rel_degree: List[int],
            state_space: List[Tuple[float, float]],
            control_bounds: List[Tuple[float, float]],
            cbf_slack_weight: Optional[List[float]] = None,
            preprocess_barrier_input_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[
                torch.Tensor]]] = None,
            load_from_file: Optional[str] = None,
    ):
        """
        A model for cloning a policy.

        args:
            hidden_layers: number of hidden layers
            hidden_layer_width: width of hidden layers (num neurons per layer)
            n_state_dims: how many input state dimensions
            n_control_dims: how many output control dimensions
            n_input_dims: how many input dimensions to the barrier net (can be different from state dims with at least n_state_dims)
            n_cbf: number of barrier functions
            n_cbf_slack: number of CBF slack variables (can be 0)
            cbf_slack_weight: weight for the slack variables for CBF constraints (must be a list of length n_cbf_slack)
            cbf_rel_degree: relative degree of the barrier functions (must be a list of length n_cbfs)
            state_space: list of tuples of (min, max) for each state dimension
            control_bounds: list of tuples of the form (min, max) for each control dimension
            preprocess_barrier_input_fn: function to preprocess the input to the barrier net (must construct matrices for clf and cbf constraints)
            load_from_file: path to a file to load the model from
            
        """
        super(PolicyCloningModel, self).__init__()

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
        self.n_output_dims = n_control_dims+sum(cbf_rel_degree)
        self.policy_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.policy_layers["input_linear"] = nn.Linear(
            self.n_input_dims,
            self.hidden_layer_width,
        )
        self.policy_layers["input_activation"] = nn.Softplus()
        for i in range(self.hidden_layers):
            self.policy_layers[f"layer_{i}_linear"] = nn.Linear(
                self.hidden_layer_width, self.hidden_layer_width
            )
            self.policy_layers[f"layer_{i}_activation"] = nn.Softplus()
        # Output the penalty parameters for cbf
        self.policy_layers["output_linear"] = nn.Linear(
            self.hidden_layer_width, self.n_output_dims
        )
        # Convert to sequential model
        self.policy_nn = nn.Sequential(self.policy_layers).to(self.device)

        # ----------------- Construct Barrier Network -----------------
        self.barrier_layer = BarrierNetLayer(
            n_state_dims=self.n_state_dims,
            n_control_dims=self.n_control_dims,
            n_cbf=n_cbf,
            n_cbf_slack=n_cbf_slack,
            cbf_slack_weight=cbf_slack_weight,
            cbf_rel_degree=cbf_rel_degree,
            control_bounds=control_bounds,
            preprocess_input_fn=preprocess_barrier_input_fn,
            device=self.device,
        )

        # Load the weights and biases if provided
        try:
            if load_from_file is not None:
                checkpoint = torch.load(load_from_file, map_location=self.device)
                self.load_state_dict(checkpoint["model_state_dict"])
        except:
            warnings.warn("Failed to load model from file")

    def forward(self, x: torch.Tensor, x_obs: Optional[torch.Tensor], x_des: Optional[torch.Tensor]):
        # Construct the input to the barrier net
        x = torch.atleast_2d(x)
        if x_obs is not None and x_des is not None:
            x_obs = x_obs.repeat(x.shape[0],1)
            x_des = x_des.repeat(x.shape[0],1)
            x_in = torch.hstack([x, x_obs, x_des]).to(self.device)
        elif x_obs is not None and x_des is None:
            x_obs = x_obs.repeat(x.shape[0],1)
            x_in = torch.hstack([x, x_obs]).to(self.device)
        else:
            x_in = x.to(self.device)
        # pass state through policy network
        u_out = self.policy_nn(x_in)
        u_ref = u_out[:, :self.n_control_dims]
        cbf_rates = u_out[:, self.n_control_dims:]
        # pass state and penalty parameters through barrier net
        return self.barrier_layer(x.to(self.device), u_ref, cbf_rates)

    def eval_np(self, x: np.ndarray, x_obs: Optional[np.ndarray] = None, x_des: Optional[np.ndarray] = None):
        # Construct the input to the barrier net

        x = torch.atleast_2d(torch.from_numpy(x)).to(self.device)
        if x_obs is not None and x_des is not None:
            x_obs = x_obs.repeat(x.shape[0], 1).to(self.device)
            x_des = x_des.repeat(x.shape[0], 1).to(self.device)
            x_in = torch.hstack([x, x_obs, x_des]).to(self.device)
        elif x_obs is not None and x_des is None:
            x_obs = x_obs.repeat(x.shape[0], 1).to(self.device)
            x_in = torch.hstack([x, x_obs]).to(self.device)
        else:
            x_in = x.to(self.device)

        u_out = self.policy_nn(x_in)
        u_ref = u_out[:, :self.n_control_dims]
        cbf_rates = u_out[:, self.n_control_dims:]
        # pass state and penalty parameters through barrier net
        return self.barrier_layer(x.to(self.device), u_ref, cbf_rates).detach().cpu().squeeze()

    def save_to_file(self, save_path: str):
        save_data = {
            "hidden_layers": self.hidden_layers,
            "hidden_layer_width": self.hidden_layer_width,
            "n_state_dims": self.n_state_dims,
            "n_control_dims": self.n_control_dims,
            "state_space": self.state_space,
            "n_output_dims": self.n_output_dims,
            "barrier_net_fn": self.barrier_net_fn,
            "state_dict": self.state_dict(),
        }
        torch.save(save_data, save_path)

    def clone(
            self,
            expert: Callable[[torch.Tensor], torch.Tensor],
            n_pts: int,
            n_epochs: int,
            learning_rate: float,
            batch_size: int = 64,
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
        x_train = torch.zeros((n_pts, self.n_state_dims))
        for dim in range(self.n_state_dims):
            x_train[:, dim] = torch.Tensor(n_pts).uniform_(*self.state_space[dim])

        # Now get the expert's control input at each of those points
        u_expert = torch.zeros((n_pts, self.n_control_dims))
        data_gen_range = tqdm(range(n_pts), ascii=True, desc="Generating data")
        data_gen_range.set_description("Generating training data...")
        for i in data_gen_range:
            u_expert[i, :] = expert(x_train[i, :])

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
            checkpoint = torch.load(load_checkpoint, map_location=self.device)
            self.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            loss = checkpoint["loss"]
            n_epochs -= epoch
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
                u_predicted = self(x_batch, x_obs.squeeze(), x_des.squeeze()).squeeze().to(self.device)
                # Compute the loss and backpropagate
                loss = mse_loss_fn(u_predicted, u_expert_batch)

                # Add L1 regularization
                for layer in self.policy_nn:
                    if not hasattr(layer, "weight"):
                        continue
                    loss += 0.001 * learning_rate * torch.norm(layer.weight, p=1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_accumulated += loss.detach()

            print(f"Epoch {epoch}: {loss_accumulated / (n_pts / batch_size)}")
            if loss_accumulated < curr_min_loss:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_accumulated,
                }, save_path)
                curr_min_loss = loss_accumulated

        # if save_path is not None:
        #     self.save_to_file(save_path)
