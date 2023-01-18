"""A class for cloning an MPC policy using a neural network"""
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Tuple, Optional

from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class BarrierNet(torch.nn.Module):
    def __init__(
        self,
        hidden_layers: int,
        hidden_layer_width: int,
        n_state_dims: int,
        n_control_dims: int,
        n_input_dims: int,
        n_output_dims: int,
        state_space: List[Tuple[float, float]],
        barrier_net_fn: Callable,
        preprocess_input_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        load_from_file: Optional[str] = None,
    ):
        """
        A model for cloning a policy.

        args:
            hidden_layers: how many hidden layers to have
            hidden_layer_width: how many neurons per hidden layer
            n_state_dims: how many input state dimensions
            n_control_dims: how many output control dimensions
            n_input_dims: how many input dimensions to the barrier net
            n_output_dims: how many output dimensions to the barrier net
            state_space: a list of lower and upper bounds for each state dimension
            barrier_net_fn: a cvxpylayer function that takes in the state and penalty parameters and returns the control
            preprocess_input_fn: a function that takes in the state and returns the state to be fed into the policy network
            load_from_file: a path to a file containing a saved instance of a policy
                cloning model. If provided, ignores all other arguments and uses the
                saved parameters.
        """
        super(BarrierNet, self).__init__()

        # If a save file is provided, use the saved parameters
        saved_data: Dict[str, Any] = {}
        if load_from_file is not None:
            saved_data = torch.load(load_from_file)
            self.hidden_layers = saved_data["hidden_layers"]
            self.hidden_layer_width = saved_data["hidden_layer_width"]
            self.n_state_dims = saved_data["n_state_dims"]
            self.n_control_dims = saved_data["n_control_dims"]
            self.state_space = saved_data["state_space"]
            self.n_input_dims = saved_data["n_input_dims"]
            self.n_output_dims = saved_data["n_output_dims"]
            self.barrier_net_fn = saved_data["barrier_net_fn"]
            self.preprocess_input_fn = saved_data["preprocess_input_fn"]
        else:  # otherwise, use the provided parameters
            self.hidden_layers = hidden_layers
            self.hidden_layer_width = hidden_layer_width
            self.n_state_dims = n_state_dims
            self.n_control_dims = n_control_dims
            self.state_space = state_space
            self.n_input_dims = n_input_dims
            self.n_output_dims = n_output_dims
            self.barrier_net_fn = barrier_net_fn
            self.preprocess_input_fn = preprocess_input_fn

        # Construct the policy network
        self.policy_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.policy_layers["input_linear"] = nn.Linear(
            self.n_input_dims,
            self.hidden_layer_width,
        )
        self.policy_layers["input_activation"] = nn.ReLU()
        for i in range(self.hidden_layers):
            self.policy_layers[f"layer_{i}_linear"] = nn.Linear(
                self.hidden_layer_width, self.hidden_layer_width
            )
            self.policy_layers[f"layer_{i}_activation"] = nn.ReLU()
        # Output the penalty parameters for cbf
        self.policy_layers["output_linear"] = nn.Linear(
            self.hidden_layer_width, self.n_output_dims
        )
        # Construct the policy network
        self.policy_nn = nn.Sequential(self.policy_layers)

        # Load the weights and biases if provided
        if load_from_file is not None:
            self.load_state_dict(saved_data["state_dict"])

        # Define device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        # pass state through policy network
        if self.preprocess_input_fn is not None:
            x = self.preprocess_input_fn(x)
        penalty_params = self.policy_nn(x.to(self.device)).relu() # relu to ensure positive penalty parameters
        return self.barrier_net_fn(x.to(self.device), penalty_params)

    def eval_np(self, x: np.ndarray):
        if self.preprocess_input_fn is not None:
            x = self.preprocess_input_fn(x)
        penalty_params = self.policy_nn(torch.from_numpy(x).float().to(self.device))
        output = self.barrier_net_fn(torch.from_numpy(x).float().to(self.device), penalty_params)
        return output.detach().cpu().numpy()

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
                batch_indices = permutation[i : i + batch_size]
                x_batch = x_train[batch_indices]
                u_expert_batch = u_expert[batch_indices]

                # Forward pass: predict the control input
                # iterate through the batch
                u_predicted = torch.zeros((x_batch.shape[0], self.n_control_dims)).to(self.device)
                for j in range(x_batch.shape[0]):
                    if self.preprocess_input_fn is not None:
                        x_in = self.preprocess_input_fn(x_batch[j, :])
                    else:
                        x_in = x_batch[j, :]
                    u_predicted[j, :] = self(x_in)

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
