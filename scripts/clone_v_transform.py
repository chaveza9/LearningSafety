"""Test the obstacle avoidance BarrierNetLayer for a dubins vehicle"""
from typing import Callable, Tuple, List, Dict, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
import copy
import datetime
import argparse
import tqdm


sys.path.append(os.path.abspath('..'))
from models import MLP
from models.UMNN.MonotonicNN import MonotonicNN
from models.hocbf.barriernet import BarrierNetLayer
from mpc.dynamics_constraints import car_2d_dynamics as dubins_car_dynamics
from utils import compute_lie_derivative_1st_order, simulate_and_plot, define_dubins_expert

# -------------------------------------------
# DEFINE DEVICE
# Define device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# -------------------------------------------
# DEFINE PI AND CONSTANTS
torch.pi = torch.acos(torch.zeros(1)).item() * 2
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
state_space = [(-3, -1),
               (-2, 2),
               (0, 2),
               (-np.pi, np.pi)]
# Define control bounds
control_bounds = [(-2.0, 2.0),
                  (-1, 1)]
# ___________________________________________
dynamics_fn = dubins_car_dynamics
# Define control affine dynamics
_f = lambda x: torch.vstack([x[2] * torch.cos(x[3]), x[2] * torch.sin(x[3]), torch.zeros(2, 1).to(device)])
_g = torch.vstack([torch.zeros(2, 2), torch.eye(2, 2)]).to(device)
# -------------------------------------------

# CONSTRUCT HOCBF CONSTRAINTS TO BE USED IN THE BARRIERNETLAYER
def compute_cbf_params(x: torch.Tensor, cbfs: List[Dict], device: torch.device = torch.device("cpu"), 
                       cntrl_weights: Optional[torch.Tensor]=torch.ones(n_controls, 1).to(device)):
    """ Compute constraint parameters for the CLF and CBF """
    # -----------------Housekeeping -------------
    # Make sure that vector is in the correct device
    x = torch.from_numpy(x).to(device)
    # Extract batch size from x vector dimension
    n_batch = x.shape[0]
    # Compute the total number of CLF and CBF constraints
    n_cbf = len(cbfs)
    # -------- Compute CBF Constraint Parameters --------
    # Create cbf matrix buffers
    A_cbf = torch.zeros(n_cbf, n_controls).repeat(n_batch,1,1).to(device)
    b_cbf = torch.zeros(n_cbf, 1).repeat(n_batch,1,1).to(device).to(device)
    # Compute CBF parameters
    for idx, cbf in enumerate (cbfs):
        barrier_function = lambda x: cbf['f'] (x, *cbf['args'])
        if (cbf["type"] == "state_constraint" or cbf["type"] == "distance_trans") and cbf['rel_degree'] == 1:
            # Extract class kappa function vararg parameters
            args = torch.tensor(cbf['alpha_args']).repeat(n_batch,1).to(device)
            # Create alpha functions for each relative degree
            alpha1 =  cbf['class_kappa'][0](x, p)
            # Compute CBF constraint
            G, F, _ = compute_lie_derivative_1st_order(x, barrier_fun= barrier_function, alpha_fun_1= alpha1)
        else :
            raise "Error: CBF type not recognized"
        # Populate CBF constraint parameters
        A_cbf[idx, :n_controls] = -G.detach()
        b_cbf[idx] = F.detach()
        # Make sure that control weights have the correct dimension
        cntrl_weights = cntrl_weights.reshape(n_controls,1)

    return A_cbf, b_cbf, cntrl_weights

def barrier_loss(u_train, cbf_params):
    """Compute the barrier loss"""
    # Compute the CBF parameters
    _, A_cbf, b_cbf = cbf_params
    LgLfb = -A_cbf
    # Compute the barrier loss
    loss = 0
    beta1 = 1
    beta2 = 0.001
    gamma = 0
    # Reshape u to match the shape of the CBF parameters
    batch_size = u_train.shape[0]
    u_train = u_train.reshape((batch_size, n_controls, 1))
    # constraint violation
    loss += beta1 * torch.sum(torch.relu(gamma-(torch.bmm(LgLfb, u_train) + b_cbf)))
    # constraint satisfaction
    loss += beta2 * torch.sum(torch.relu(torch.tanh(torch.bmm(LgLfb, u_train) + b_cbf+gamma)))
    # # violation
    # loss += beta1*torch.sum(torch.relu(-psi_n_1))
    # # saturation
    # loss += beta2*torch.sum(torch.relu(torch.tanh(psi_n_1)))

    return loss / batch_size

def train(args: Dict, expert: Callable[[torch.Tensor, int], torch.Tensor], clone_model: torch.nn,
          obs_class_kappa: torch.nn, state_class_kappa: torch.nn, cbf: List[Dict], load_checkpoint: str,device: torch.device):
    # ___________________________________________
    # Create Trainig Data
    # ___________________________________________
    # Create buffer for training data
    n_pts = int(args['n_steps']*args['n_expert'])
    x_train = torch.zeros((n_pts, n_states))
    u_expert = torch.zeros((n_pts, n_controls))
    # Create random initial conditions
    n_x0s = args['n_expert']
    x0s = []
    
    if args['verbose']:
        data_gen_range = tqdm(range(n_x0s), ascii=True, desc="Generating data")
        data_gen_range.set_description("Generating training data...")
    else:
        data_gen_range = range(n_x0s)
        
    for i in data_gen_range:
            x0s.append(np.zeros(n_states))
            for dim in range(n_states - 1):
                x0s[i][dim] = np.random.uniform(state_space[dim][0], state_space[dim][1])
            # Run expert policy for n_steps
            x_train[i * args['n_steps']:(i + 1) * args['n_steps'], :], 
            u_expert[i * args['n_steps']:(i + 1) * args['n_steps'], :] \
                = expert(x0s[i], args['n_steps'] + 1)
    
    # Move data to device
    x_train = x_train.to(device)
    u_expert = u_expert.to(device)
    
    # ___________________________________________
    # DEFINE LOSS FUNCTION AND OPTIMIZER
    # ___________________________________________
    mse_loss_fn = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(
        list(clone_model.parameters()) 
        + list(obs_class_kappa.parameters()) 
        + list(state_class_kappa.parameters()), lr=args['lr'])
    
    # ___________________________________________
    # LOAD PREVIOUS MODEL IF AVAILABLE
    # ___________________________________________
    if args['load_model']:
        try:
            checkpoint = torch.load(load_checkpoint)
            clone_model.load_state_dict(checkpoint['clone_model_state_dict'])
            obs_class_kappa.load_state_dict(checkpoint['obs_class_kappa_state_dict'])
            state_class_kappa.load_state_dict(checkpoint['state_class_kappa_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loss = checkpoint['loss']
            if args['verbose']:
                print("Loaded model from {load_checkpoint} at epoch {curr_epoch} with loss {loss}")
        except FileNotFoundError:
            print(f"Checkpoint {load_checkpoint} not found. Starting from scratch.")   
    
    # ___________________________________________
    # TRAINING LOOP
    # ___________________________________________
    # current min loss
    curr_min_loss = 1e10
    # Optimize in minibatches
    for epoch in args['epochs']:
        # Shuffle data at the beginning of each epoch
        idx = np.random.permutation(n_pts)
        # reset loss
        loss_accum = 0
        # Loop over minibatches
        if args['verbose']:
            train_range = tqdm(range(0, n_pts, args['batch_size']), ascii=True, desc="Training Epoch")
            train_range.set_description("Training...")
        else:
            train_range = range(0, n_pts, args['batch_size'])
        for i in train_range:
            # Get minibatch
            x_batch = x_train[idx[i:i + args['batch_size']], :]
            u_batch = u_expert[idx[i:i + args['batch_size']], :]
            # Compute CBF parameters
            cbf_params = compute_cbf_params(x_batch, u_batch, cbf, device)
            # Compute loss
            loss = mse_loss_fn(clone_model(x_batch), u_batch) + barrier_loss(u_batch, cbf_params)
            # Accumulate loss
            loss_accum += loss.item()
            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    

def main():
    # ___________________________________________
    parser = argparse.ArgumentParser(description="Define the policy")
    parser.add_argument('--policy', type=str, default='umnn', help='Class kappa type for cbf {umnn, mlp, linear}')
    parser.add_argument('--batch_size', type=int, default=124, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--n_hidden', type=int, default=3, help='Number of hidden layers for training')
    parser.add_argument('--n_units', type=int, default=32, help='Number of units per hidden layer for training')
    parser.add_argument('--train', type=bool, default=True, help='Train the policy')
    parser.add_argument('--path', type=str, default="..\\data\\", help='Path to save the policy')
    parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
                        help='Name of the policy')
    parser.add_argument('--notes', type=str, default="", help='Notes for the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='Load the policy')
    parser.add_argument('--n_steps', type=int, default=10, help='Horizon for the expert policy')
    parser.add_argument('--n_expert', type=int, default=100, help='Number of expert trajectories')
    parser.add_argument('--verbose', type=bool, default=True, help='Print training information')
    # Parse the arguments
    args = parser.parse_args()
    # Convert to dictionary
    args = vars(args) 
    # define path to save the model
    path = args['path'] + args['name'] 
    # Create a text file that contains the arguments and notes
    with open(os.path.join(path,"args.txt"), "w") as f:
        f.write("Arguments: " + str(args))
    model_path = os.path.join(path,"model.pt")
    # ___________________________________________
    # Define the problem
    # ___________________________________________
    # Define goal state
    x_goal = torch.tensor([2.5, 0.001, 0.5, 0.0]).to(device)
    # Define obstacle
    radius = 0.5
    margin = 0.1
    center = [0.0, 0.0]
    x_obstacle = torch.tensor([center]).to(device).requires_grad_(False)
    r_obstacle = torch.tensor([radius + margin]).to(device).requires_grad_(False)
    # ___________________________________________
    # Define aV-CBF-CLF policy
    # ___________________________________________
    # Define the total number of cbf
    n_cbf = 3  # Number of CBF constraints [b_radius, b_v_min, b_v_max]   
    cbf_rel_degree = [1, 1, 1]
    # Define Number of cbf and clf constraints
    cv = 2*torch.pi
    av_p = torch.tensor([0.21, 3.5]).to(device).requires_grad_(False)
    aV = lambda x, p: p[0]*x[3]**2+p[1]*(x[3]+cv)**2  
    # Define CBF constraints structure
    distance_cbf_aV = {'type': 'distance_trans',
                    'f': lambda x, x_obst, radius: (x[0] - x_obst[0,0]) ** 2 + (x[1] - x_obst[0,1]) ** 2 - radius ** 2  - aV(x, av_p),
                    'f_args': [x_obstacle, r_obstacle], 
                    'rel_degree': 1,
                    'slack': False,
                    'alpha_args':[x_obstacle, r_obstacle], 
                    }  
    vel_min_cbf = {'type': 'state_constraint',
                   'f': lambda x, v_min: x[2] - v_min,
                   'f_args': [0.01],
                   'rel_degree': 1,
                   'slack': False,
                   'alpha_args':[x_obstacle, r_obstacle], 
                   }
    vel_max_cbf = {'type': 'state_constraint',
                   'f': lambda x, v_max: v_max - x[2],
                   'f_args': [1.0],
                   'rel_degree': 1,
                   'slack': False,
                   'alpha_args':[x_obstacle, r_obstacle], 
                   }
    cbf = [distance_cbf_aV, vel_min_cbf, vel_max_cbf]
    # -------------------------------------------
    # Define differential  CBF solver (diff QP)
    barrier_layer = BarrierNetLayer(
            n_state_dims=n_states,
            n_control_dims=n_controls,
            n_cbf=len(cbf),
            cbf_rel_degree=cbf_rel_degree,
            control_bounds=control_bounds,
            device=device,
        )
    # ___________________________________________
    # DEFINE LEARNING MODELS
    # ___________________________________________
    # -------------------------------------------
    # Define expert policy
    mpc_expert = lambda x0, hor: define_dubins_expert(x0, n_steps=hor)
    # -------------------------------------------
    # Define cloning policy model
    policy_nn = MLP(n_states, n_controls, n_hidden=args['n_hidden'],
                    n_units=args['n_units']).to(device)
    # Define MLP class kappa (states + obstacle location + obstacle radius)-> state cbf constraints
    class_kappa_state = MLP(n_states+3, n_cbf-1, n_hidden=args['n_hidden'], 
                            n_units=args['n_units']).to(device)
    # Define class kappa function for obstacle cbf constraint
    if args['policy'] == 'mlp':
        class_kappa_obs = MLP(n_states+3, 1, n_hidden=args['n_hidden'],
                            n_units=args['n_units']).to(device)
    elif args['policy'] == 'umnn':
        class_kappa_obs = MonotonicNN(n_states+3, 
                                      args['n_units']*args['n_hidden'], nb_steps=50, dev=device).to(device)

    # -------------------------------------------
    # Policy parameters (CBF)
    # Linear class k fuctions only
    cntrl_weights = torch.tensor([1., 1.]).to(device) # Same weights for all policies
    # Distance cbf
    cbf_params = copy.deepcopy(cbf)
    # Define class kappa function for obstacle cbf constraint
    if args['policy'] == 'mlp': #linear class kappa with MLP
        cbf_params[0]['class_kappa'] = lambda psi, x, *args: class_kappa_obs(torch.cat([x, *args], dim=1))*psi
    elif args['policy'] == 'umnn': #parametric class kappa with UMNN
        cbf_params[0]['class_kappa'] = lambda psi, x, *args: class_kappa_obs(psi, torch.cat([x, *args], dim=1))
    # Define class kappa function for state cbf constraints
    cbf_params[1]['class_kappa'] = lambda psi, x, *args: class_kappa_state(torch.cat([x, *args], dim=1))[0]*psi
    cbf_params[2]['class_kappa'] = lambda psi, x, *args: class_kappa_state(torch.cat([x, *args], dim=1))[1]*psi
    # Define alpha varargs for obstacle cbf constraints
    cbf_params[0]['class_k_vargs'] = [x_obstacle, r_obstacle]
    cbf_params[1]['class_k_vargs'] = [x_obstacle, r_obstacle]
    cbf_params[2]['class_k_vargs'] = [x_obstacle, r_obstacle]

    # -------------------------------------------
    # Plot Policies
    # -------------------------------------------
    # Define initial conditions
    x0s = [
        np.array([-2.0, 0.1, 0.0, 0.0]),
        np.array([-2.0, 0.1, 0.0, 0.0]),
        np.array([-2.0, 0.2, 0.0, 0.0]),
        np.array([-2.0, 0.5, 0.0, 0.0]),
        np.array([-2.0, -0.1, 0.0, 0.0]),
        np.array([-2.0, -0.2, 0.0, 0.0]),
        np.array([-2.0, -0.5, 0.0, 0.0]),

    ]
    # Simulate and plot
    simulate_and_plot(policies, x_goal.cpu().numpy(), center, radius, margin, x0s)
