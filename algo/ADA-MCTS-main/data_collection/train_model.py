import pickle
import os
import sys
import autograd.numpy as np
from BNN.BayesianNeuralNetwork import BayesianNeuralNetwork
from BNN.ExperienceReplay import ExperienceReplay

# Setup
if not os.path.isdir('./results'):
    os.mkdir('results')

# Get domain from command line argument (default to frozenlake)
domain = sys.argv[1] if len(sys.argv) > 1 else 'frozenlake'

# Domain-specific parameters
if domain == 'frozenlake':
    num_dims = 2  # 2D grid coordinates
    num_actions = 4  # Left, Down, Right, Up
elif domain == 'paratransit':
    # PAPER-COMPLIANT: BNN learns p(s'|s,a) where s,s' are 16D observations (NO traffic_level!)
    # 16D observation: [epoch, V0.loc, V0.time, V0.occ, V1.loc, ..., V4.occ]
    # = 1 (global) + 3*5 (vehicles) = 16
    # Action is separately one-hot encoded (5D: 5 vehicles only, no reject)
    num_dims = 16
    num_actions = 5  # Number of actions (5 vehicles, no reject)
else:
    raise ValueError(f"Unknown domain: {domain}")

num_batch_instances = 1
bnn_hidden_layer_size = 25
bnn_num_hidden_layers = 3
bnn_network_weights = None
eps_min = 0.1
grid_beta = 0.1
state_diffs = False
num_wb = 5

# Load experience buffer
with open(f'data_buffer/{domain}_exp_buffer.pkl', 'rb') as f:
    exp_buffer = pickle.load(f)
exp_buffer_np = np.vstack(exp_buffer)
inst_indices = exp_buffer_np[:, 4].astype(int)

# Group experiences by instance
exp_dict = {idx: exp_buffer_np[inst_indices == idx] for idx in range(1)}

# Prepare input and output for BNN
X = np.array([np.hstack([exp_buffer_np[tt, 0], exp_buffer_np[tt, 1]]) for tt in range(exp_buffer_np.shape[0])])
y = np.array([exp_buffer_np[tt, 3] for tt in range(exp_buffer_np.shape[0])])

# PAPER-COMPLIANT: For paratransit, BNN learns complete state transition (16D → 16D)
if domain == 'paratransit':
    # y is next_state_obs (16D) from data_generation.py
    print(f"[train_model.py] PAPER-COMPLIANT Paratransit: BNN learns p(s'|s,a)")
    print(f"[train_model.py] y.shape = {y.shape} (should be [N, 16] for next_state_obs)")

    # Verify shape matches 16D observation
    if len(y.shape) == 1:
        # If somehow flattened, this is an error
        raise ValueError(f"Expected y to be 2D array of next_state_obs, got 1D shape {y.shape}")

    if y.shape[1] != 16:
        raise ValueError(f"Buffer data incompatible. Expected 16D next_state_obs, got {y.shape[1]}D. "
                        f"This likely means you're using old buffer data with different dimensions. "
                        f"Please delete old buffers and re-run data_generation.py")

    num_output_dims = 16  # BNN predicts complete next state observation
else:
    # For frozen lake, keep original behavior
    num_output_dims = num_dims
    if state_diffs:
        y -= X[:, :num_dims]

# Set up parameters for Bayesian Neural Network
relu = lambda x: np.maximum(x, 0.)
param_set = {
    'bnn_layer_sizes': [num_dims + num_actions + num_wb] + [bnn_hidden_layer_size] * bnn_num_hidden_layers + [num_output_dims * 2],
    'weight_count': num_wb,
    'num_state_dims': num_output_dims,  # Use num_output_dims for BNN output shape
    'bnn_num_samples': 50,
    'bnn_batch_size': 32,
    'num_strata_samples': 5,
    'bnn_training_epochs': 1,
    'bnn_v_prior': 1,
    'bnn_learning_rate': 0.0005,
    'bnn_alpha': 0.5,
    'wb_num_epochs': 1,
    'wb_learning_rate': 0.0005
}

print(f"Training BNN for domain: {domain}")
print(f"State dimensions: {num_dims}, Actions: {num_actions}")
print(f"Experience buffer size: {exp_buffer_np.shape[0]}")

# Initialize latent weights for each instance
full_task_weights = np.random.normal(0., 0.1, (1, num_wb))

# Initialize Bayesian Neural Network
network = BayesianNeuralNetwork(param_set, nonlinearity=relu)

# Compute error before training
l2_errors = network.get_td_error(np.hstack((X, full_task_weights[inst_indices])), y, location=0.0, scale=1.0, by_dim=False)
print(f"Before training: Mean Error: {np.mean(l2_errors)}, Std Error: {np.std(l2_errors)}")

# Function to get random sample of indices
def get_random_sample(start, stop, size):
    return np.random.choice(np.arange(start, stop), size=size, replace=False)

# Train BNN and update latent weights
sample_size = 1000
for i in range(5):
    # Update BNN network weights
    network.fit_network(exp_buffer_np, full_task_weights, 0, state_diffs=state_diffs, use_all_exp=True)
    print(f'Finished BNN update {i}')

    # Compute error on random sample of transitions
    sample_indices = get_random_sample(0, X.shape[0], sample_size)
    l2_errors = network.get_td_error(np.hstack((X[sample_indices], full_task_weights[inst_indices[sample_indices]])), y[sample_indices], location=0.0, scale=1.0, by_dim=False)
    print(f"After BNN update: iter: {i}, Mean Error: {np.mean(l2_errors)}, Std Error: {np.std(l2_errors)}")

    # Update latent weights
    for inst in np.random.permutation(1):
        full_task_weights[inst, :] = network.optimize_latent_weighting_stochastic(exp_dict[inst], np.atleast_2d(full_task_weights[inst, :]), 0, state_diffs=state_diffs, use_all_exp=True)
    print(f'Finished WB update {i}')

    # Compute error after latent weight update
    l2_errors = network.get_td_error(np.hstack((X[sample_indices], full_task_weights[inst_indices[sample_indices]])), y[sample_indices], location=0.0, scale=1.0, by_dim=False)
    print(f"After Latent update: iter: {i}, Mean Error: {np.mean(l2_errors)}, Std Error: {np.std(l2_errors)}")

    # Save model weights
    with open(f'models/{domain}_network_weights_itr_{i}', 'wb') as f:
        pickle.dump(network.weights, f)
    with open(f'models/{domain}_latent_weights_itr_{i}', 'wb') as f:
        pickle.dump(full_task_weights, f)