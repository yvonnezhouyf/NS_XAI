import math
import random
import sys
import pickle
from BNN.BayesianNeuralNetwork import *
import autograd.numpy as np
import utils.distribution as distribution
import matplotlib.pyplot as plt
import time
from collections import Counter
from multiprocessing import Pool
from HiPMDP import HiPMDP, train_model
import logging
from adamcts import MCTS
import os
from adamcts import Node

# Get domain from command line or environment variable
domain = sys.argv[1] if len(sys.argv) > 1 else 'frozenlake'

# Import appropriate model class based on domain
if domain == 'frozenlake':
    from nsfrozenlake.nsfrozenlake_v0 import NSFrozenLakeV0 as model
    num_actions = 4
elif domain == 'paratransit':
    from nsparatransit.nsparatransit_v0 import NSParatransitV0 as model
    # Paratransit has 5 actions (5 vehicles, no reject)
    num_actions = 5
else:
    raise ValueError(f"Unknown domain: {domain}")


def __encode_action(action, domain_name=domain):
    """One-hot encodes the integer action supplied."""
    # Paratransit has 5 actions (5 vehicles, no reject)
    n_actions = 4 if domain_name == 'frozenlake' else 5
    a = np.array([0] * n_actions)
    a[action] = 1
    return a

def run_task(seed, domain_name, network_weights1, latent_weights1, latent_std_input, bnn_hidden_layer_size, bnn_num_hidden_layers, preset_hidden_params, run_type, global_buffer_D=None, N_min=10, N_interval=20, max_D_size=10000):
    """
    Run a single task episode with ADA-MCTS.

    Args:
        seed: Random seed for reproducibility
        domain_name: Domain name ('frozenlake' or 'paratransit')
        network_weights1: Pre-trained BNN network weights
        latent_weights1: Pre-trained latent weights (mean of P_W)
        latent_std_input: Standard deviation of P_W (updated across episodes)
        bnn_hidden_layer_size: BNN hidden layer size
        bnn_num_hidden_layers: Number of BNN hidden layers
        preset_hidden_params: Hidden parameters for HiP-MDP
        run_type: Run type ('full' or other)
        global_buffer_D: Global replay buffer D (shared across episodes)
        N_min: Minimum buffer size before first training (Paper hyperparameter)
        N_interval: Update frequency in steps (Paper hyperparameter)
        max_D_size: Maximum size of global buffer D
    """
    # BNN1 = M̂k (current model, will be updated with new data)
    hipmdp1 = HiPMDP(domain_name, preset_hidden_params,
                     run_type=run_type,
                     bnn_hidden_layer_size=bnn_hidden_layer_size,
                     bnn_num_hidden_layers=bnn_num_hidden_layers,
                     bnn_network_weights=network_weights1)
    hipmdp1._HiPMDP__initialize_BNN()

    # Paper Algorithm 1, Line 2: Sample w_b ~ P_W
    # Use P_W distribution parameters from previous episode (updated across episodes)
    latent_mean = latent_weights1.reshape(latent_weights1.shape[1], )
    latent_std = latent_std_input  # Use passed-in std (updated across episodes)

    # Sample w_b from P_W = N(latent_mean, latent_std^2)
    np.random.seed(seed)
    weight_set1 = latent_mean + latent_std * np.random.randn(len(latent_mean))

    # Paper Algorithm 1, Line 3: Initialize global replay buffer D
    # D stores experiences across multiple episodes for better generalization
    if global_buffer_D is None:
        global_buffer_D = []

    # Initialize task based on domain
    if domain_name == 'frozenlake':
        task = model()
        task.reset(1, seed)
    elif domain_name == 'paratransit':
        task = model(
            seed=seed,
            w_fulfillment=1.0,
            w_timing=1.0,
            n_requests=10,
            max_time=480,       # Match data collection
            n_vehicles=5,
            vehicle_capacity=3,
            n_nodes=20,
            traffic_condition=0.3
        )
        task.reset()

    # PAPER-COMPLIANT: BNN2 = M̂k-1 (model from PREVIOUS episode/environment segment)
    # According to paper's DPAS (Algorithm 2), we compare:
    # - M̂k: current model (BNN1) that gets updated during THIS episode
    # - M̂k-1: model from PREVIOUS episode (BNN2) - initialized ONCE at episode start
    #
    # CRITICAL FIX (Issue 1): M̂k-1 represents "previous environment segment's model",
    # NOT "snapshot before each training step". It should remain FIXED throughout the episode.
    #
    # At episode start: BNN1 initialized from previous episode's final weights,
    #                   BNN2 represents the previous episode's model (stays frozen)
    # → DPAS compares current updated model vs. previous episode's model
    hipmdp2 = HiPMDP(domain_name, preset_hidden_params, run_type=run_type,
                     bnn_hidden_layer_size=bnn_hidden_layer_size,
                     bnn_num_hidden_layers=bnn_num_hidden_layers,
                     bnn_network_weights=network_weights1)  # Initialize with previous episode's weights
    hipmdp2._HiPMDP__initialize_BNN()
    weight_set2 = latent_mean.copy()

    # CRITICAL: M̂k-1 (hipmdp2, weight_set2) is now FROZEN for this entire episode
    # It will NOT be updated during the episode, only hipmdp1 and weight_set1 will be updated

    # PAPER-COMPLIANT FIX (Issue 8): Use global buffer D for pretraining/initialization
    # Paper Algorithm 1, Line 3: "Initialize global replay buffer D"
    # Paper Algorithm 1, Line 10: "Store... in D_b AND D"
    # The intent: D accumulates experiences across episodes for better generalization
    # Use D for initial model warmup, then use D_b for online adaptation during episode
    if global_buffer_D and len(global_buffer_D) >= N_min:
        # Save global buffer D for training
        os.makedirs('data_buffer', exist_ok=True)
        exp_list_D_pretrain = np.vstack(global_buffer_D)
        with open('data_buffer/{}_{}_exp_buffer_D'.format(domain_name, seed), 'wb') as f:
            pickle.dump(exp_list_D_pretrain, f)

        # Pretrain M̂k using global buffer D for initialization
        print(f"   [PRETRAIN] Warming up M̂k with {len(global_buffer_D)} experiences from global buffer D")
        updated_network_weights, updated_latent_weights, _, _, latent_variance = train_model(
            seed, domain_name, hipmdp1, weight_set1, 100, 100, 0, use_global_buffer=True
        )
        # Apply pretrained weights to M̂k
        hipmdp1.network.weights = updated_network_weights
        weight_set1 = updated_latent_weights
        latent_mean = updated_latent_weights
        latent_std = np.sqrt(latent_variance + 1e-8)
        print(f"   [PRETRAIN] Completed. M̂k initialized with knowledge from {len(global_buffer_D)} past experiences.")

    reward = 0
    # Paper Algorithm 1, Line 4: Initialize episode buffer D_b
    episode_buffer_bnn = []  # D_b in the paper
    best_network_error = 100
    best_latent_error = 100
    local_converge_count = 0

    # Print episode header for paratransit
    if domain_name == 'paratransit':
        print(f"\n{'='*60}")
        print(f"  Running Paratransit Episode (Seed {seed})")
        print(f"  Reward Weights: Fulfillment=1.0, Timing=1.0")
        print(f"{'='*60}")

    while not task.is_done():
        # Get state for MCTS (domain-specific)
        if domain_name == 'frozenlake':
            mcts_state = task.state.index  # Integer state index
        elif domain_name == 'paratransit':
            mcts_state = task.state  # ParatransitState object (carries full fleet state)
        else:
            mcts_state = task.state

        # PAPER-COMPLIANT: Use state_to_observation() instead of observe()
        # This ensures we never leak hidden parameters (traffic_level for paratransit, intend_prob for frozenlake)
        state_obs = task.state_to_observation(task.state)
        mcts_instance = MCTS(state_obs, mcts_state, hipmdp1, hipmdp2, weight_set1, weight_set2, 0, task,
                             0.02, False, False)

        # DEBUG: Print before MCTS search to detect if it hangs
        if domain_name == 'paratransit':
            print(f"  [MCTS] Starting search for decision {mcts_state.decision_epoch if hasattr(mcts_state, 'decision_epoch') else 'unknown'}...", flush=True)

        mcts_instance.search(3000)

        if domain_name == 'paratransit':
            print(f"  [MCTS] Search completed.", flush=True)

        best_action = mcts_instance.best_action()

        # Get state observation for BNN (domain-aware, consistent with data collection)
        # PAPER-COMPLIANT: Always use state_to_observation() to avoid leaking hidden parameters
        state_obs = task.state_to_observation(task.state)

        next_state, reward, done, info = task.step(best_action)

        # Get next state observation for BNN (domain-aware)
        if domain_name == 'paratransit':
            # PAPER-COMPLIANT: BNN learns p(s'|s,a) - full next state observation
            # Use state_to_observation() to get 16D next_state_obs (NO traffic_level)
            next_state_obs = task.state_to_observation(next_state)
        else:
            next_state_obs = next_state

        # PAPER-COMPLIANT: For ALL domains, store complete next_state_obs
        # BNN learns p(s'|s,a) where s,s' are observable state features
        # For paratransit: 16D [norm_epoch, v0_loc, v0_time, v0_occ, ..., v4_loc, v4_time, v4_occ]
        # For frozen lake: 2D [row, col]
        target = next_state_obs

        # Paper Algorithm 1, Line 10: Add (s_t, a_t, r_t, s_{t+1}, w_b) to D_b AND D
        # Store the sampled latent weight w_b (weight_set1) instead of instance id
        # For ALL domains: stores complete next_state_obs (16D for paratransit, 2D for frozen lake)
        transition = np.array([state_obs, __encode_action(best_action, domain_name), reward, target, weight_set1], dtype=object).reshape([1, 5])

        # Add to episode buffer D_b
        episode_buffer_bnn.append(transition)

        # IMMEDIATELY add to global buffer D (Paper: each step adds to both D and D_b)
        global_buffer_D.append(transition)

        # Limit D size to prevent memory overflow
        if len(global_buffer_D) > max_D_size:
            global_buffer_D.pop(0)  # Remove oldest transition

        # Print execution trace for paratransit
        if domain_name == 'paratransit':
            n_requests = task.n_requests
            current_step = mcts_state.decision_epoch if hasattr(mcts_state, 'decision_epoch') else 0
            print(f"  [Step {current_step+1}/{n_requests}] Assigned Vehicle {best_action} → Reward: {reward:.2f}")
            if 'pickup_delay' in info:
                pickup_status = "✓ On-time" if info['pickup_delay'] <= 0 else f"✗ Late {info['pickup_delay']:.1f}min"
                dropoff_status = "✓ On-time" if info['dropoff_delay'] <= 0 else f"✗ Late {info['dropoff_delay']:.1f}min"
                print(f"           Pickup: {pickup_status}, Dropoff: {dropoff_status}")

        # Paper Algorithm 1: Update model periodically during episode
        # Uses configurable hyperparameters N_min and N_interval
        if len(episode_buffer_bnn) >= N_min and len(episode_buffer_bnn) % N_interval == 0:
            # Paper Algorithm 1: Prepare buffers for training
            # D_b: current episode buffer - already a list of arrays
            # D: global buffer (already updated in real-time above, no need to extend here)

            # Save both D_b (episode buffer) and D (global buffer) to disk
            os.makedirs('data_buffer', exist_ok=True)
            # episode_buffer_bnn is already a list of (1,5) arrays, vstack will work
            exp_list_Db = np.vstack(episode_buffer_bnn)
            with open('data_buffer/{}_{}_exp_buffer_Db'.format(domain_name, seed), 'wb') as f:
                pickle.dump(exp_list_Db, f)
            exp_list_D = np.vstack(global_buffer_D)
            with open('data_buffer/{}_{}_exp_buffer_D'.format(domain_name, seed), 'wb') as f:
                pickle.dump(exp_list_D, f)

            # PAPER-COMPLIANT FIX (Issue 1): DO NOT snapshot BNN1 → BNN2 here!
            # M̂k-1 (hipmdp2, weight_set2) represents the PREVIOUS episode's model
            # and should remain FROZEN throughout this episode.
            # Only M̂k (hipmdp1, weight_set1) gets updated during the episode.
            #
            # The old code incorrectly treated M̂k-1 as "snapshot before each training step",
            # but the paper defines it as "model from previous environment segment" (previous episode).

            # Paper Algorithm 1, Line 6-7: Update W_k and w_b from D_b (NOT from global D)
            # D_b contains only current episode's experiences for adaptation to non-stationarity
            updated_network_weights, updated_latent_weights, best_network_error, best_latent_error, latent_variance = train_model(
                seed, domain_name, hipmdp1, weight_set1, best_network_error, best_latent_error, local_converge_count, use_global_buffer=False
            )

            # APPLY updated weights to BNN1 (M̂k)
            hipmdp1.network.weights = updated_network_weights
            weight_set1 = updated_latent_weights

            # Paper: Update P_W distribution based on optimized w_b
            # The optimized w_b becomes the new mean, variance from buffer samples becomes new std
            latent_mean = updated_latent_weights
            latent_std = np.sqrt(latent_variance + 1e-8)  # Add small epsilon for numerical stability

            # Clear BNN cache so new predictions use updated weights
            Node.bnn_cache = {}
            Node.aleatoric_cache = {}

            print(f"   [TRAIN] Updated M̂k (hipmdp1) after {len(episode_buffer_bnn)} steps. M̂k-1 (hipmdp2) remains frozen from episode start. Error: {best_latent_error:.4f}")
            print(f"   [P_W] Updated distribution - mean: {latent_mean[:3]}, std: {latent_std[:3]}")

        # Render (only for frozen lake)
        if domain_name == 'frozenlake':
            task.render()
        time.sleep(0.01)

    # Print episode summary for paratransit
    if domain_name == 'paratransit':
        print(f"\n{'='*60}")
        print(f"  Episode Completed (Seed {seed})")
        print(f"  Final Cumulative Reward: {reward:.2f}")
        print(f"  Total Steps: {len(episode_buffer_bnn)}")
        # Calculate statistics from task
        num_fulfilled = sum(1 for status in task.request_status.values()
                           if status in ["in-transit", "dropped-off"])
        fulfillment_rate = num_fulfilled / task.n_requests
        print(f"  Requests Fulfilled: {num_fulfilled}/{task.n_requests} ({fulfillment_rate*100:.1f}%)")
        print(f"{'='*60}\n")

    # Return reward, updated global buffer D, updated parameters for next episode
    # Paper: P_W (latent_mean, latent_std) and W_k (network weights) should persist across episodes
    return reward, global_buffer_D, latent_mean, latent_std, hipmdp1.network.weights

if __name__ == '__main__':
    # Set multiprocessing start method for CUDA compatibility
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    # domain variable already set at line 18 from command line argument
    with open('models/{}_network_weights_itr_2'.format(domain), 'rb') as f1:
        network_weights1 = pickle.load(f1)
    with open('models/{}_latent_weights_itr_2'.format(domain), 'rb') as f3:
        latent_weights1 = pickle.load(f3)

    bnn_hidden_layer_size = 25
    bnn_num_hidden_layers = 3
    preset_hidden_params = [{'latent_code': 1}]
    run_type = "full"
    seeds = [2]  # Single seed for faster testing

    # Paper Algorithm 1, Line 3: Initialize global buffer D (shared across episodes)
    global_buffer_D = []

    # Initialize P_W distribution parameters (will be updated across episodes)
    latent_mean = latent_weights1.reshape(latent_weights1.shape[1], )
    latent_std = np.ones_like(latent_mean) * 0.1

    # Initialize network weights (will be updated across episodes)
    current_network_weights = network_weights1

    # Use sequential execution for paratransit to avoid nested multiprocessing issues
    # MCTS already uses parallel BNN predictions internally
    if domain == 'paratransit':
        results = []
        for seed in seeds:
            result, global_buffer_D, latent_mean, latent_std, current_network_weights = run_task(
                seed, domain, current_network_weights,
                latent_mean.reshape(1, -1),  # Pass mean as (1, n) for compatibility
                latent_std,  # Pass updated std from previous episode
                bnn_hidden_layer_size, bnn_num_hidden_layers, preset_hidden_params, run_type, global_buffer_D,
                N_min=5, N_interval=5
            )
            results.append(result)
    else:
        # For frozen lake, run episodes sequentially to maintain global buffer D
        results = []
        for seed in seeds:
            result, global_buffer_D, latent_mean, latent_std, current_network_weights = run_task(
                seed, domain, current_network_weights,
                latent_mean.reshape(1, -1),  # Pass mean as (1, n) for compatibility
                latent_std,  # Pass updated std from previous episode
                bnn_hidden_layer_size, bnn_num_hidden_layers, preset_hidden_params, run_type, global_buffer_D
            )
            results.append(result)

    # Set up logging configuration at the beginning
    logging.basicConfig(
        filename='results.log',
        filemode='a',  # Append to the file
        level=logging.INFO,  # Set the minimum level of logging to INFO
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    # Create a logger
    logger = logging.getLogger(__name__)

    # Log header
    logger.info("Discounted Return, Count")

    # Log final results if needed
    for seed, reward in zip(seeds, results):
        if reward is not None:
            logger.info(f"Final Reward for Seed {seed}: {reward}")
