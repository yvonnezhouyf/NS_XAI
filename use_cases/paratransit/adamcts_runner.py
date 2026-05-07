"""ADA-MCTS Runner for Paratransit Domain (Python 3.8)

This module runs ADA-MCTS for paratransit vehicle assignment and captures
data needed for XAI explanations, including:
1. Per-epoch MCTS trees and PCTL evidence
2. MDP comparison data (MDP_{t-n} vs MDP_t) for non-stationarity analysis
3. Counter-intuitive assignment detection
"""

import hashlib
import json
import math
import os
import io
import pickle
import random
import sys
import traceback
from copy import deepcopy

import numpy as np

# Import autograd.numpy.random for reproducibility (used by BNN)
try:
    import autograd.numpy.random as npr
    HAS_AUTOGRAD = True
except ImportError:
    HAS_AUTOGRAD = False

# Import torch for GPU BNN reproducibility (optional)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Add paths for algo and environment
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up to ns_explainer/
ada_mcts_path = os.path.join(project_root, 'algo', 'ADA-MCTS-main')

sys.path.insert(0, ada_mcts_path)

from adamcts import MCTS
from HiPMDP import HiPMDP, train_model
from nsparatransit.nsparatransit_v0 import NSParatransitV0, VehicleState, PassengerRequest, ParatransitState

DEBUG_MODE = True

# ============================================================================
# Counter-Intuitive Assignment Detection
# ============================================================================

def detect_counter_intuitive_assignment(assignment_detail, mcts_instance, env, decision_epoch):
    """
    Detect if an assignment is counter-intuitive (closest vehicle not chosen).

    An assignment is counter-intuitive when:
    - The assigned vehicle is farther than the closest vehicle

    This simple distance-based detection captures the essence of counter-intuitive
    decisions: ADA-MCTS chose a farther vehicle over a closer one, indicating
    that factors beyond proximity (like workload, timing, traffic adaptation)
    influenced the decision.

    Args:
        assignment_detail: Dict with assignment info (from run_ada_mcts_paratransit)
        mcts_instance: MCTS object with current tree (unused but kept for API compatibility)
        env: NSParatransitV0 environment (unused but kept for API compatibility)
        decision_epoch: Current decision epoch

    Returns:
        Dict with counter-intuitive analysis, or None if not counter-intuitive
    """
    closest_vehicle = assignment_detail.get('closest_vehicle')
    assigned_vehicle = assignment_detail.get('assigned_vehicle')
    closest_distance = assignment_detail.get('closest_distance')
    assigned_distance = assignment_detail.get('chosen_vehicle_distance')

    # Not counter-intuitive if closest vehicle was assigned
    if closest_vehicle is None or assigned_vehicle == closest_vehicle:
        return None

    # Not counter-intuitive if we don't have distance data
    if closest_distance is None or assigned_distance is None:
        return None

    # Counter-intuitive: assigned vehicle is farther than closest vehicle
    # (with small tolerance for floating point comparison)
    if assigned_distance <= closest_distance + 0.01:
        return None

    # Extract Q-values from MCTS tree if available
    q_assigned = 0.0
    q_closest = 0.0
    if mcts_instance is not None and hasattr(mcts_instance, 'root'):
        root = mcts_instance.root
        for child in getattr(root, 'children', []) or []:
            if hasattr(child, 'action'):
                if child.action == assigned_vehicle:
                    q_assigned = float(getattr(child, 'q', 0.0))
                elif child.action == closest_vehicle:
                    q_closest = float(getattr(child, 'q', 0.0))

    return {
        'is_counter_intuitive': True,
        'decision_epoch': decision_epoch,
        'assigned_vehicle': assigned_vehicle,
        'closest_vehicle': closest_vehicle,
        'assigned_distance': float(assigned_distance),
        'closest_distance': float(closest_distance),
        'distance_difference': float(assigned_distance - closest_distance),
        'q_assigned': q_assigned,
        'q_closest': q_closest,
    }


def extract_pctl_metrics_for_vehicle(mcts_instance, vehicle_id, env, decision_epoch):
    """
    Extract MCTS metrics for a specific vehicle from MCTS tree.
    
    Returns metrics like:
    - Visit counts and Q-value
    - Uncertainty information (epistemic, aleatoric)
    - Branch information (pessimistic vs regular)
    
    Args:
        mcts_instance: MCTS object with tree
        vehicle_id: Which vehicle to analyze
        env: Environment
        decision_epoch: Current epoch
        
    Returns:
        Dict of metrics
    """
    if mcts_instance is None or not hasattr(mcts_instance, 'root'):
        return {}
    
    root = mcts_instance.root
    metrics = {
        'vehicle_id': vehicle_id,
        'decision_epoch': decision_epoch,
    }
    
    # Find the chance node for this vehicle
    chance_node = None
    for child in getattr(root, 'children', []) or []:
        if (hasattr(child, 'type') and child.type == "chance" and
            hasattr(child, 'action') and child.action == vehicle_id):
            chance_node = child
            break
    
    if chance_node is None:
        return metrics
    
    # Extract basic metrics
    metrics['visits'] = getattr(chance_node, 'visits', 0)
    metrics['q_value'] = float(getattr(chance_node, 'q', 0.0))
    metrics['value'] = float(getattr(chance_node, 'v', 0.0))
    
    # Extract uncertainty information (for DPAS analysis)
    if hasattr(chance_node, 'epistemic_uncertainty'):
        metrics['epistemic_uncertainty'] = float(chance_node.epistemic_uncertainty or 0.0)
    if hasattr(chance_node, 'aleatoric_uncertainty'):
        metrics['aleatoric_uncertainty'] = float(chance_node.aleatoric_uncertainty or 0.0)
    
    # Extract branch information (pessimistic vs regular sampling)
    if hasattr(chance_node, 'explain') and chance_node.explain:
        metrics['branch'] = chance_node.explain.get('branch', 'unknown')
        metrics['delta_E'] = chance_node.explain.get('delta_E')
        metrics['delta_A'] = chance_node.explain.get('delta_A')
    
    return metrics


def _reconstruct_paratransit_state(state_dict):
    """Reconstruct a ParatransitState from a serialized dict.

    Used by rebuild_mcts_with_old_weights to convert dict-only snapshots
    back into live objects that env.state_to_observation() and MCTS can use.
    """
    if state_dict is None:
        return None
    if not isinstance(state_dict, dict):
        return state_dict  # Already an object

    vehicles = []
    for v in state_dict.get('vehicles', []):
        vehicles.append(VehicleState(
            vehicle_id=v['vehicle_id'],
            current_location=v['current_location'],
            current_time=v['current_time'],
            current_occupancy=v['current_occupancy'],
            capacity=v['capacity'],
            route=v.get('route', []),
            next_time=v.get('next_time'),
        ))

    cr = state_dict.get('current_request')
    current_request = None
    if cr is not None:
        current_request = PassengerRequest(
            request_id=cr['request_id'],
            pickup_node=cr['pickup_node'],
            dropoff_node=cr['dropoff_node'],
            request_time=cr['request_time'],
            earliest_pickup=cr['earliest_pickup'],
            latest_dropoff=cr['latest_dropoff'],
        )

    state = ParatransitState(
        decision_epoch=state_dict['decision_epoch'],
        current_request=current_request,
        vehicles=vehicles,
        traffic_level=state_dict.get('traffic_level', 0.3),
    )
    return state


def _serialize_paratransit_state(state):
    """
    Serialize ParatransitState to a structured dict for JSON storage.

    This enables formula_evaluator to access vehicles/current_request from
    serialized trees for ETA calculations.
    
    Args:
        state: ParatransitState object or None
        
    Returns:
        Dict representation of state, or None if state is None
    """
    if state is None:
        return None
    
    # Handle case where state is already a dict (deserialized)
    if isinstance(state, dict):
        return state
    
    # Handle case where state is a string (legacy format)
    if isinstance(state, str):
        return {'_str_repr': state}
    
    result = {
        'decision_epoch': getattr(state, 'decision_epoch', None),
        'traffic_level': getattr(state, 'traffic_level', None),
    }
    
    # Serialize current_request
    current_request = getattr(state, 'current_request', None)
    if current_request is not None:
        result['current_request'] = {
            'request_id': getattr(current_request, 'request_id', None),
            'pickup_node': getattr(current_request, 'pickup_node', None),
            'dropoff_node': getattr(current_request, 'dropoff_node', None),
            'request_time': getattr(current_request, 'request_time', None),
            'earliest_pickup': getattr(current_request, 'earliest_pickup', None),
            'latest_dropoff': getattr(current_request, 'latest_dropoff', None),
        }
    else:
        result['current_request'] = None
    
    # Serialize vehicles
    vehicles = getattr(state, 'vehicles', [])
    result['vehicles'] = []
    for v in vehicles:
        v_dict = {
            'vehicle_id': getattr(v, 'vehicle_id', None),
            'current_location': getattr(v, 'current_location', None),
            'current_time': getattr(v, 'current_time', None),
            'current_occupancy': getattr(v, 'current_occupancy', None),
            'capacity': getattr(v, 'capacity', None),
            'next_time': getattr(v, 'next_time', None),
        }
        # Serialize route as list of (request_id, location) tuples
        route = getattr(v, 'route', [])
        v_dict['route'] = [(entry[0], entry[1]) if isinstance(entry, (list, tuple)) else entry 
                          for entry in route]
        result['vehicles'].append(v_dict)
    
    return result


def serialize_node_for_json(node, max_depth=8, current_depth=0):
    """
    Serialize MCTS node to JSON-compatible dict.

    Args:
        node: MCTS Node object
        max_depth: Maximum depth to serialize (8 for policy-conditioned PCTL tree walk)
        current_depth: Current recursion depth

    Returns:
        Dict representation of node

    Trace storage strategy:
    - 'rollout_traces': Node's DIRECT traces only (prefix-correct, semantically valid
                        for X/F<=k/G/U evaluation starting from this node)
    - No subtree aggregation is stored to avoid step-shift and double-counting

    """
    if node is None or current_depth >= max_depth:
        return None

    visits = getattr(node, 'visits', 0)
    cumulative_value = float(getattr(node, 'value', 0.0))
    q_value = cumulative_value / visits if visits > 0 else 0.0

    node_dict = {
        'visits': visits,
        'value': cumulative_value,
        'q_value': q_value,
        'type': getattr(node, 'type', 'unknown'),
        'action': getattr(node, 'action', None),
        'state': _serialize_paratransit_state(getattr(node, 'state', None)),
        'children': []
    }

    # Add cached atomic propositions if available
    if hasattr(node, '_cached_props'):
        node_dict['atomic_props'] = node._cached_props

    # Add explain field (contains branch, uncertainties, etc.)
    if hasattr(node, 'explain'):
        node_dict['explain'] = node.explain

    # Add transition_ap for trace prefix reconstruction
    if hasattr(node, 'transition_ap') and node.transition_ap is not None:
        node_dict['transition_ap'] = node.transition_ap

    # Add rollout traces for trace-based PCTL evaluation
    #
    # CRITICAL: For action nodes (depth=1, chance nodes representing vehicle choices),
    # For action nodes (root's children), use direct rollout_traces only.
    # This avoids semantic issues with X/F<=k formulas by not mixing in
    # traces from deeper nodes that would shift step counts.
    direct_traces = getattr(node, 'rollout_traces', None)
    if direct_traces:
        node_dict['rollout_traces'] = direct_traces

    # Recursively serialize children (up to max_depth)
    if current_depth < max_depth - 1:
        for child in getattr(node, 'children', []) or []:
            child_dict = serialize_node_for_json(child, max_depth, current_depth + 1)
            if child_dict:
                node_dict['children'].append(child_dict)

    return node_dict


def run_ada_mcts_paratransit(n_vehicles=5, n_requests=10, max_iterations=5000, seed=42,
                              custom_traffic=None, store_per_step_trees=True,
                              early_stop_epoch=None, fixed_request_ids=None, data_path=None):
    """
    Run ADA-MCTS for paratransit vehicle assignment.

    Args:
        n_vehicles: Number of vehicles
        n_requests: Number of passenger requests
        max_iterations: MCTS iterations per decision epoch
        seed: Random seed
        custom_traffic: Optional custom traffic condition (for counterfactuals)
        store_per_step_trees: Whether to store trees at each decision epoch
        early_stop_epoch: If set, stop after this many epochs (for demo/testing)
        fixed_request_ids: If provided, use these specific request IDs for reproducibility

    Returns:
        Dict with MCTS data, environment info, and results
    """
    try:
        domain = 'paratransit'
        models_path = os.path.join(ada_mcts_path, 'models')

        # Load pre-trained BNN weights
        # According to ADA-MCTS paper Algorithm 1 & 2:
        # - weight_set1 (M̂k): current epoch's trained weights
        # - weight_set2 (M̂k-1): previous epoch's trained weights
        # DPAS uses δE = VarE(M̂k) - VarE(M̂k-1) for epistemic uncertainty comparison
        with open(os.path.join(models_path, f'{domain}_network_weights_itr_2'), 'rb') as f:
            network_weights = pickle.load(f)
        with open(os.path.join(models_path, f'{domain}_latent_weights_itr_2'), 'rb') as f:
            latent_weights = pickle.load(f)  # M̂k (current model)
        with open(os.path.join(models_path, f'{domain}_latent_weights_itr_1'), 'rb') as f:
            latent_weights_prev = pickle.load(f)  # M̂k-1 (previous model)

        # BNN configuration
        bnn_hidden_layer_size = 25
        bnn_num_hidden_layers = 3
        preset_hidden_params = [{'latent_code': 1}]
        run_type = "full"

        # Create HiPMDP models
        hipmdp = HiPMDP(
            domain,
            preset_hidden_params,
            run_type=run_type,
            bnn_hidden_layer_size=bnn_hidden_layer_size,
            bnn_num_hidden_layers=bnn_num_hidden_layers,
            bnn_network_weights=network_weights
        )
        hipmdp._HiPMDP__initialize_BNN()
        weight_set1 = latent_weights.reshape(latent_weights.shape[1], )

        # Use previous epoch's trained weights for M̂k-1 (paper-compliant DPAS)
        weight_set2 = latent_weights_prev.reshape(latent_weights_prev.shape[1], )

        # Set random seed for environment
        random.seed(seed)
        np.random.seed(seed)

        # Create paratransit environment
        traffic_condition = custom_traffic if custom_traffic is not None else 0.3
        env = NSParatransitV0(
            n_vehicles=n_vehicles,
            n_requests=n_requests,
            traffic_condition=traffic_condition,
            seed=seed,
            data_path=data_path,
            use_real_data=True,
            fixed_request_ids=fixed_request_ids,
            max_time=480
        )

        # BNN signature for hashing
        bnn_signature = f"bnn_{hashlib.md5(str(network_weights).encode()).hexdigest()[:8]}"

        # Storage for per-step data
        per_step_trees = {}
        recommended_assignments = []  # [(request_id, vehicle_id), ...]
        assignment_details = []  # Detailed info for each assignment

        # Track completed requests per vehicle for XAI display
        # completed_by_vehicle[v_id] = set of request_ids that vehicle v_id has completed (dropped off)
        completed_by_vehicle = [set() for _ in range(n_vehicles)]

        # DPAS history for TURNING_INTERVAL analysis
        # Tracks regular_pct per epoch for behavioral transition detection
        dpas_history = {}  # {epoch: {regular_pct, pessimistic_pct, avg_delta_E, avg_delta_A}}

        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0.0
        decision_epoch = 0

        print(f"\n[EXECUTING] Running MCTS policy for {n_requests} requests...")
        print(f"  Using {max_iterations} iterations per decision epoch")
        if early_stop_epoch is not None:
            print(f"  Early stop enabled: will stop after epoch {early_stop_epoch}")
        print()

        # Execute policy: assign each request using MCTS
        while not done and decision_epoch < n_requests:
            # Early stop for demo/testing
            if early_stop_epoch is not None and decision_epoch >= early_stop_epoch:
                print(f"\n[EARLY STOP] Stopping at epoch {decision_epoch} (early_stop_epoch={early_stop_epoch})")
                break
            
            # Create MCTS instance for this decision epoch
            # MCTS constructor signature: (initial_state_coordinate, initial_state_index,
            #                              bnn1, bnn2, ws1, ws2, time, task, threshold,
            #                              training_started, danger)
            # NOTE: For paratransit, initial_state_index should be the full state object
            # (not just an int like frozen lake), because env.step() expects self.state to
            # be a ParatransitState object
            #
            # DPAS (Dual-Phase Adaptive Sampling) mechanism:
            # Branch switching is now handled automatically based on epistemic/aleatoric uncertainty
            # comparison (see adamcts.py expand() method). No time-based training_started needed.
            # Early epochs: high uncertainty → use pessimistic1 (risk-averse)
            # Later epochs: uncertainty decreases → automatically switch to transition2 (reward-maximizing)

            # Reset per-epoch DPAS stats before each epoch
            MCTS.reset_epoch_dpas_stats()

            # PAPER-COMPLIANT: Use state_to_observation() instead of observe()
            # This ensures we NEVER leak traffic_level to the agent
            state_obs = env.state_to_observation(state)

            mcts = MCTS(
                state_obs,               # initial_state_coordinate (16D observation, NO traffic_level!)
                state,                   # initial_state_index (ParatransitState object, NOT int!)
                hipmdp,                  # bnn1
                hipmdp,                  # bnn2 (using same BNN model for paratransit)
                weight_set1,             # ws1: M̂k (current epoch's trained weights)
                weight_set2,             # ws2: M̂k-1 (previous epoch's trained weights)
                decision_epoch,          # time
                env,                     # task
                0.02,                    # threshold
                False,                   # training_started (kept False for frozen lake compatibility)
                False,                   # danger
                seed=seed                # Reproducible seed
            )

            # Run MCTS search from current state
            mcts.search(max_iterations)

            _log_and_store_dpas(mcts, decision_epoch, dpas_history)
            best_action = _select_best_action(mcts)

            # Store tree for this step
            step_key = f"ep_1|{decision_epoch}"

            # Serialize the tree
            if store_per_step_trees:
                # Use direct rollout_traces only (no subtree aggregation)
                per_step_trees[step_key] = serialize_node_for_json(
                    mcts.root,
                    max_depth=8
                )

            # Snapshot vehicle states BEFORE step to capture decision-time occupancy
            # We only need occupancy and route info, not full deep copy
            vehicle_snapshot = []
            for veh in state.vehicles:
                vehicle_snapshot.append({
                    'current_occupancy': getattr(veh, 'current_occupancy', 0),
                    'route': list(getattr(veh, 'route', [])),  # Shallow copy of route list
                    'current_location': getattr(veh, 'current_location', 0),
                    'capacity': getattr(veh, 'capacity', 3)
                })

            # Execute action in environment
            state_before_step = state  # Keep reference for request info
            state, reward, done, info = env.step(best_action)

            # Update completed requests tracking from env state (post-step)
            for v_id in range(n_vehicles):
                completed_by_vehicle[v_id] = set(
                    req_id for req_id, veh_id in env.request_assignments.items()
                    if veh_id == v_id and env.request_status.get(req_id) == "dropped-off"
                )

            # Record assignment with detailed information
            # Convert PassengerRequest to request_id for JSON serialization
            if hasattr(state_before_step, 'current_request'):
                request = state_before_step.current_request
                request_id = request.request_id if hasattr(request, 'request_id') else decision_epoch

                # Collect assignment details for informative display
                detail = {
                    'request_id': request_id,
                    'assigned_vehicle': best_action,
                    'pickup_node': getattr(request, 'pickup_node', None),
                    'dropoff_node': getattr(request, 'dropoff_node', None),
                    'request_time': getattr(request, 'request_time', None),
                    'earliest_pickup': getattr(request, 'earliest_pickup', None),
                    'latest_dropoff': getattr(request, 'latest_dropoff', None),
                }

                # Use vehicle snapshot (before step) for decision-time state
                if vehicle_snapshot:
                    vo = _build_vehicle_options(
                        vehicle_snapshot, request, env,
                        state_before_step.traffic_level, best_action, completed_by_vehicle)
                    detail.update({k: v for k, v in vo.items() if k != 'distance_by_vehicle'})

                assignment_details.append(detail)
            else:
                request_id = decision_epoch

            recommended_assignments.append((request_id, best_action))
            total_reward += reward
            decision_epoch += 1

            # Enhanced progress output with request details
            # Build info string from the detail dict we just created
            if assignment_details:
                last_detail = assignment_details[-1]
                earliest = last_detail.get('earliest_pickup')
                latest = last_detail.get('latest_dropoff')
                pickup_node = last_detail.get('pickup_node')
                dropoff_node = last_detail.get('dropoff_node')
                info_parts = [f"Request {request_id} -> Vehicle {best_action}"]
                if pickup_node is not None and dropoff_node is not None:
                    info_parts.append(f"({pickup_node}->{dropoff_node})")
                if earliest is not None and latest is not None:
                    info_parts.append(f"[window:{earliest:.0f}-{latest:.0f}]")
                # Add actual pickup/dropoff times from env
                actual_pickup = env.pickup_times.get(request_id)
                actual_dropoff = env.dropoff_times.get(request_id)
                if actual_pickup is not None or actual_dropoff is not None:
                    pickup_str = f"{actual_pickup:.0f}" if actual_pickup is not None else "?"
                    dropoff_str = f"{actual_dropoff:.0f}" if actual_dropoff is not None else "?"
                    info_parts.append(f"[actual:{pickup_str}/{dropoff_str}]")
                info_parts.append(f"reward:{reward:.2f}")
                print(f"  Step {decision_epoch}/{n_requests}: {' '.join(info_parts)}")
            else:
                print(f"  Step {decision_epoch}/{n_requests}: Request {request_id} -> Vehicle {best_action} (reward: {reward:.2f})")

        # Print summary
        print(f"\n[COMPLETE] Assigned all {n_requests} requests")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Average reward per request: {total_reward/n_requests:.2f}")
        print()

        # Collect environment data
        env_data = {
            'n_vehicles': n_vehicles,
            'n_requests': n_requests,
            'traffic_level': traffic_condition,
            'recommended_assignments': recommended_assignments,
            'assignment_details': assignment_details,
            'total_reward': total_reward,
            'final_state': str(state),
            'violations': getattr(env, 'violations', []),
            'seed': seed,
            'domain': 'paratransit',
            'dpas_history': dpas_history,
        }

        # Create planned_index_map for XAI query support
        # For paratransit: request number IS the decision epoch, so mapping is trivial
        # Map format: "ep_1|request_num" -> "ep_1|decision_epoch"
        planned_index_map = {}
        for req_id, _ in recommended_assignments:
            # req_id is the request number, which equals decision_epoch for that request
            map_key = f"ep_1|{req_id}"
            planned_index_map[map_key] = map_key  # Identity mapping for paratransit

        # Package all data
        result = {
            'env_data': env_data,
            'per_step_trees': per_step_trees,
            'planned_index_map': planned_index_map,
            'tree_data': {
                'recommended_assignments': recommended_assignments,
                'n_epochs': decision_epoch
            },
            'bnn_signature': bnn_signature,
            'reward': total_reward
        }

        return result

    except Exception as e:
        print(f"[ERROR] ADA-MCTS execution failed: {e}")
        traceback.print_exc()
        raise


# ============================================================================
# Model Version Tracking for Non-Stationarity Explanation
# ============================================================================

def rebuild_mcts_with_old_weights(
    state_snapshot,
    old_network_weights,
    old_latent_weights,
    baseline_network_weights,
    baseline_latent_weights,
    max_iterations=3000,
    verbose=True
):
    """
    Re-run MCTS from state_snapshot using old model weights to rebuild MDP_{t-n}.

    This creates an MCTS tree representing what the model at version t-n would have
    predicted for the CURRENT state (at epoch t), using:
    - BNN1 (M̂k) = old model weights at version t-n
    - BNN2 (M̂k-1) = baseline weights at version t-n (frozen at that time)

    This preserves the DPAS logic correctly, since DPAS uses the difference between
    M̂k and M̂k-1 to choose exploration branches.

    Args:
        state_snapshot: Saved state from CI detection containing:
            - 'state': ParatransitState object
            - 'state_obs': observation array (optional, will compute if missing)
            - 'epoch': decision epoch
            - 'env_params': dict with n_vehicles, n_requests, traffic_condition, seed
        old_network_weights: BNN network weights at version t-n (for M̂k)
        old_latent_weights: Latent codes at version t-n
        baseline_network_weights: Baseline BNN weights at version t-n (for M̂k-1)
        baseline_latent_weights: Baseline latent codes at version t-n
        max_iterations: MCTS iterations (default 3000)
        verbose: If True, print progress messages

    Returns:
        Serialized MCTS tree dict (same format as per_step_trees entries)
    """
    # 0) Extract state / observation / env_params
    epoch = state_snapshot.get('epoch', 0)
    state = state_snapshot.get('state')
    # Reconstruct ParatransitState from dict if needed (dict-only snapshot format)
    if isinstance(state, dict):
        state = _reconstruct_paratransit_state(state)
    state_obs = state_snapshot.get('state_obs')
    env_params = state_snapshot.get('env_params', {})

    if state is None:
        if verbose:
            print(f"[REBUILD] Error: No state in snapshot for epoch {epoch}")
        return None

    if not env_params:
        if verbose:
            print(f"[REBUILD] Error: No env_params in snapshot for epoch {epoch}")
        return None

    # Create a fresh NSParatransitV0 env using saved params
    env = NSParatransitV0(
        n_vehicles=env_params.get('n_vehicles', 5),
        n_requests=env_params.get('n_requests', 10),
        traffic_condition=env_params.get('traffic_condition', 0.3),
        seed=env_params.get('seed', 42),
        data_path=env_params.get('data_path'),
        use_real_data=True,
        fixed_request_ids=env_params.get('fixed_request_ids'),
        requests_csv_path=env_params.get('requests_csv_path'),
        max_time=env_params.get('max_time', 480)
    )

    # Apply event configuration for MDP_{t-n} rebuild (preferred for Case 2 & 3)
    # The rebuild should use the SAME event environment, just with OLD BNN weights
    # This allows comparing "old BNN understanding" vs "updated BNN understanding" in event conditions
    event_epoch = env_params.get('event_epoch')
    event_nodes = env_params.get('event_nodes')
    event_multiplier = env_params.get('event_multiplier', 3.0)
    if event_epoch is not None and event_nodes:
        env.set_event_config(
            event_epoch=event_epoch,
            event_nodes=event_nodes,
            event_multiplier=event_multiplier
        )
        # Set current_decision_epoch so event takes effect immediately
        env.current_decision_epoch = epoch
        if verbose:
            print(f"        Applied event config: epoch={event_epoch}, nodes={event_nodes}, mult={event_multiplier}x")

    # Apply bridge accident configuration for MDP_{t-n} rebuild (Case 1)
    bridge_accident_epoch = env_params.get('bridge_accident_epoch')
    bridge_accident_multiplier = env_params.get('bridge_accident_multiplier', 2.0)
    if bridge_accident_epoch is not None:
        env.set_bridge_accident_config(bridge_accident_epoch, bridge_accident_multiplier)
        env.current_decision_epoch = epoch
        if verbose:
            print(f"        Applied bridge accident config: epoch={bridge_accident_epoch}, mult={bridge_accident_multiplier}x")

    # Apply citywide congestion configuration for MDP_{t-n} rebuild (Case 3)
    congestion_epoch = env_params.get('congestion_epoch')
    congestion_traffic_level = env_params.get('congestion_traffic_level', 1.0)
    if congestion_epoch is not None:
        env.set_congestion_config(congestion_epoch, congestion_traffic_level)
        env.current_decision_epoch = epoch
        if verbose:
            print(f"        Applied congestion config: epoch={congestion_epoch}, level={congestion_traffic_level}")

    # Restore historical trajectory data for delay atomic props
    # Without this, pickup_delay/dropoff_delay would be underestimated in MDP_{t-n}
    env_history = state_snapshot.get('env_history', {})
    if env_history:
        env.pickup_times = env_history.get('pickup_times', {})
        env.dropoff_times = env_history.get('dropoff_times', {})
        # Restore additional state for full reproducibility
        if 'request_status' in env_history:
            env.request_status = env_history['request_status']
        if 'request_assignments' in env_history:
            env.request_assignments = env_history['request_assignments']
        # Note: request_ids are used for verification, requests list is already
        # initialized by NSParatransitV0 with same seed

    if state_obs is None:
        # Paper-compliant: use state_to_observation
        state_obs = env.state_to_observation(state)

    if verbose:
        print(f"[REBUILD] Rebuilding MDP_{{t-n}} for epoch {epoch}...")

    try:
        # 1) Reset cross-run caches / DPAS stats (avoid contamination)
        MCTS.clear_bnn_cache()
        MCTS.reset_epoch_dpas_stats()
        MCTS.previous_epistemic_uncertainty = None
        MCTS.previous_aleatoric_uncertainty = None

        # 2) Build HiPMDP models
        #    M̂k = "old" model from t-n (primary, evaluated model)
        hipmdp_old = HiPMDP(
            domain='paratransit',
            preset_hidden_params=[{'latent_code': 1}],
            run_type='full',
            bnn_hidden_layer_size=25,
            bnn_num_hidden_layers=3,
            bnn_network_weights=deepcopy(old_network_weights),
            paratransit_requests_csv_path=env_params.get('requests_csv_path')
        )
        hipmdp_old._HiPMDP__initialize_BNN()

        #    M̂k-1 = baseline/frozen model at that time (for DPAS comparison)
        hipmdp_base = HiPMDP(
            domain='paratransit',
            preset_hidden_params=[{'latent_code': 1}],
            run_type='full',
            bnn_hidden_layer_size=25,
            bnn_num_hidden_layers=3,
            bnn_network_weights=deepcopy(baseline_network_weights),
            paratransit_requests_csv_path=env_params.get('requests_csv_path')
        )
        hipmdp_base._HiPMDP__initialize_BNN()

        # 3) Initialize MCTS with dual models (paper: M̂k vs M̂k-1)
        #    BNN1 = M̂k (old), BNN2 = M̂k-1 (baseline)
        mcts = MCTS(
            state_obs,      # initial_state_coordinate
            state,          # initial_state_index
            hipmdp_old,     # bnn1 = M̂k (old)
            hipmdp_base,    # bnn2 = M̂k-1 (baseline)
            old_latent_weights.copy(),       # ws1
            baseline_latent_weights.copy(),  # ws2
            epoch,          # time
            env,            # task
            0.02,           # threshold
            False,          # training_started
            False,          # danger
            seed=env_params.get('seed', 42)  # Reproducible seed
        )

        # 4) Run MCTS search (DPAS inside expand/rollout)
        if verbose:
            print(f"        Running {max_iterations} MCTS iterations...")
        mcts.search(max_iterations)

        if verbose:
            print(f"[REBUILD] Complete.")

        # 5) Serialize tree for later PCTL checking (direct rollout_traces only)
        tree_dict = serialize_node_for_json(mcts.root, max_depth=8)

        return tree_dict

    except Exception as e:
        if verbose:
            print(f"[REBUILD] Error: {e}")
            traceback.print_exc()
        return None


# ============================================================================
# Common Initialization Helper
# ============================================================================

def _init_hipmdp_models(seed, requests_csv_path=None):
    """Initialize BNN weights and HiPMDP model pair (M̂k and M̂k-1).

    Returns:
        Tuple of (hipmdp1, hipmdp2, weight_set1, weight_set2,
                  latent_mean, latent_std, network_weights)
    """
    domain = 'paratransit'
    models_path = os.path.join(ada_mcts_path, 'models')

    with open(os.path.join(models_path, f'{domain}_network_weights_itr_2'), 'rb') as f:
        network_weights = pickle.load(f)
    with open(os.path.join(models_path, f'{domain}_latent_weights_itr_2'), 'rb') as f:
        latent_weights = pickle.load(f)

    bnn_hidden_layer_size = 25
    bnn_num_hidden_layers = 3
    preset_hidden_params = [{'latent_code': 1}]
    run_type = "full"

    # M̂k (hipmdp1): Current model, updated online during episode
    hipmdp1 = HiPMDP(
        domain, preset_hidden_params,
        run_type=run_type,
        bnn_hidden_layer_size=bnn_hidden_layer_size,
        bnn_num_hidden_layers=bnn_num_hidden_layers,
        bnn_network_weights=deepcopy(network_weights),
        paratransit_requests_csv_path=requests_csv_path
    )
    hipmdp1._HiPMDP__initialize_BNN()

    latent_mean = latent_weights.reshape(latent_weights.shape[1], )
    latent_std = np.ones_like(latent_mean) * 0.1
    np.random.seed(seed)
    weight_set1 = latent_mean + latent_std * np.random.randn(len(latent_mean))

    # M̂k-1 (hipmdp2): Previous episode model, FROZEN throughout episode
    hipmdp2 = HiPMDP(
        domain, preset_hidden_params,
        run_type=run_type,
        bnn_hidden_layer_size=bnn_hidden_layer_size,
        bnn_num_hidden_layers=bnn_num_hidden_layers,
        bnn_network_weights=deepcopy(network_weights),
        paratransit_requests_csv_path=requests_csv_path
    )
    hipmdp2._HiPMDP__initialize_BNN()
    weight_set2 = latent_mean.copy()

    # Paper Algorithm 1, Line 4: W_k ← W_{k-1}
    # Both models start with IDENTICAL posteriors (same network weights).
    # δE ≈ 0 at start → regular sampling (model trusts prior knowledge).
    # As M̂k trains on new data:
    #   - If environment changed: new data shifts M̂k's posterior → δE grows → pessimistic
    #   - If environment stable: posterior barely changes → δE stays small → regular
    # This is the paper's "Act as You Learn" mechanism.

    # CRITICAL FIX: Share frozen_randn between BNN1 and BNN2
    # Paper Eq. 9: VarE(M̂k; s,a) is a deterministic function of model parameters.
    # Both BNNs must use the SAME frozen noise vectors so that
    # δE = VarE(M̂k) - VarE(M̂k-1) reflects only parameter changes,
    # not sampling noise from different random vectors.
    # (Common random numbers — standard variance reduction technique)
    num_params = hipmdp1.network.num_weights
    shared_frozen_randn = npr.randn(hipmdp1.network.num_weight_samples, num_params)
    hipmdp1.network._frozen_randn = shared_frozen_randn
    hipmdp2.network._frozen_randn = shared_frozen_randn

    # DIAGNOSTIC: Verify VarE at initialization
    _diag_input = np.random.rand(1, 26)  # Random test input
    _, _, _ve1, _ = hipmdp1.network.feed_forward_distribution(_diag_input)
    _, _, _ve2, _ = hipmdp2.network.feed_forward_distribution(_diag_input)
    _de = np.mean(_ve1) - np.mean(_ve2)
    print(f"[INIT DIAG] VarE(M̂k)={np.mean(_ve1):.6f}, VarE(M̂k-1)={np.mean(_ve2):.6f}, δE={_de:.6f}")
    print(f"[INIT DIAG] BNN1 type={type(hipmdp1.network).__name__}, BNN2 type={type(hipmdp2.network).__name__}")
    print(f"[INIT DIAG] frozen_randn shared: {hipmdp1.network._frozen_randn is hipmdp2.network._frozen_randn}")
    print(f"[INIT DIAG] BNN1 v_prior={hipmdp1.network.v_prior}, BNN2 v_prior={hipmdp2.network.v_prior}")

    return hipmdp1, hipmdp2, weight_set1, weight_set2, latent_mean, latent_std, network_weights


def _train_model_from_buffers(
    hipmdp1, hipmdp2, weight_set1, weight_set2,
    episode_buffer,
    seed, domain, ada_mcts_path,
    best_network_error, best_latent_error, local_converge_count, Nu,
    decision_epoch, current_model_version, model_update_history,
    verbose=False,
):
    """Run BNN training and apply updated weights.  Returns mutated scalars.

    Encapsulates: prediction error measurement, buffer serialisation,
    train_model() call, weight application, BNN cache clear, error tracking,
    and model version history update.

    Returns
    -------
    dict with keys:
        weight_set1, best_network_error, best_latent_error,
        latent_mean, latent_std, current_model_version
    """
    # Prediction error BEFORE update
    recent_exp = episode_buffer[-1]
    s_obs_train, a_oh_train, _, s_next_obs_train, _ = recent_exp
    aug_input = np.hstack([s_obs_train, a_oh_train, weight_set1]).reshape((1, -1))
    pred_mean_before, _, _, _ = hipmdp1.network.feed_forward_distribution(aug_input)
    pre_update_error = float(np.mean((pred_mean_before[0] - s_next_obs_train) ** 2))

    # Save episode buffer to disk (train_model reads from data_buffer/)
    data_buffer_path = os.path.join(ada_mcts_path, 'data_buffer')
    os.makedirs(data_buffer_path, exist_ok=True)

    exp_list_Db = np.array(episode_buffer, dtype=object)
    buffer_file = os.path.join(data_buffer_path, f'{domain}_{seed}_exp_buffer_Db')
    with open(buffer_file, 'wb') as f:
        pickle.dump(exp_list_Db, f)

    # train_model uses relative paths — temporarily chdir
    original_cwd = os.getcwd()
    os.chdir(ada_mcts_path)
    try:
        updated_network_weights, updated_latent_weights, best_network_error, best_latent_error, latent_variance = train_model(
            seed, domain, hipmdp1, weight_set1,
            best_network_error, best_latent_error, local_converge_count,
            use_global_buffer=False,
            Nu=Nu,
        )
    finally:
        os.chdir(original_cwd)

    # Apply updated weights
    hipmdp1.network.weights = updated_network_weights
    weight_set1 = updated_latent_weights
    latent_mean = updated_latent_weights
    latent_std = np.sqrt(latent_variance + 1e-8)
    MCTS.clear_bnn_cache()

    # Prediction error AFTER update
    aug_input = np.hstack([s_obs_train, a_oh_train, weight_set1]).reshape((1, -1))
    pred_mean_after, _, _, _ = hipmdp1.network.feed_forward_distribution(aug_input)
    post_update_error = float(np.mean((pred_mean_after[0] - s_next_obs_train) ** 2))

    error_reduction = pre_update_error - post_update_error
    model_update_history.append({
        'epoch': decision_epoch,
        'pre_update_error': pre_update_error,
        'post_update_error': post_update_error,
        'error_reduction': error_reduction,
        'buffer_size': len(episode_buffer),
    })

    # Model version history update
    current_model_version += 1
    model_version_history_entry = {
        'start_epoch': decision_epoch + 1,
        'network_weights': deepcopy(hipmdp1.network.weights),
        'latent_weights': weight_set1.copy(),
        'baseline_network_weights': deepcopy(hipmdp2.network.weights),
        'baseline_latent_weights': weight_set2.copy(),
    }

    if verbose:
        print(f"        [TRAIN] Updated M̂k after {len(episode_buffer)} steps. "
              f"Error: {pre_update_error:.4f}→{post_update_error:.4f} (Δ={error_reduction:+.4f})")
        print(f"        [VERSION] Model version {current_model_version} active from epoch {decision_epoch + 1}")
        print(f"        [P_W] mean={latent_mean[:2]}, std={latent_std[:2]}")
    else:
        print(f"        [TRAIN] Updated M̂k. Error: {pre_update_error:.4f}→{post_update_error:.4f}")

    return {
        'weight_set1': weight_set1,
        'best_network_error': best_network_error,
        'best_latent_error': best_latent_error,
        'latent_mean': latent_mean,
        'latent_std': latent_std,
        'current_model_version': current_model_version,
        'version_history_entry': model_version_history_entry,
    }


def _log_and_store_dpas(mcts, decision_epoch, dpas_history, model_version=None):
    """Log DPAS stats and store history entry for the current epoch.

    Args:
        mcts: MCTS instance (after search).
        decision_epoch: Current epoch number.
        dpas_history: Dict to update in-place.
        model_version: Optional model version to include in the entry.
    """
    vw = mcts.compute_visit_weighted_dpas()
    dpas_stats = MCTS.get_epoch_dpas_stats()
    if dpas_stats:
        print(f"        [DPAS Epoch {decision_epoch}] regular={dpas_stats['regular_pct']:.1%} "
              f"pessimistic={dpas_stats['pessimistic_pct']:.1%} "
              f"EU={dpas_stats['epoch_eu']:.6f} AU={dpas_stats['epoch_au']:.6f}"
              + (f" invalid={dpas_stats['invalid_count']}" if dpas_stats.get('invalid_count', 0) > 0 else ""))
        print(f"          Root-W  avgδE={vw['root_weighted_dE']:.6f} avgδA={vw['root_weighted_dA']:.6f} (visits={vw['root_total_visits']})")
        print(f"          All-CW  avgδE={vw['all_weighted_dE']:.6f} avgδA={vw['all_weighted_dA']:.6f} (visits={vw['all_total_visits']})")
        entry = {
            'regular_pct': dpas_stats['regular_pct'],
            'pessimistic_pct': dpas_stats['pessimistic_pct'],
            'epoch_eu': dpas_stats['epoch_eu'],
            'epoch_au': dpas_stats['epoch_au'],
            'root_weighted_dE': vw['root_weighted_dE'],
            'root_weighted_dA': vw['root_weighted_dA'],
            'all_weighted_dE': vw['all_weighted_dE'],
            'all_weighted_dA': vw['all_weighted_dA'],
        }
        if model_version is not None:
            entry['model_version'] = model_version
        dpas_history[decision_epoch] = entry


def _select_best_action(mcts):
    """Return best action from MCTS, suppressing stdout from best_action()."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return mcts.best_action()
    finally:
        sys.stdout = old_stdout


def _build_vehicle_options(vehicle_snapshot, request, env, traffic_level,
                           best_action, completed_by_vehicle):
    """Compute per-vehicle distances and identify closest/chosen vehicle.

    Returns a dict with keys: closest_vehicle, closest_distance,
    vehicle_options, distance_by_vehicle, chosen_vehicle_pending,
    chosen_vehicle_completed, chosen_vehicle_distance.
    """
    min_dist = float('inf')
    closest_vehicle = None
    vehicle_options = []
    distance_by_vehicle = {}

    for v_id, veh_snap in enumerate(vehicle_snapshot):
        if hasattr(request, 'pickup_node'):
            try:
                dist = env.compute_travel_time(
                    veh_snap['current_location'], request.pickup_node,
                    traffic_level, deterministic=True)
                distance_by_vehicle[v_id] = float(dist)
                vehicle_options.append({
                    'vehicle_id': v_id,
                    'distance': float(dist),
                    'current_occupancy': veh_snap['current_occupancy'],
                    'pending_requests': len(set(req_id for req_id, _ in veh_snap['route'])),
                    'completed_requests': len(completed_by_vehicle[v_id]),
                    'capacity': veh_snap['capacity'],
                })
                if dist < min_dist:
                    min_dist = dist
                    closest_vehicle = v_id
            except:
                pass

    result = {
        'closest_vehicle': closest_vehicle,
        'closest_distance': float(min_dist) if min_dist != float('inf') else None,
        'vehicle_options': vehicle_options,
        'distance_by_vehicle': distance_by_vehicle,
    }

    if best_action < len(vehicle_snapshot):
        chosen_snap = vehicle_snapshot[best_action]
        result['chosen_vehicle_pending'] = len(set(req_id for req_id, _ in chosen_snap['route']))
        result['chosen_vehicle_completed'] = len(completed_by_vehicle[best_action])
        result['chosen_vehicle_distance'] = distance_by_vehicle.get(best_action)

    return result


# ============================================================================
# Counter-Intuitive Vehicle Assignment Scenario (Scenario 1)
# ============================================================================

def run_counter_intuitive_scenario(n_vehicles=5, n_requests=10, max_iterations=3000, seed=42,
                                    N_min=3, N_interval=2, fixed_request_ids=None,
                                    requests_csv_path=None, data_path=None,
                                    bridge_accident_epoch=None, bridge_accident_multiplier=2.0,
                                    Nu=3):
    """
    Run ADA-MCTS for counter-intuitive vehicle assignment scenario.

    This scenario demonstrates non-stationarity by:
    1. Running MCTS at each decision epoch
    2. Detecting counter-intuitive assignments (closest != chosen)
    3. For such assignments, saving state snapshot for lazy MDP_{t-n} reconstruction

    PAPER-COMPLIANT Implementation (Algorithm 1):
    - M̂k (hipmdp1): Current model, updated during episode via train_model()
    - M̂k-1 (hipmdp2): Previous EPISODE's model, FROZEN throughout this episode
    - D_b: Episode buffer for online adaptation
    - D: Global buffer for cross-episode learning (not used in single-episode demo)
    - N_min: Minimum buffer size before first training
    - N_interval: Update frequency in steps
    - P_W: Latent weight distribution (mean, std) updated during training

    δE and δA Calculation (DPAS - Algorithm 2):
    - δE = VarE(M̂k; s,a) - VarE(M̂k-1; s,a) for same (s,a)
    - δA = VarA(M̂k; S') - VarA(M̂k-1; S') for same action subset S'
    - Both computed using SAME state with BOTH models

    Args:
        n_vehicles: Number of vehicles (default 5)
        n_requests: Number of requests (default 10)
        max_iterations: MCTS iterations per epoch (default 3000)
        seed: Random seed
        N_min: Minimum buffer size before first update (Paper hyperparameter)
        N_interval: Update frequency in steps (Paper hyperparameter)
        fixed_request_ids: If provided, use these specific request IDs for reproducibility
        requests_csv_path: If provided, load requests from this CSV file
        Nu: Number of TuneModel training iterations per update (Paper Algorithm 1)

    Returns:
        Dict with scenario data including MDP comparison for counter-intuitive cases
    """
    try:
        domain = 'paratransit'

        hipmdp1, hipmdp2, weight_set1, weight_set2, latent_mean, latent_std, network_weights = \
            _init_hipmdp_models(seed, requests_csv_path)

        # Create environment
        traffic_condition = 0.3
        env = NSParatransitV0(
            n_vehicles=n_vehicles,
            n_requests=n_requests,
            traffic_condition=traffic_condition,
            seed=seed,
            data_path=data_path,
            use_real_data=True,
            fixed_request_ids=fixed_request_ids,
            requests_csv_path=requests_csv_path,
            max_time=480
        )

        # Apply bridge accident if configured (Case 1: S→N cross-river congestion)
        if bridge_accident_epoch is not None:
            env.set_bridge_accident_config(bridge_accident_epoch, bridge_accident_multiplier)
            print(f"        Applied bridge accident config: epoch={bridge_accident_epoch}, "
                  f"S→N multiplier={bridge_accident_multiplier}x, "
                  f"south_nodes={len(env.south_nodes)}, north_nodes={len(env.north_nodes)}")

        # IMPORTANT: Sync n_requests with env's actual value
        # (env.n_requests = len(fixed_request_ids) if provided, else n_requests param)
        n_requests = env.n_requests

        # BNN signature for hashing
        bnn_signature = f"bnn_{hashlib.md5(str(network_weights).encode()).hexdigest()[:8]}"

        # Storage
        per_step_trees = {}
        recommended_assignments = []
        assignment_details = []
        completed_by_vehicle = [set() for _ in range(n_vehicles)]

        # MDP comparison storage (for counter-intuitive cases)
        mdp_comparisons = {}
        counter_intuitive_epochs = []

        # ========================================================================
        # Model Version Tracking for t-n comparison
        # ========================================================================
        # Each version records the model state at the time of update
        # Version 0 = initial state, Version N = after N-th train_model() call
        current_model_version = 0
        model_version_history = {
            0: {
                'start_epoch': 0,  # Version 0 active from epoch 0
                'network_weights': deepcopy(hipmdp1.network.weights),
                'latent_weights': weight_set1.copy(),
                # Baseline at version 0 is same as initial (hipmdp2)
                'baseline_network_weights': deepcopy(hipmdp2.network.weights),
                'baseline_latent_weights': weight_set2.copy(),
            }
        }

        # ========================================================================
        # Runtime implementation: keep only the episode buffer D_b for online
        # adaptation. This paratransit runner does not maintain a separate
        # cross-episode global replay buffer D.
        # ========================================================================
        episode_buffer = []  # D_b - stores (s, a, r, s', w_b)

        # PAPER Algorithm 1, Line 5: step_idx for explicit step counting
        step_idx = 0  # Episode-level step index (reset at episode boundary)

        # Track model updates for learning trend
        model_update_history = []

        # DPAS history for TURNING_INTERVAL analysis
        dpas_history = {}  # {epoch: {regular_pct, pessimistic_pct, avg_delta_E, avg_delta_A}}

        # Training state
        best_network_error = 100.0
        best_latent_error = 100.0
        local_converge_count = 0

        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0.0
        decision_epoch = 0

        print(f"\n[SCENARIO] Counter-Intuitive Vehicle Assignment (PAPER Algorithm 1)")
        print(f"  Vehicles: {n_vehicles}")
        print(f"  Requests: {n_requests}")
        print(f"  Iterations: {max_iterations}")
        print(f"  N_min={N_min}, N_interval={N_interval}")
        print(f"  M̂k-1 frozen for entire episode (per paper)")
        if bridge_accident_epoch is not None:
            print(f"  Episode boundary at epoch {bridge_accident_epoch} (bridge accident, M̂k-1 will be refreshed)")
        print()

        # Episode boundary epoch (paper: M_{k-1} -> M_k transition)
        episode_boundary_epoch = bridge_accident_epoch
        episode_boundary_applied = False
        current_episode = 1  # Track which episode we're in

        # Reset DPAS state
        MCTS.previous_epistemic_uncertainty = None
        MCTS.previous_aleatoric_uncertainty = None

        while not done and decision_epoch < n_requests:
            # ================================================================
            # PAPER Algorithm 1: Episode boundary reset
            # When environment changes (M_{k-1} -> M_k), start new episode:
            #   - M̂k-1 := snapshot of current M̂k (learned dynamics of old env)
            #   - Clear episode buffer D_b
            #   - Sample new w_b from updated P_W
            #   - Reset training state
            # ENV and vehicle state continue naturally (no env reset)
            # ================================================================
            if (episode_boundary_epoch is not None and
                    decision_epoch == episode_boundary_epoch and
                    not episode_boundary_applied):
                print(f"\n{'='*60}")
                print(f"  [EPISODE BOUNDARY] Epoch {decision_epoch}: Environment change detected")
                print(f"  Episode 1 -> Episode 2 (Paper: M_{{k-1}} -> M_k)")
                print(f"  M̂k-1 refreshed: now contains Episode 1's learned model")
                print(f"  Episode buffer D_b cleared, new w_b sampled from P_W")

                # 1. M̂k-1 := current M̂k snapshot (Algorithm 1, conceptual)
                #    hipmdp2 gets current hipmdp1's learned weights (FULL, including tight variance)
                old_w_b = weight_set1.copy()
                hipmdp2.network.weights = deepcopy(hipmdp1.network.weights)
                weight_set2 = weight_set1.copy()

                # Paper Algorithm 1, Line 4: W_k ← W_{k-1}
                # Both models now share the same posterior. δE ≈ 0 at boundary start.
                # As M̂k trains on post-change data, δE will grow if dynamics changed.

                # 2. Clear episode buffer D_b (Algorithm 1, Line 3)
                episode_buffer.clear()

                # Reset step_idx for new episode (Issue 5: explicit step counter)
                step_idx = 0

                # 3. Sample new w_b from P_W (Algorithm 1, Line 2)
                # Paper: "w_b ~ P_W, which could be a standard Gaussian distribution"
                # Practical floor for demo sensitivity: avoid near-zero latent std collapse.
                ep2_seed = seed + 10000 + decision_epoch
                np.random.seed(ep2_seed)
                effective_std = np.maximum(latent_std, 0.05)
                weight_set1 = latent_mean + effective_std * np.random.randn(len(latent_mean))
                print(f"  P_W std: {latent_std} -> effective: {effective_std}")
                print(f"  Old w_b: {old_w_b} → New w_b: {weight_set1}")
                print(f"  w_b shift: {np.linalg.norm(weight_set1 - old_w_b):.6f}")
                print(f"{'='*60}\n")

                # 4. Reset training state for new episode
                best_network_error = 100.0
                best_latent_error = 100.0
                local_converge_count = 0

                # 5. Reset DPAS state for new episode
                MCTS.previous_epistemic_uncertainty = None
                MCTS.previous_aleatoric_uncertainty = None

                # 6. Record new model version for the episode boundary
                current_model_version += 1
                model_version_history[current_model_version] = {
                    'start_epoch': decision_epoch,
                    'network_weights': deepcopy(hipmdp1.network.weights),
                    'latent_weights': weight_set1.copy(),
                    'baseline_network_weights': deepcopy(hipmdp2.network.weights),
                    'baseline_latent_weights': weight_set2.copy(),
                    'is_episode_boundary': True,
                }

                # 7. Clear BNN cache for fresh predictions
                MCTS.clear_bnn_cache()

                episode_boundary_applied = True
                current_episode = 2

            state_obs = env.state_to_observation(state)

            # REPRODUCIBILITY: Reset ALL random seeds before each epoch
            # This ensures MCTS rollouts and trace sampling are deterministic
            epoch_seed = seed + decision_epoch
            random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            # Also seed autograd.numpy.random (used by BNN internals)
            if HAS_AUTOGRAD:
                npr.seed(epoch_seed)
            # Also seed torch if using GPU BNN
            if HAS_TORCH:
                torch.manual_seed(epoch_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(epoch_seed)

            # Reset per-epoch DPAS stats before each epoch
            MCTS.reset_epoch_dpas_stats()

            # ================================================================
            # PAPER Algorithm 2 (DPAS): Compute δE and δA using SAME state
            # δE = VarE(M̂k; s,a) - VarE(M̂k-1; s,a)
            # δA = VarA(M̂k; S') - VarA(M̂k-1; S')
            # ================================================================

            # Create and run MCTS for current epoch
            # PAPER-COMPLIANT: Use hipmdp1 (M̂k) and hipmdp2 (M̂k-1)
            # M̂k-1 is FROZEN for entire episode (not updated per-epoch)
            mcts = MCTS(
                state_obs, state, hipmdp1, hipmdp2,
                weight_set1, weight_set2,
                decision_epoch, env, 0.02, False, False,
                seed=epoch_seed  # Use epoch-specific seed for reproducibility
            )
            mcts.search(max_iterations)

            _log_and_store_dpas(mcts, decision_epoch, dpas_history, model_version=current_model_version)
            best_action = _select_best_action(mcts)

            # NOTE: Tree/evidence is saved for ALL epochs (enables querying any epoch)
            step_key = f"ep_1|{decision_epoch}"

            # Snapshot vehicle states BEFORE step
            vehicle_snapshot = []
            for veh in state.vehicles:
                vehicle_snapshot.append({
                    'current_occupancy': getattr(veh, 'current_occupancy', 0),
                    'route': list(getattr(veh, 'route', [])),
                    'current_location': getattr(veh, 'current_location', 0),
                    'capacity': getattr(veh, 'capacity', 3)
                })

            # Execute action
            state_before_step = state
            state, reward, done, info = env.step(best_action)

            # Update completed requests tracking from env state (post-step)
            for v_id in range(n_vehicles):
                completed_by_vehicle[v_id] = set(
                    req_id for req_id, veh_id in env.request_assignments.items()
                    if veh_id == v_id and env.request_status.get(req_id) == "dropped-off"
                )

            # PAPER-COMPLIANT FIX (Issue 1): Collect experience for online BNN update
            # Buffer format: (state_obs, action_one_hot, reward, next_state_obs, w_b)
            state_obs_before = env.state_to_observation(state_before_step)
            state_obs_after = env.state_to_observation(state)
            action_one_hot = np.zeros(n_vehicles)
            action_one_hot[best_action] = 1
            transition = [
                state_obs_before,      # s
                action_one_hot,        # a (one-hot)
                reward,                # r
                state_obs_after,       # s'
                weight_set1.copy()     # w_b (current latent weights)
            ]
            # Store the transition in the episode buffer D_b.
            # This runner's online updates train from D_b only.
            episode_buffer.append(transition)

            # Increment step_idx (Issue 5: explicit step counter)
            step_idx += 1

            # Build assignment detail
            if hasattr(state_before_step, 'current_request'):
                request = state_before_step.current_request
                request_id = request.request_id if hasattr(request, 'request_id') else decision_epoch

                detail = {
                    'request_id': request_id,
                    'assigned_vehicle': best_action,
                    'pickup_node': getattr(request, 'pickup_node', None),
                    'dropoff_node': getattr(request, 'dropoff_node', None),
                    'request_time': getattr(request, 'request_time', None),
                    'earliest_pickup': getattr(request, 'earliest_pickup', None),
                    'latest_dropoff': getattr(request, 'latest_dropoff', None),
                }

                vo = _build_vehicle_options(
                    vehicle_snapshot, request, env,
                    state_before_step.traffic_level, best_action, completed_by_vehicle)
                closest_vehicle = vo['closest_vehicle']
                detail.update({k: v for k, v in vo.items() if k != 'distance_by_vehicle'})

                assignment_details.append(detail)

                # Save tree for ALL epochs (needed for querying any epoch)
                per_step_trees[step_key] = serialize_node_for_json(mcts.root, max_depth=8)

                # Save state snapshot for ALL epochs (needed for MDP comparison at any epoch)
                # This enables complete derived/comparison analysis for all epochs
                # Compute prev_version_id at snapshot time (avoids scan at query time)
                _prev_vid = None
                for _v in range(current_model_version - 1, -1, -1):
                    if _v in model_version_history:
                        _prev_vid = _v
                        break
                if _prev_vid is None and 0 in model_version_history:
                    _prev_vid = 0

                state_snapshot = {
                    'state_obs': env.state_to_observation(state_before_step),
                    'state': _serialize_paratransit_state(state_before_step),
                    'epoch': decision_epoch,
                    'model_version_at_t': current_model_version,
                    'prev_version_id': _prev_vid,
                    'env_params': {
                        'n_vehicles': n_vehicles,
                        'n_requests': n_requests,
                        'traffic_condition': traffic_condition,
                        'seed': seed,
                        'data_path': data_path,
                        'fixed_request_ids': fixed_request_ids,
                        'requests_csv_path': requests_csv_path,
                        'bridge_accident_epoch': bridge_accident_epoch,
                        'bridge_accident_multiplier': bridge_accident_multiplier,
                        'max_time': env.max_time,
                    },
                    # Store historical trajectory data for delay atomic props
                    # Without this, pickup_delay/dropoff_delay would be underestimated
                    'env_history': {
                        'pickup_times': dict(getattr(env, 'pickup_times', {})),
                        'dropoff_times': dict(getattr(env, 'dropoff_times', {})),
                        # Additional state for full reproducibility
                        'request_status': dict(getattr(env, 'request_status', {})),
                        'request_assignments': dict(getattr(env, 'request_assignments', {})),
                        # Store original CSV indices for reproducibility (not reassigned IDs)
                        'original_request_ids': [r.original_id for r in getattr(env, 'requests', [])],
                    },
                }

                # Extract PCTL metrics for assigned and closest vehicles (for all epochs)
                mdp_t_metrics = {
                    'assigned_vehicle': extract_pctl_metrics_for_vehicle(mcts, best_action, env, decision_epoch),
                }

                # Always extract closest_vehicle metrics (for complete derived analysis)
                if closest_vehicle is not None:
                    mdp_t_metrics['closest_vehicle'] = extract_pctl_metrics_for_vehicle(mcts, closest_vehicle, env, decision_epoch)

                # Create assignment_info for ALL epochs (enables all CLOSEST_*/ETA_* derived)
                assignment_info = {
                    'assigned_vehicle': best_action,
                    'closest_vehicle': closest_vehicle,
                    'assigned_distance': detail.get('chosen_vehicle_distance'),
                    'closest_distance': detail.get('closest_distance'),
                    'assigned_pending': detail.get('chosen_vehicle_pending'),
                    'closest_pending': None,  # Will be populated if needed
                }
                # Get closest vehicle pending requests if available
                if closest_vehicle is not None and closest_vehicle < len(vehicle_snapshot):
                    closest_snap = vehicle_snapshot[closest_vehicle]
                    assignment_info['closest_pending'] = len(set(req_id for req_id, _ in closest_snap['route']))

                mdp_t_metrics['assignment_info'] = assignment_info

                # Check for counter-intuitive assignment
                ci_info = detect_counter_intuitive_assignment(detail, mcts, env, decision_epoch)
                if ci_info:
                    counter_intuitive_epochs.append(decision_epoch)
                    print(f"  [CI] Epoch {decision_epoch}: Vehicle {best_action} chosen over closer Vehicle {closest_vehicle}")

                    # Store with CI-specific info
                    mdp_comparisons[decision_epoch] = {
                        'mdp_t': mdp_t_metrics,
                        'mdp_t_minus_n': None,  # Will be rebuilt on-demand using model_version_history
                        'counter_intuitive_info': ci_info,
                        'state_snapshot': state_snapshot,  # For lazy rebuild

                    }
                else:
                    # For non-CI epochs, still store complete info (for full queryability)
                    mdp_comparisons[decision_epoch] = {
                        'mdp_t': mdp_t_metrics,
                        'mdp_t_minus_n': None,  # Will be rebuilt on-demand using model_version_history
                        'state_snapshot': state_snapshot,  # For lazy rebuild

                    }
            else:
                request_id = decision_epoch

            recommended_assignments.append((request_id, best_action))
            total_reward += reward

            # Enhanced progress output with request details
            if assignment_details:
                last_detail = assignment_details[-1]
                earliest = last_detail.get('earliest_pickup')
                latest = last_detail.get('latest_dropoff')
                pickup_node = last_detail.get('pickup_node')
                dropoff_node = last_detail.get('dropoff_node')
                info_parts = [f"Request {request_id} -> Vehicle {best_action}"]
                if pickup_node is not None and dropoff_node is not None:
                    info_parts.append(f"({pickup_node}->{dropoff_node})")
                if earliest is not None and latest is not None:
                    info_parts.append(f"[window:{earliest:.0f}-{latest:.0f}]")
                # Add actual pickup/dropoff times from env
                actual_pickup = env.pickup_times.get(request_id)
                actual_dropoff = env.dropoff_times.get(request_id)
                if actual_pickup is not None or actual_dropoff is not None:
                    pickup_str = f"{actual_pickup:.0f}" if actual_pickup is not None else "?"
                    dropoff_str = f"{actual_dropoff:.0f}" if actual_dropoff is not None else "?"
                    info_parts.append(f"[actual:{pickup_str}/{dropoff_str}]")
                info_parts.append(f"reward:{reward:.2f}")
                print(f"  Epoch {decision_epoch}: {' '.join(info_parts)}")
            else:
                print(f"  Epoch {decision_epoch}: Request {request_id} -> Vehicle {best_action} (reward: {reward:.2f})")

            # ================================================================
            # PAPER Algorithm 1, Line 10: Update model periodically
            # Trigger: step_idx mod N_interval == 0 AND |D_b| >= N_threshold
            # Issue 5: Use explicit step_idx instead of len(D_b)
            # ================================================================
            should_update = (step_idx % N_interval == 0 and
                             len(episode_buffer) >= N_min)

            if should_update:
                try:
                    train_result = _train_model_from_buffers(
                        hipmdp1, hipmdp2, weight_set1, weight_set2,
                        episode_buffer,
                        seed, domain, ada_mcts_path,
                        best_network_error, best_latent_error, local_converge_count, Nu,
                        decision_epoch, current_model_version, model_update_history,
                        verbose=True,
                    )
                    weight_set1 = train_result['weight_set1']
                    best_network_error = train_result['best_network_error']
                    best_latent_error = train_result['best_latent_error']
                    latent_mean = train_result['latent_mean']
                    latent_std = train_result['latent_std']
                    current_model_version = train_result['current_model_version']
                    model_version_history[current_model_version] = train_result['version_history_entry']

                    # Incremental prune: drop versions no longer referenced by any snapshot
                    _needed = {0, current_model_version}
                    for _comp in mdp_comparisons.values():
                        _ss = _comp.get('state_snapshot', {})
                        _needed.add(_ss.get('model_version_at_t', 0))
                        _pv = _ss.get('prev_version_id')
                        if _pv is not None:
                            _needed.add(_pv)
                    _to_drop = [v for v in model_version_history if v not in _needed]
                    for v in _to_drop:
                        del model_version_history[v]

                except Exception as e:
                    if DEBUG_MODE:
                        print(f"        [WARNING] train_model failed: {e}")
                        import traceback
                        traceback.print_exc()

            decision_epoch += 1

        print(f"\n[COMPLETE] Total reward: {total_reward:.2f}")
        if counter_intuitive_epochs:
            print(f"[CI] Counter-intuitive epochs: {counter_intuitive_epochs}")
        else:
            print(f"[CI] No counter-intuitive assignments detected")

        # PAPER-COMPLIANT FIX (Issue 4): Print learning trend summary
        if model_update_history:
            avg_error = np.mean([h['post_update_error'] for h in model_update_history])
            print(f"[Learning] Average prediction error: {avg_error:.6f}")
            print(f"[Learning] Total experience collected: D_b={len(episode_buffer)} transitions")

        # Package results
        env_data = {
            'n_vehicles': n_vehicles,
            'n_requests': n_requests,
            'traffic_level': traffic_condition,
            'recommended_assignments': recommended_assignments,
            'assignment_details': assignment_details,
            'total_reward': total_reward,
            'final_state': str(state),
            'violations': getattr(env, 'violations', []),
            'seed': seed,
            'domain': 'paratransit',
            'scenario': 'counter_intuitive_assignment',
        }

        planned_index_map = {f"ep_1|{req_id}": f"ep_1|{req_id}" for req_id, _ in recommended_assignments}

        # Prune model_version_history to only keep versions referenced by snapshots
        _needed_versions = {0}  # always keep initial
        for _comp in mdp_comparisons.values():
            _ss = _comp.get('state_snapshot', {})
            _needed_versions.add(_ss.get('model_version_at_t', 0))
            _pv = _ss.get('prev_version_id')
            if _pv is not None:
                _needed_versions.add(_pv)
        model_version_history = {v: model_version_history[v]
                                 for v in _needed_versions if v in model_version_history}

        result = {
            'env_data': env_data,
            'per_step_trees': per_step_trees,
            'planned_index_map': planned_index_map,
            'tree_data': {
                'recommended_assignments': recommended_assignments,
                'n_epochs': decision_epoch
            },
            'bnn_signature': bnn_signature,
            'reward': total_reward,
            'scenario_data': {
                'type': 'counter_intuitive_assignment',
                'counter_intuitive_epochs': counter_intuitive_epochs,
                'mdp_comparisons': mdp_comparisons,
                # Model version history for t-n comparison (lazy rebuild)
                'model_version_history': model_version_history,
                'final_model_version': current_model_version,
                # Learning trend data
                'model_update_history': model_update_history,
                'total_experience_collected_Db': len(episode_buffer),
                # DPAS history for TURNING_INTERVAL analysis
                'dpas_history': dpas_history,
                # Bridge accident config (Case 1: S→N cross-river congestion)
                'bridge_accident_epoch': bridge_accident_epoch,
                'bridge_accident_multiplier': bridge_accident_multiplier,
                # Episode boundary info (paper: M_{k-1} -> M_k)
                'episode_boundary_epoch': episode_boundary_epoch,
                'num_episodes': current_episode,
            }
        }

        return result

    except Exception as e:
        print(f"[ERROR] Counter-intuitive scenario failed: {e}")
        traceback.print_exc()
        raise


# ============================================================================
# Event Scenario (Scenario 2 & 3)
# ============================================================================

def run_event_scenario(n_vehicles=5, n_requests=30, max_iterations=3000, seed=42,
                       N_min=3, N_interval=2, fixed_request_ids=None,
                       event_epoch=10, event_nodes=None, event_multiplier=3.0,
                       expect_assignment_change=True, scenario_mode="change",
                       requests_csv_path=None, data_path=None, Nu=3):
    """
    Run ADA-MCTS for event-based non-stationarity scenario (Case 2).

    This scenario demonstrates non-stationarity caused by a sudden event.
    Some epochs may show assignment changes (old BNN vs updated BNN chose
    differently), while others show PCTL shifts with unchanged assignment.

    EVENT MODE: node-based congestion
    - Any route where pickup OR dropoff is in event_nodes will have travel time multiplied
    - Realistic for scenarios like stadium events, road closures at intersections

    Args:
        n_vehicles: Number of vehicles (default 5)
        n_requests: Number of requests (default 10)
        max_iterations: MCTS iterations per epoch (default 3000)
        seed: Random seed
        N_min: Minimum buffer size before first update
        N_interval: Update frequency in steps
        fixed_request_ids: If provided, use these specific request IDs for reproducibility
        Nu: Number of TuneModel training iterations (Paper Algorithm 1, configurable)
        event_epoch: Epoch at which the event occurs (default 10)
        event_nodes: List of affected node IDs, e.g., [5, 12, 3].
                    Any route with from_node OR to_node in this set will be affected.
        event_multiplier: Travel time multiplier for event-affected routes (default 3x)
        expect_assignment_change: Whether assignment change is expected (default True).
        scenario_mode: Scenario mode identifier (default "change")
        requests_csv_path: If provided, load requests from this CSV file instead of
                          train_chains.csv. When set, fixed_request_ids is ignored.

    Returns:
        Dict with scenario data including MDP comparison at event epoch
    """
    try:
        domain = 'paratransit'

        hipmdp1, hipmdp2, weight_set1, weight_set2, latent_mean, latent_std, network_weights = \
            _init_hipmdp_models(seed, requests_csv_path)

        # Create environment
        traffic_condition = 0.3
        env = NSParatransitV0(
            n_vehicles=n_vehicles,
            n_requests=n_requests,
            traffic_condition=traffic_condition,
            seed=seed,
            data_path=data_path,
            use_real_data=True,
            fixed_request_ids=fixed_request_ids,
            requests_csv_path=requests_csv_path,
            max_time=480
        )

        n_requests = env.n_requests

        # EVENT MODE: node-based congestion
        # Convert event_nodes to list if it's a set
        if event_nodes is None:
            event_nodes = []
        event_nodes_list = list(event_nodes) if isinstance(event_nodes, set) else list(event_nodes)
        
        # Configure event in environment
        env.set_event_config(
            event_epoch=event_epoch,
            event_nodes=event_nodes_list,
            event_multiplier=event_multiplier
        )
        
        # Use event_epoch as the trigger epoch
        trigger_epoch = event_epoch
        
        scenario_name = "Event-Based Non-Stationarity"
        print(f"\n[SCENARIO] {scenario_name}")
        print(f"  Event occurs at epoch {event_epoch}")
        print(f"  Affected nodes: {event_nodes_list}")
        print(f"  Travel time multiplier: {event_multiplier}x")
        print(f"  (Any route with pickup/dropoff at these nodes is affected)")

        # BNN signature
        bnn_signature = f"bnn_{hashlib.md5(str(network_weights).encode()).hexdigest()[:8]}"

        # Storage
        per_step_trees = {}
        recommended_assignments = []
        assignment_details = []
        completed_by_vehicle = [set() for _ in range(n_vehicles)]

        # MDP comparison storage
        mdp_comparisons = {}

        # Model version tracking
        current_model_version = 0
        model_version_history = {
            0: {
                'start_epoch': 0,
                'network_weights': deepcopy(hipmdp1.network.weights),
                'latent_weights': weight_set1.copy(),
                'baseline_network_weights': deepcopy(hipmdp2.network.weights),
                'baseline_latent_weights': weight_set2.copy(),
            }
        }

        # Episode buffer
        episode_buffer = []
        step_idx = 0  # Episode-level step index (Issue 5)
        model_update_history = []

        # DPAS history for TURNING_INTERVAL analysis
        dpas_history = {}  # {epoch: {regular_pct, pessimistic_pct, avg_delta_E, avg_delta_A}}

        best_network_error = 100.0
        best_latent_error = 100.0
        local_converge_count = 0

        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0.0
        decision_epoch = 0

        print(f"\n[SCENARIO] Running with {n_vehicles} vehicles, {n_requests} requests")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Episode boundary at epoch {event_epoch} (M̂k-1 will be refreshed)")
        print()

        # Episode boundary epoch = event epoch (paper: M_{k-1} -> M_k transition)
        episode_boundary_epoch = event_epoch
        episode_boundary_applied = False
        current_episode = 1

        MCTS.previous_epistemic_uncertainty = None
        MCTS.previous_aleatoric_uncertainty = None

        while not done and decision_epoch < n_requests:
            # ================================================================
            # PAPER Algorithm 1: Episode boundary reset
            # When environment changes (M_{k-1} -> M_k), start new episode:
            #   - M̂k-1 := snapshot of current M̂k (learned dynamics of old env)
            #   - Clear episode buffer D_b
            #   - Sample new w_b from updated P_W
            #   - Reset training state
            # ENV and vehicle state continue naturally (no env reset)
            # ================================================================
            if (episode_boundary_epoch is not None and
                    decision_epoch == episode_boundary_epoch and
                    not episode_boundary_applied):
                print(f"\n{'='*60}")
                print(f"  [EPISODE BOUNDARY] Epoch {decision_epoch}: Event detected")
                print(f"  Episode 1 -> Episode 2 (Paper: M_{{k-1}} -> M_k)")
                print(f"  M̂k-1 refreshed: now contains Episode 1's learned model")
                print(f"  Episode buffer D_b cleared, new w_b sampled from P_W")

                # 1. M̂k-1 := current M̂k snapshot
                old_w_b = weight_set1.copy()
                hipmdp2.network.weights = deepcopy(hipmdp1.network.weights)
                weight_set2 = weight_set1.copy()

                # Paper Algorithm 1, Line 4: Both models share identical posterior at boundary.

                # 2. Clear episode buffer D_b
                episode_buffer.clear()
                step_idx = 0  # Reset step index for new episode (Issue 5)

                # 3. Sample new w_b from P_W
                # Practical floor for demo sensitivity: avoid near-zero latent std collapse.
                ep2_seed = seed + 10000 + decision_epoch
                np.random.seed(ep2_seed)
                effective_std = np.maximum(latent_std, 0.05)
                weight_set1 = latent_mean + effective_std * np.random.randn(len(latent_mean))
                print(f"  P_W std: {latent_std} -> effective: {effective_std}")
                print(f"  Old w_b: {old_w_b} → New w_b: {weight_set1}")
                print(f"  w_b shift: {np.linalg.norm(weight_set1 - old_w_b):.6f}")
                print(f"{'='*60}\n")

                # 4. Reset training state
                best_network_error = 100.0
                best_latent_error = 100.0
                local_converge_count = 0

                # 5. Reset DPAS state
                MCTS.previous_epistemic_uncertainty = None
                MCTS.previous_aleatoric_uncertainty = None

                # 6. Record episode boundary in model version history
                current_model_version += 1
                model_version_history[current_model_version] = {
                    'start_epoch': decision_epoch,
                    'network_weights': deepcopy(hipmdp1.network.weights),
                    'latent_weights': weight_set1.copy(),
                    'baseline_network_weights': deepcopy(hipmdp2.network.weights),
                    'baseline_latent_weights': weight_set2.copy(),
                    'is_episode_boundary': True,
                }

                # 7. Clear BNN cache
                MCTS.clear_bnn_cache()

                episode_boundary_applied = True
                current_episode = 2

            state_obs = env.state_to_observation(state)

            # Set seeds for reproducibility
            epoch_seed = seed + decision_epoch
            random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            if HAS_AUTOGRAD:
                npr.seed(epoch_seed)
            if HAS_TORCH:
                torch.manual_seed(epoch_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(epoch_seed)

            MCTS.reset_epoch_dpas_stats()

            # Create and run MCTS
            mcts = MCTS(
                state_obs, state, hipmdp1, hipmdp2,
                weight_set1, weight_set2,
                decision_epoch, env, 0.02, False, False,
                seed=epoch_seed
            )
            mcts.search(max_iterations)

            _log_and_store_dpas(mcts, decision_epoch, dpas_history, model_version=current_model_version)
            best_action = _select_best_action(mcts)

            step_key = f"ep_1|{decision_epoch}"

            # Snapshot vehicle states BEFORE step
            vehicle_snapshot = []
            for veh in state.vehicles:
                vehicle_snapshot.append({
                    'current_occupancy': getattr(veh, 'current_occupancy', 0),
                    'route': list(getattr(veh, 'route', [])),
                    'current_location': getattr(veh, 'current_location', 0),
                    'capacity': getattr(veh, 'capacity', 3)
                })

            # Execute action
            state_before_step = state
            state, reward, done, info = env.step(best_action)

            # Update completed requests tracking from env state (post-step)
            for v_id in range(n_vehicles):
                completed_by_vehicle[v_id] = set(
                    req_id for req_id, veh_id in env.request_assignments.items()
                    if veh_id == v_id and env.request_status.get(req_id) == "dropped-off"
                )

            # Collect experience
            state_obs_before = env.state_to_observation(state_before_step)
            state_obs_after = env.state_to_observation(state)
            action_one_hot = np.zeros(n_vehicles)
            action_one_hot[best_action] = 1
            transition = [
                state_obs_before,
                action_one_hot,
                reward,
                state_obs_after,
                weight_set1.copy()
            ]
            # Store the transition in the episode buffer D_b.
            episode_buffer.append(transition)

            # Increment step_idx (Issue 5)
            step_idx += 1

            # Build assignment detail
            if hasattr(state_before_step, 'current_request'):
                request = state_before_step.current_request
                request_id = request.request_id if hasattr(request, 'request_id') else decision_epoch

                detail = {
                    'request_id': request_id,
                    'decision_epoch': decision_epoch,
                    'assigned_vehicle': best_action,
                    'pickup_node': getattr(request, 'pickup_node', None),
                    'dropoff_node': getattr(request, 'dropoff_node', None),
                    'request_time': getattr(request, 'request_time', None),
                    'earliest_pickup': getattr(request, 'earliest_pickup', None),
                    'latest_dropoff': getattr(request, 'latest_dropoff', None),
                    'is_event_epoch': decision_epoch >= trigger_epoch,  # Mark all event-active epochs
                    'traffic_level': getattr(state_before_step, 'traffic_level', None),
                }

                vo = _build_vehicle_options(
                    vehicle_snapshot, request, env,
                    state_before_step.traffic_level, best_action, completed_by_vehicle)
                closest_vehicle = vo['closest_vehicle']
                detail.update({k: v for k, v in vo.items() if k != 'distance_by_vehicle'})

                assignment_details.append(detail)

                # Save tree for ALL epochs (needed for querying any epoch)
                per_step_trees[step_key] = serialize_node_for_json(mcts.root, max_depth=8)

                # Save state snapshot for ALL epochs (needed for MDP comparison at any epoch)
                # This enables complete derived/comparison analysis for all epochs
                # Compute prev_version_id at snapshot time
                _prev_vid = None
                for _v in range(current_model_version - 1, -1, -1):
                    if _v in model_version_history:
                        _prev_vid = _v
                        break
                if _prev_vid is None and 0 in model_version_history:
                    _prev_vid = 0

                state_snapshot = {
                    'state_obs': env.state_to_observation(state_before_step),
                    'state': _serialize_paratransit_state(state_before_step),
                    'epoch': decision_epoch,
                    'model_version_at_t': current_model_version,
                    'prev_version_id': _prev_vid,
                    'env_params': {
                        'n_vehicles': n_vehicles,
                        'n_requests': n_requests,
                        'traffic_condition': traffic_condition,
                        'seed': seed,
                        'data_path': data_path,
                        'fixed_request_ids': fixed_request_ids,
                        'requests_csv_path': requests_csv_path,
                        'event_epoch': event_epoch,
                        'event_nodes': event_nodes_list,
                        'event_multiplier': event_multiplier,
                        'max_time': env.max_time,
                    },
                    'env_history': {
                        'pickup_times': dict(getattr(env, 'pickup_times', {})),
                        'dropoff_times': dict(getattr(env, 'dropoff_times', {})),
                        'request_status': dict(getattr(env, 'request_status', {})),
                        'request_assignments': dict(getattr(env, 'request_assignments', {})),
                        'original_request_ids': [r.original_id for r in getattr(env, 'requests', [])],
                    },
                }

                # Extract PCTL metrics for assigned and closest vehicles (all epochs)
                mdp_t_metrics = {
                    'assigned_vehicle': extract_pctl_metrics_for_vehicle(mcts, best_action, env, decision_epoch),
                }

                # Always extract closest_vehicle metrics separately (for complete derived analysis)
                if closest_vehicle is not None:
                    mdp_t_metrics['closest_vehicle'] = extract_pctl_metrics_for_vehicle(mcts, closest_vehicle, env, decision_epoch)

                # Create assignment_info for ALL epochs (enables all CLOSEST_*/ETA_* derived)
                assignment_info = {
                    'assigned_vehicle': best_action,
                    'closest_vehicle': closest_vehicle,
                    'assigned_distance': detail.get('chosen_vehicle_distance'),
                    'closest_distance': detail.get('closest_distance'),
                    'assigned_pending': detail.get('chosen_vehicle_pending'),
                    'closest_pending': None,
                }
                # Get closest vehicle pending requests if available
                if closest_vehicle is not None and closest_vehicle < len(vehicle_snapshot):
                    closest_snap = vehicle_snapshot[closest_vehicle]
                    assignment_info['closest_pending'] = len(set(req_id for req_id, _ in closest_snap['route']))

                mdp_t_metrics['assignment_info'] = assignment_info

                # Check if event is active at this epoch
                is_trigger_active = (decision_epoch >= trigger_epoch)

                if is_trigger_active:
                    # First time at trigger epoch: announce it
                    if decision_epoch == trigger_epoch:
                        print(f"  [EVENT] Epoch {decision_epoch}: Event triggered at nodes {event_nodes_list}")

                    # Build event_info for event-active epochs
                    event_info = {
                        # Event-specific fields
                        'event_epoch': event_epoch,
                        'event_nodes': event_nodes_list,
                        'event_multiplier': event_multiplier,
                        # Common fields
                        'trigger_epoch': trigger_epoch,
                        'mdp_t_assignment': best_action,  # What updated BNN chose at this epoch
                        'mdp_t_minus_n_assignment': None,  # Placeholder - filled during MDP_{t-n} rebuild
                        'assignment_changed': None,  # Placeholder - computed after rebuild
                        'current_epoch': decision_epoch,
                        'model_version': current_model_version,
                        # Assignment details for derived analysis
                        'closest_vehicle': closest_vehicle,
                        'assigned_distance': detail.get('chosen_vehicle_distance'),
                        'closest_distance': detail.get('closest_distance'),
                    }

                    # Store with event info
                    mdp_comparisons[decision_epoch] = {
                        'mdp_t': mdp_t_metrics,
                        'mdp_t_minus_n': None,  # Will be rebuilt on demand
                        'state_snapshot': state_snapshot,
                        'event_info': event_info,

                    }
                else:
                    # For non-event epochs, still store complete info (for full queryability)
                    mdp_comparisons[decision_epoch] = {
                        'mdp_t': mdp_t_metrics,
                        'mdp_t_minus_n': None,  # Will be rebuilt on demand
                        'state_snapshot': state_snapshot,

                    }
            else:
                request_id = decision_epoch

            recommended_assignments.append((request_id, best_action))
            total_reward += reward

            # Enhanced progress output with request details
            if assignment_details:
                last_detail = assignment_details[-1]
                earliest = last_detail.get('earliest_pickup')
                latest = last_detail.get('latest_dropoff')
                pickup_node = last_detail.get('pickup_node')
                dropoff_node = last_detail.get('dropoff_node')
                info_parts = [f"Request {request_id} -> Vehicle {best_action}"]
                if pickup_node is not None and dropoff_node is not None:
                    info_parts.append(f"({pickup_node}->{dropoff_node})")
                if earliest is not None and latest is not None:
                    info_parts.append(f"[window:{earliest:.0f}-{latest:.0f}]")
                # Add actual pickup/dropoff times from env
                actual_pickup = env.pickup_times.get(request_id)
                actual_dropoff = env.dropoff_times.get(request_id)
                if actual_pickup is not None or actual_dropoff is not None:
                    pickup_str = f"{actual_pickup:.0f}" if actual_pickup is not None else "?"
                    dropoff_str = f"{actual_dropoff:.0f}" if actual_dropoff is not None else "?"
                    info_parts.append(f"[actual:{pickup_str}/{dropoff_str}]")
                info_parts.append(f"reward:{reward:.2f}")
                print(f"  Epoch {decision_epoch}: {' '.join(info_parts)}")
            else:
                print(f"  Epoch {decision_epoch}: Request {request_id} -> Vehicle {best_action} (reward: {reward:.2f})")

            # PAPER Algorithm 1, Line 10: Update model periodically
            # Trigger: step_idx mod N_interval == 0 AND |D_b| >= N_threshold (Issue 5)
            should_update = (step_idx % N_interval == 0 and
                             len(episode_buffer) >= N_min)

            if should_update:
                try:
                    train_result = _train_model_from_buffers(
                        hipmdp1, hipmdp2, weight_set1, weight_set2,
                        episode_buffer,
                        seed, domain, ada_mcts_path,
                        best_network_error, best_latent_error, local_converge_count, Nu,
                        decision_epoch, current_model_version, model_update_history,
                        verbose=False,
                    )
                    weight_set1 = train_result['weight_set1']
                    best_network_error = train_result['best_network_error']
                    best_latent_error = train_result['best_latent_error']
                    latent_mean = train_result['latent_mean']
                    latent_std = train_result['latent_std']
                    current_model_version = train_result['current_model_version']
                    model_version_history[current_model_version] = train_result['version_history_entry']

                    # Incremental prune: drop versions no longer referenced by any snapshot
                    _needed = {0, current_model_version}
                    for _comp in mdp_comparisons.values():
                        _ss = _comp.get('state_snapshot', {})
                        _needed.add(_ss.get('model_version_at_t', 0))
                        _pv = _ss.get('prev_version_id')
                        if _pv is not None:
                            _needed.add(_pv)
                    _to_drop = [v for v in model_version_history if v not in _needed]
                    for v in _to_drop:
                        del model_version_history[v]

                except Exception as e:
                    if DEBUG_MODE:
                        print(f"        [WARNING] train_model failed: {e}")

            decision_epoch += 1

        print(f"\n[COMPLETE] Total reward: {total_reward:.2f}")

        # Get traffic level history from env
        traffic_level_history = getattr(env, 'traffic_level_history', {})

        # Determine scenario name
        scenario_name = 'event_assignment_change'

        env_data = {
            'n_vehicles': n_vehicles,
            'n_requests': n_requests,
            'traffic_level': traffic_condition,
            'recommended_assignments': recommended_assignments,
            'assignment_details': assignment_details,
            'total_reward': total_reward,
            'final_state': str(state),
            'violations': getattr(env, 'violations', []),
            'seed': seed,
            'domain': 'paratransit',
            'scenario': scenario_name,
            'traffic_level_history': traffic_level_history,
        }

        planned_index_map = {f"ep_1|{req_id}": f"ep_1|{req_id}" for req_id, _ in recommended_assignments}

        # Prune model_version_history to only keep versions referenced by snapshots
        _needed_versions = {0}
        for _comp in mdp_comparisons.values():
            _ss = _comp.get('state_snapshot', {})
            _needed_versions.add(_ss.get('model_version_at_t', 0))
            _pv = _ss.get('prev_version_id')
            if _pv is not None:
                _needed_versions.add(_pv)
        model_version_history = {v: model_version_history[v]
                                 for v in _needed_versions if v in model_version_history}

        scenario_type = scenario_name

        result = {
            'env_data': env_data,
            'per_step_trees': per_step_trees,
            'planned_index_map': planned_index_map,
            'tree_data': {
                'recommended_assignments': recommended_assignments,
                'n_epochs': decision_epoch
            },
            'bnn_signature': bnn_signature,
            'reward': total_reward,
            'scenario_data': {
                'type': scenario_type,
                'mdp_comparisons': mdp_comparisons,
                'model_version_history': model_version_history,
                'final_model_version': current_model_version,
                'model_update_history': model_update_history,
                'total_experience_collected_Db': len(episode_buffer),
                'event_config': {
                    'event_epoch': event_epoch,
                    'event_nodes': event_nodes_list,
                    'event_multiplier': event_multiplier,
                },
                # Combined trigger info
                'trigger_epoch': trigger_epoch,
                # Scenario expectation flags
                'expect_assignment_change': expect_assignment_change,
                'scenario_mode': scenario_mode,
                # Placeholder for true comparison result (filled after MDP rebuild):
                'assignment_changed': None,  # True result: old BNN vs updated BNN at same epoch
                'mdp_t_minus_n_assignment': None,  # What old BNN would choose at comparison epoch
                'mdp_t_assignment': None,  # What updated BNN chose at comparison epoch  
                'comparison_epoch': None,  # Which epoch was actually compared
                # DPAS history for TURNING_INTERVAL analysis
                'dpas_history': dpas_history,
                # Episode boundary info (paper: M_{k-1} -> M_k)
                'episode_boundary_epoch': episode_boundary_epoch,
                'num_episodes': current_episode,
            }
        }

        return result

    except Exception as e:
        print(f"[ERROR] Event scenario failed: {e}")
        traceback.print_exc()
        raise


def run_congestion_scenario(n_vehicles=5, n_requests=30, max_iterations=3000, seed=42,
                            N_min=3, N_interval=2, fixed_request_ids=None,
                            congestion_epoch=10, congestion_traffic_level=1.0,
                            requests_csv_path=None, data_path=None, Nu=3):
    """
    Run ADA-MCTS for citywide congestion scenario (Case 3).

    This scenario demonstrates non-stationarity caused by a global traffic_level
    jump at congestion_epoch. Unlike Case 2 (node-based event multiplier), this
    changes the interpolation coefficient affecting ALL OD pairs.

    Args:
        n_vehicles: Number of vehicles (default 5)
        n_requests: Number of requests (default 30)
        max_iterations: MCTS iterations per epoch (default 3000)
        seed: Random seed
        N_min: Minimum buffer size before first update
        N_interval: Update frequency in steps
        fixed_request_ids: If provided, use these specific request IDs
        Nu: Number of TuneModel training iterations
        congestion_epoch: Epoch at which citywide congestion begins (default 10)
        congestion_traffic_level: Traffic level after congestion onset (default 1.0)
        requests_csv_path: Path to requests CSV file
        data_path: Path to data directory

    Returns:
        Dict with scenario data including MDP comparison at congestion epochs
    """
    try:
        domain = 'paratransit'

        hipmdp1, hipmdp2, weight_set1, weight_set2, latent_mean, latent_std, network_weights = \
            _init_hipmdp_models(seed, requests_csv_path)

        # Create environment with low base traffic (near free-flow)
        traffic_condition = 0.1
        env = NSParatransitV0(
            n_vehicles=n_vehicles,
            n_requests=n_requests,
            traffic_condition=traffic_condition,
            seed=seed,
            data_path=data_path,
            use_real_data=True,
            fixed_request_ids=fixed_request_ids,
            requests_csv_path=requests_csv_path,
            max_time=480
        )

        n_requests = env.n_requests

        # Configure citywide congestion in environment
        env.set_congestion_config(
            congestion_epoch=congestion_epoch,
            congestion_traffic_level=congestion_traffic_level
        )

        trigger_epoch = congestion_epoch

        scenario_name = "Citywide Congestion"
        print(f"\n[SCENARIO] {scenario_name}")
        print(f"  Congestion begins at epoch {congestion_epoch}")
        print(f"  Traffic level: {traffic_condition} -> {congestion_traffic_level}")
        print(f"  (All routes affected through global travel time interpolation)")

        # BNN signature
        bnn_signature = f"bnn_{hashlib.md5(str(network_weights).encode()).hexdigest()[:8]}"

        # Storage
        per_step_trees = {}
        recommended_assignments = []
        assignment_details = []
        completed_by_vehicle = [set() for _ in range(n_vehicles)]

        # MDP comparison storage
        mdp_comparisons = {}

        # Model version tracking
        current_model_version = 0
        model_version_history = {
            0: {
                'start_epoch': 0,
                'network_weights': deepcopy(hipmdp1.network.weights),
                'latent_weights': weight_set1.copy(),
                'baseline_network_weights': deepcopy(hipmdp2.network.weights),
                'baseline_latent_weights': weight_set2.copy(),
            }
        }

        # Episode buffer
        episode_buffer = []
        step_idx = 0
        model_update_history = []

        # DPAS history
        dpas_history = {}

        best_network_error = 100.0
        best_latent_error = 100.0
        local_converge_count = 0

        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0.0
        decision_epoch = 0

        print(f"\n[SCENARIO] Running with {n_vehicles} vehicles, {n_requests} requests")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Episode boundary at epoch {congestion_epoch} (M̂k-1 will be refreshed)")
        print()

        # Episode boundary epoch = congestion epoch
        episode_boundary_epoch = congestion_epoch
        episode_boundary_applied = False
        current_episode = 1

        MCTS.previous_epistemic_uncertainty = None
        MCTS.previous_aleatoric_uncertainty = None

        while not done and decision_epoch < n_requests:
            # ================================================================
            # PAPER Algorithm 1: Episode boundary reset
            # ================================================================
            if (episode_boundary_epoch is not None and
                    decision_epoch == episode_boundary_epoch and
                    not episode_boundary_applied):
                print(f"\n{'='*60}")
                print(f"  [EPISODE BOUNDARY] Epoch {decision_epoch}: Citywide congestion detected")
                print(f"  Episode 1 -> Episode 2 (Paper: M_{{k-1}} -> M_k)")
                print(f"  Traffic level: {traffic_condition} -> {congestion_traffic_level}")
                print(f"  M̂k-1 refreshed: now contains Episode 1's learned model")
                print(f"  Episode buffer D_b cleared, new w_b sampled from P_W")

                # 1. M̂k-1 := current M̂k snapshot
                old_w_b = weight_set1.copy()
                hipmdp2.network.weights = deepcopy(hipmdp1.network.weights)
                weight_set2 = weight_set1.copy()

                # 2. Clear episode buffer D_b
                episode_buffer.clear()
                step_idx = 0

                # 3. Sample new w_b from P_W
                ep2_seed = seed + 10000 + decision_epoch
                np.random.seed(ep2_seed)
                effective_std = np.maximum(latent_std, 0.05)
                weight_set1 = latent_mean + effective_std * np.random.randn(len(latent_mean))
                print(f"  P_W std: {latent_std} -> effective: {effective_std}")
                print(f"  Old w_b: {old_w_b} → New w_b: {weight_set1}")
                print(f"  w_b shift: {np.linalg.norm(weight_set1 - old_w_b):.6f}")
                print(f"{'='*60}\n")

                # 4. Reset training state
                best_network_error = 100.0
                best_latent_error = 100.0
                local_converge_count = 0

                # 5. Reset DPAS state
                MCTS.previous_epistemic_uncertainty = None
                MCTS.previous_aleatoric_uncertainty = None

                # 6. Record episode boundary in model version history
                current_model_version += 1
                model_version_history[current_model_version] = {
                    'start_epoch': decision_epoch,
                    'network_weights': deepcopy(hipmdp1.network.weights),
                    'latent_weights': weight_set1.copy(),
                    'baseline_network_weights': deepcopy(hipmdp2.network.weights),
                    'baseline_latent_weights': weight_set2.copy(),
                    'is_episode_boundary': True,
                }

                # 7. Clear BNN cache
                MCTS.clear_bnn_cache()

                episode_boundary_applied = True
                current_episode = 2

            state_obs = env.state_to_observation(state)

            # Set seeds for reproducibility
            epoch_seed = seed + decision_epoch
            random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            if HAS_AUTOGRAD:
                npr.seed(epoch_seed)
            if HAS_TORCH:
                torch.manual_seed(epoch_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(epoch_seed)

            MCTS.reset_epoch_dpas_stats()

            # Create and run MCTS
            mcts = MCTS(
                state_obs, state, hipmdp1, hipmdp2,
                weight_set1, weight_set2,
                decision_epoch, env, 0.02, False, False,
                seed=epoch_seed
            )
            mcts.search(max_iterations)

            _log_and_store_dpas(mcts, decision_epoch, dpas_history, model_version=current_model_version)
            best_action = _select_best_action(mcts)

            step_key = f"ep_1|{decision_epoch}"

            # Snapshot vehicle states BEFORE step
            vehicle_snapshot = []
            for veh in state.vehicles:
                vehicle_snapshot.append({
                    'current_occupancy': getattr(veh, 'current_occupancy', 0),
                    'route': list(getattr(veh, 'route', [])),
                    'current_location': getattr(veh, 'current_location', 0),
                    'capacity': getattr(veh, 'capacity', 3)
                })

            # Execute action
            state_before_step = state
            state, reward, done, info = env.step(best_action)

            # Update completed requests tracking
            for v_id in range(n_vehicles):
                completed_by_vehicle[v_id] = set(
                    req_id for req_id, veh_id in env.request_assignments.items()
                    if veh_id == v_id and env.request_status.get(req_id) == "dropped-off"
                )

            # Collect experience
            state_obs_before = env.state_to_observation(state_before_step)
            state_obs_after = env.state_to_observation(state)
            action_one_hot = np.zeros(n_vehicles)
            action_one_hot[best_action] = 1
            transition = [
                state_obs_before,
                action_one_hot,
                reward,
                state_obs_after,
                weight_set1.copy()
            ]
            episode_buffer.append(transition)
            step_idx += 1

            # Build assignment detail
            if hasattr(state_before_step, 'current_request'):
                request = state_before_step.current_request
                request_id = request.request_id if hasattr(request, 'request_id') else decision_epoch

                detail = {
                    'request_id': request_id,
                    'decision_epoch': decision_epoch,
                    'assigned_vehicle': best_action,
                    'pickup_node': getattr(request, 'pickup_node', None),
                    'dropoff_node': getattr(request, 'dropoff_node', None),
                    'request_time': getattr(request, 'request_time', None),
                    'earliest_pickup': getattr(request, 'earliest_pickup', None),
                    'latest_dropoff': getattr(request, 'latest_dropoff', None),
                    'is_congestion_epoch': decision_epoch >= trigger_epoch,
                    'traffic_level': getattr(state_before_step, 'traffic_level', None),
                }

                vo = _build_vehicle_options(
                    vehicle_snapshot, request, env,
                    state_before_step.traffic_level, best_action, completed_by_vehicle)
                closest_vehicle = vo['closest_vehicle']
                detail.update({k: v for k, v in vo.items() if k != 'distance_by_vehicle'})

                assignment_details.append(detail)

                # Save tree for ALL epochs
                per_step_trees[step_key] = serialize_node_for_json(mcts.root, max_depth=8)

                # Save state snapshot for ALL epochs
                _prev_vid = None
                for _v in range(current_model_version - 1, -1, -1):
                    if _v in model_version_history:
                        _prev_vid = _v
                        break
                if _prev_vid is None and 0 in model_version_history:
                    _prev_vid = 0

                state_snapshot = {
                    'state_obs': env.state_to_observation(state_before_step),
                    'state': _serialize_paratransit_state(state_before_step),
                    'epoch': decision_epoch,
                    'model_version_at_t': current_model_version,
                    'prev_version_id': _prev_vid,
                    'env_params': {
                        'n_vehicles': n_vehicles,
                        'n_requests': n_requests,
                        'traffic_condition': traffic_condition,
                        'seed': seed,
                        'data_path': data_path,
                        'fixed_request_ids': fixed_request_ids,
                        'requests_csv_path': requests_csv_path,
                        'congestion_epoch': congestion_epoch,
                        'congestion_traffic_level': congestion_traffic_level,
                        'max_time': env.max_time,
                    },
                    'env_history': {
                        'pickup_times': dict(getattr(env, 'pickup_times', {})),
                        'dropoff_times': dict(getattr(env, 'dropoff_times', {})),
                        'request_status': dict(getattr(env, 'request_status', {})),
                        'request_assignments': dict(getattr(env, 'request_assignments', {})),
                        'original_request_ids': [r.original_id for r in getattr(env, 'requests', [])],
                    },
                }

                # Extract PCTL metrics for assigned and closest vehicles
                mdp_t_metrics = {
                    'assigned_vehicle': extract_pctl_metrics_for_vehicle(mcts, best_action, env, decision_epoch),
                }
                if closest_vehicle is not None:
                    mdp_t_metrics['closest_vehicle'] = extract_pctl_metrics_for_vehicle(mcts, closest_vehicle, env, decision_epoch)

                assignment_info = {
                    'assigned_vehicle': best_action,
                    'closest_vehicle': closest_vehicle,
                    'assigned_distance': detail.get('chosen_vehicle_distance'),
                    'closest_distance': detail.get('closest_distance'),
                    'assigned_pending': detail.get('chosen_vehicle_pending'),
                    'closest_pending': None,
                }
                if closest_vehicle is not None and closest_vehicle < len(vehicle_snapshot):
                    closest_snap = vehicle_snapshot[closest_vehicle]
                    assignment_info['closest_pending'] = len(set(req_id for req_id, _ in closest_snap['route']))

                mdp_t_metrics['assignment_info'] = assignment_info

                # Check if congestion is active
                is_trigger_active = (decision_epoch >= trigger_epoch)

                if is_trigger_active:
                    if decision_epoch == trigger_epoch:
                        print(f"  [CONGESTION] Epoch {decision_epoch}: Citywide congestion active (traffic_level -> {congestion_traffic_level})")

                    # Build event_info (reuses Case 2 structure for enrichment chain compatibility)
                    event_info = {
                        'trigger_epoch': trigger_epoch,
                        'mdp_t_assignment': best_action,
                        'mdp_t_minus_n_assignment': None,
                        'assignment_changed': None,
                        'current_epoch': decision_epoch,
                        'model_version': current_model_version,
                        'closest_vehicle': closest_vehicle,
                        'assigned_distance': detail.get('chosen_vehicle_distance'),
                        'closest_distance': detail.get('closest_distance'),
                        # Case-3-specific fields
                        'congestion_epoch': congestion_epoch,
                        'base_traffic_level': traffic_condition,
                        'congestion_traffic_level': congestion_traffic_level,
                    }

                    mdp_comparisons[decision_epoch] = {
                        'mdp_t': mdp_t_metrics,
                        'mdp_t_minus_n': None,
                        'state_snapshot': state_snapshot,
                        'event_info': event_info,
                    }
                else:
                    mdp_comparisons[decision_epoch] = {
                        'mdp_t': mdp_t_metrics,
                        'mdp_t_minus_n': None,
                        'state_snapshot': state_snapshot,
                    }
            else:
                request_id = decision_epoch

            recommended_assignments.append((request_id, best_action))
            total_reward += reward

            # Progress output
            if assignment_details:
                last_detail = assignment_details[-1]
                earliest = last_detail.get('earliest_pickup')
                latest = last_detail.get('latest_dropoff')
                pickup_node = last_detail.get('pickup_node')
                dropoff_node = last_detail.get('dropoff_node')
                tl = last_detail.get('traffic_level', '?')
                info_parts = [f"Request {request_id} -> Vehicle {best_action}"]
                if pickup_node is not None and dropoff_node is not None:
                    info_parts.append(f"({pickup_node}->{dropoff_node})")
                if earliest is not None and latest is not None:
                    info_parts.append(f"[window:{earliest:.0f}-{latest:.0f}]")
                actual_pickup = env.pickup_times.get(request_id)
                actual_dropoff = env.dropoff_times.get(request_id)
                if actual_pickup is not None or actual_dropoff is not None:
                    pickup_str = f"{actual_pickup:.0f}" if actual_pickup is not None else "?"
                    dropoff_str = f"{actual_dropoff:.0f}" if actual_dropoff is not None else "?"
                    info_parts.append(f"[actual:{pickup_str}/{dropoff_str}]")
                info_parts.append(f"tl:{tl}")
                info_parts.append(f"reward:{reward:.2f}")
                print(f"  Epoch {decision_epoch}: {' '.join(info_parts)}")
            else:
                print(f"  Epoch {decision_epoch}: Request {request_id} -> Vehicle {best_action} (reward: {reward:.2f})")

            # PAPER Algorithm 1, Line 10: Update model periodically
            should_update = (step_idx % N_interval == 0 and
                             len(episode_buffer) >= N_min)

            if should_update:
                try:
                    train_result = _train_model_from_buffers(
                        hipmdp1, hipmdp2, weight_set1, weight_set2,
                        episode_buffer,
                        seed, domain, ada_mcts_path,
                        best_network_error, best_latent_error, local_converge_count, Nu,
                        decision_epoch, current_model_version, model_update_history,
                        verbose=False,
                    )
                    weight_set1 = train_result['weight_set1']
                    best_network_error = train_result['best_network_error']
                    best_latent_error = train_result['best_latent_error']
                    latent_mean = train_result['latent_mean']
                    latent_std = train_result['latent_std']
                    current_model_version = train_result['current_model_version']
                    model_version_history[current_model_version] = train_result['version_history_entry']

                    # Incremental prune
                    _needed = {0, current_model_version}
                    for _comp in mdp_comparisons.values():
                        _ss = _comp.get('state_snapshot', {})
                        _needed.add(_ss.get('model_version_at_t', 0))
                        _pv = _ss.get('prev_version_id')
                        if _pv is not None:
                            _needed.add(_pv)
                    _to_drop = [v for v in model_version_history if v not in _needed]
                    for v in _to_drop:
                        del model_version_history[v]

                except Exception as e:
                    if DEBUG_MODE:
                        print(f"        [WARNING] train_model failed: {e}")

            decision_epoch += 1

        print(f"\n[COMPLETE] Total reward: {total_reward:.2f}")

        # Get traffic level history from env
        traffic_level_history = getattr(env, 'traffic_level_history', {})

        scenario_name = 'citywide_congestion'

        env_data = {
            'n_vehicles': n_vehicles,
            'n_requests': n_requests,
            'traffic_level': traffic_condition,
            'recommended_assignments': recommended_assignments,
            'assignment_details': assignment_details,
            'total_reward': total_reward,
            'final_state': str(state),
            'violations': getattr(env, 'violations', []),
            'seed': seed,
            'domain': 'paratransit',
            'scenario': scenario_name,
            'traffic_level_history': traffic_level_history,
        }

        planned_index_map = {f"ep_1|{req_id}": f"ep_1|{req_id}" for req_id, _ in recommended_assignments}

        # Prune model_version_history
        _needed_versions = {0}
        for _comp in mdp_comparisons.values():
            _ss = _comp.get('state_snapshot', {})
            _needed_versions.add(_ss.get('model_version_at_t', 0))
            _pv = _ss.get('prev_version_id')
            if _pv is not None:
                _needed_versions.add(_pv)
        model_version_history = {v: model_version_history[v]
                                 for v in _needed_versions if v in model_version_history}

        result = {
            'env_data': env_data,
            'per_step_trees': per_step_trees,
            'planned_index_map': planned_index_map,
            'tree_data': {
                'recommended_assignments': recommended_assignments,
                'n_epochs': decision_epoch
            },
            'bnn_signature': bnn_signature,
            'reward': total_reward,
            'scenario_data': {
                'type': scenario_name,
                'mdp_comparisons': mdp_comparisons,
                'model_version_history': model_version_history,
                'final_model_version': current_model_version,
                'model_update_history': model_update_history,
                'total_experience_collected_Db': len(episode_buffer),
                'congestion_config': {
                    'congestion_epoch': congestion_epoch,
                    'congestion_traffic_level': congestion_traffic_level,
                    'base_traffic_level': traffic_condition,
                },
                'trigger_epoch': trigger_epoch,
                # Placeholder for comparison result (filled after MDP rebuild)
                'assignment_changed': None,
                'mdp_t_minus_n_assignment': None,
                'mdp_t_assignment': None,
                'comparison_epoch': None,
                # DPAS history
                'dpas_history': dpas_history,
                # Episode boundary info
                'episode_boundary_epoch': episode_boundary_epoch,
                'num_episodes': current_episode,
            }
        }

        return result

    except Exception as e:
        print(f"[ERROR] Congestion scenario failed: {e}")
        traceback.print_exc()
        raise


def run_scenario0_scenario(n_vehicles=5, n_requests=30, max_iterations=3000, seed=42,
                           N_min=3, N_interval=2, fixed_request_ids=None,
                           congestion_epoch=10, congestion_traffic_level=1.0,
                           requests_csv_path=None, data_path=None, Nu=3,
                           probe_configs=None):
    """
    Run ADA-MCTS for Scenario 0 — fixed-snapshot probe during real congestion adaptation.

    The underlying real run is a Case-3 congestion adaptation run (identical to
    run_congestion_scenario). In addition, at every real decision epoch t, after the
    real transition has been appended to episode_buffer and BEFORE the training
    trigger fires, we evaluate each candidate probe snapshot:
      - fixed pickup/dropoff/fleet/offsets
      - request_time inherits state_before_step.current_request.request_time
      - traffic_level inherits state_before_step.traffic_level
      - current BNN model (hipmdp1, weight_set1)
    The probe never mutates env/buffer. Probe results are stored under the plural
    schema scenario_data['probe_results_by_candidate'][candidate_name][epoch]; the
    orchestration wrapper later splits per-candidate and renames plural → singular.

    Args:
        probe_configs: Dict[str, Dict] — one entry per candidate. Each entry has:
            'pickup' (int), 'dropoff' (int), 'delta_pickup' (float minutes),
            'delta_dropoff' (float minutes), 'vehicle_locations' (List[int]).

    Returns:
        Dict with plural probe schema: scenario_data['probe_results_by_candidate'],
        scenario_data['probe_configs'], per_step_trees keyed f"probe|{name}|{ep}".
    """
    if not probe_configs:
        raise ValueError("run_scenario0_scenario requires non-empty probe_configs")

    try:
        domain = 'paratransit'

        hipmdp1, hipmdp2, weight_set1, weight_set2, latent_mean, latent_std, network_weights = \
            _init_hipmdp_models(seed, requests_csv_path)

        traffic_condition = 0.1
        env = NSParatransitV0(
            n_vehicles=n_vehicles,
            n_requests=n_requests,
            traffic_condition=traffic_condition,
            seed=seed,
            data_path=data_path,
            use_real_data=True,
            fixed_request_ids=fixed_request_ids,
            requests_csv_path=requests_csv_path,
            max_time=480
        )

        n_requests = env.n_requests

        env.set_congestion_config(
            congestion_epoch=congestion_epoch,
            congestion_traffic_level=congestion_traffic_level
        )

        trigger_epoch = congestion_epoch

        scenario_name = "Scenario 0: Controlled Adaptation (real Case 3 + fixed probe)"
        print(f"\n[SCENARIO] {scenario_name}")
        print(f"  Congestion begins at epoch {congestion_epoch}")
        print(f"  Traffic level: {traffic_condition} -> {congestion_traffic_level}")
        print(f"  Probe candidates: {list(probe_configs.keys())}")

        bnn_signature = f"bnn_{hashlib.md5(str(network_weights).encode()).hexdigest()[:8]}"

        per_step_trees = {}
        recommended_assignments = []
        assignment_details = []
        completed_by_vehicle = [set() for _ in range(n_vehicles)]
        mdp_comparisons = {}

        # Probe storage — plural schema
        probe_results_by_candidate = {name: {} for name in probe_configs.keys()}

        current_model_version = 0
        model_version_history = {
            0: {
                'start_epoch': 0,
                'network_weights': deepcopy(hipmdp1.network.weights),
                'latent_weights': weight_set1.copy(),
                'baseline_network_weights': deepcopy(hipmdp2.network.weights),
                'baseline_latent_weights': weight_set2.copy(),
            }
        }

        episode_buffer = []
        step_idx = 0
        model_update_history = []
        dpas_history = {}

        best_network_error = 100.0
        best_latent_error = 100.0
        local_converge_count = 0

        state = env.reset()
        done = False
        total_reward = 0.0
        decision_epoch = 0

        print(f"\n[SCENARIO] Running with {n_vehicles} vehicles, {n_requests} requests")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Episode boundary at epoch {congestion_epoch} (M̂k-1 will be refreshed)")
        print()

        episode_boundary_epoch = congestion_epoch
        episode_boundary_applied = False
        current_episode = 1

        MCTS.previous_epistemic_uncertainty = None
        MCTS.previous_aleatoric_uncertainty = None

        while not done and decision_epoch < n_requests:
            # Episode boundary reset (identical to run_congestion_scenario)
            if (episode_boundary_epoch is not None and
                    decision_epoch == episode_boundary_epoch and
                    not episode_boundary_applied):
                print(f"\n{'='*60}")
                print(f"  [EPISODE BOUNDARY] Epoch {decision_epoch}: Citywide congestion detected")
                print(f"  Traffic level: {traffic_condition} -> {congestion_traffic_level}")

                old_w_b = weight_set1.copy()
                hipmdp2.network.weights = deepcopy(hipmdp1.network.weights)
                weight_set2 = weight_set1.copy()

                episode_buffer.clear()
                step_idx = 0

                ep2_seed = seed + 10000 + decision_epoch
                np.random.seed(ep2_seed)
                effective_std = np.maximum(latent_std, 0.05)
                weight_set1 = latent_mean + effective_std * np.random.randn(len(latent_mean))
                print(f"  w_b shift: {np.linalg.norm(weight_set1 - old_w_b):.6f}")
                print(f"{'='*60}\n")

                best_network_error = 100.0
                best_latent_error = 100.0
                local_converge_count = 0

                MCTS.previous_epistemic_uncertainty = None
                MCTS.previous_aleatoric_uncertainty = None

                current_model_version += 1
                model_version_history[current_model_version] = {
                    'start_epoch': decision_epoch,
                    'network_weights': deepcopy(hipmdp1.network.weights),
                    'latent_weights': weight_set1.copy(),
                    'baseline_network_weights': deepcopy(hipmdp2.network.weights),
                    'baseline_latent_weights': weight_set2.copy(),
                    'is_episode_boundary': True,
                }

                MCTS.clear_bnn_cache()
                episode_boundary_applied = True
                current_episode = 2

            state_obs = env.state_to_observation(state)

            epoch_seed = seed + decision_epoch
            random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            if HAS_AUTOGRAD:
                npr.seed(epoch_seed)
            if HAS_TORCH:
                torch.manual_seed(epoch_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(epoch_seed)

            MCTS.reset_epoch_dpas_stats()

            mcts = MCTS(
                state_obs, state, hipmdp1, hipmdp2,
                weight_set1, weight_set2,
                decision_epoch, env, 0.02, False, False,
                seed=epoch_seed
            )
            mcts.search(max_iterations)

            _log_and_store_dpas(mcts, decision_epoch, dpas_history, model_version=current_model_version)
            best_action = _select_best_action(mcts)

            step_key = f"ep_1|{decision_epoch}"

            vehicle_snapshot = []
            for veh in state.vehicles:
                vehicle_snapshot.append({
                    'current_occupancy': getattr(veh, 'current_occupancy', 0),
                    'route': list(getattr(veh, 'route', [])),
                    'current_location': getattr(veh, 'current_location', 0),
                    'capacity': getattr(veh, 'capacity', 3)
                })

            state_before_step = state
            state, reward, done, info = env.step(best_action)

            for v_id in range(n_vehicles):
                completed_by_vehicle[v_id] = set(
                    req_id for req_id, veh_id in env.request_assignments.items()
                    if veh_id == v_id and env.request_status.get(req_id) == "dropped-off"
                )

            state_obs_before = env.state_to_observation(state_before_step)
            state_obs_after = env.state_to_observation(state)
            action_one_hot = np.zeros(n_vehicles)
            action_one_hot[best_action] = 1
            transition = [
                state_obs_before,
                action_one_hot,
                reward,
                state_obs_after,
                weight_set1.copy()
            ]
            episode_buffer.append(transition)
            step_idx += 1

            # ============ PROBE INSERTION ============
            # Evaluate each candidate probe snapshot against state_before_step's
            # request_time + traffic_level under the current BNN, AFTER real
            # transition is recorded and BEFORE any training trigger this epoch.
            for candidate_idx, (candidate_name, cfg) in enumerate(probe_configs.items()):
                probe_reference_time = state_before_step.current_request.request_time
                probe_traffic_level = state_before_step.traffic_level

                probe_vehicles = [
                    VehicleState(
                        vehicle_id=i,
                        current_location=cfg['vehicle_locations'][i],
                        current_time=probe_reference_time,
                        current_occupancy=0,
                        capacity=3,
                    )
                    for i in range(n_vehicles)
                ]

                probe_request_id = 100000 + candidate_idx * 1000 + decision_epoch
                probe_request = PassengerRequest(
                    request_id=probe_request_id,
                    pickup_node=cfg['pickup'],
                    dropoff_node=cfg['dropoff'],
                    request_time=probe_reference_time,
                    earliest_pickup=probe_reference_time + cfg['delta_pickup'],
                    latest_dropoff=probe_reference_time + cfg['delta_dropoff'],
                )

                probe_state = ParatransitState(
                    decision_epoch=decision_epoch,
                    current_request=probe_request,
                    vehicles=probe_vehicles,
                    traffic_level=probe_traffic_level,
                )
                probe_obs = env.state_to_observation(probe_state)

                probe_seed = epoch_seed + 777 + candidate_idx
                random.seed(probe_seed)
                np.random.seed(probe_seed)
                if HAS_AUTOGRAD:
                    npr.seed(probe_seed)
                if HAS_TORCH:
                    torch.manual_seed(probe_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(probe_seed)

                MCTS.reset_epoch_dpas_stats()
                probe_mcts = MCTS(
                    probe_obs, probe_state, hipmdp1, hipmdp2,
                    weight_set1, weight_set2,
                    decision_epoch, env, 0.02, False, False,
                    seed=probe_seed,
                )
                probe_mcts.search(max_iterations)
                probe_best_action = _select_best_action(probe_mcts)

                probe_vehicle_snapshot = [
                    {
                        'vehicle_id': i,
                        'current_location': cfg['vehicle_locations'][i],
                        'current_time': probe_reference_time,
                        'current_occupancy': 0,
                        'capacity': 3,
                        'route': [],
                        'next_time': probe_reference_time,
                    }
                    for i in range(n_vehicles)
                ]
                probe_vehicle_options = _build_vehicle_options(
                    probe_vehicle_snapshot, probe_request, env,
                    probe_traffic_level, probe_best_action,
                    completed_by_vehicle=[set() for _ in range(n_vehicles)],
                )

                probe_tree_key = f"probe|{candidate_name}|{decision_epoch}"
                probe_results_by_candidate[candidate_name][decision_epoch] = {
                    'assigned_vehicle': probe_best_action,
                    'vehicle_options': probe_vehicle_options.get('vehicle_options', []),
                    'closest_vehicle': probe_vehicle_options.get('closest_vehicle'),
                    'closest_distance': probe_vehicle_options.get('closest_distance'),
                    'chosen_vehicle_distance': probe_vehicle_options.get('chosen_vehicle_distance'),
                    'vehicle_snapshot': probe_vehicle_snapshot,
                    'traffic_level': probe_traffic_level,
                    'model_version': current_model_version,
                    'tree_key': probe_tree_key,
                    'request_id': probe_request_id,
                    'request_time': probe_request.request_time,
                    'earliest_pickup': probe_request.earliest_pickup,
                    'latest_dropoff': probe_request.latest_dropoff,
                }
                per_step_trees[probe_tree_key] = serialize_node_for_json(probe_mcts.root, max_depth=8)

                print(f"    [PROBE {candidate_name}] epoch {decision_epoch}: V{probe_best_action} "
                      f"(traffic_level={probe_traffic_level}, model_version={current_model_version})")
            # ============ END PROBE ============

            # Build assignment detail for the real transition
            if hasattr(state_before_step, 'current_request'):
                request = state_before_step.current_request
                request_id = request.request_id if hasattr(request, 'request_id') else decision_epoch

                detail = {
                    'request_id': request_id,
                    'decision_epoch': decision_epoch,
                    'assigned_vehicle': best_action,
                    'pickup_node': getattr(request, 'pickup_node', None),
                    'dropoff_node': getattr(request, 'dropoff_node', None),
                    'request_time': getattr(request, 'request_time', None),
                    'earliest_pickup': getattr(request, 'earliest_pickup', None),
                    'latest_dropoff': getattr(request, 'latest_dropoff', None),
                    'is_congestion_epoch': decision_epoch >= trigger_epoch,
                    'traffic_level': getattr(state_before_step, 'traffic_level', None),
                }

                vo = _build_vehicle_options(
                    vehicle_snapshot, request, env,
                    state_before_step.traffic_level, best_action, completed_by_vehicle)
                closest_vehicle = vo['closest_vehicle']
                detail.update({k: v for k, v in vo.items() if k != 'distance_by_vehicle'})

                assignment_details.append(detail)
                per_step_trees[step_key] = serialize_node_for_json(mcts.root, max_depth=8)

                _prev_vid = None
                for _v in range(current_model_version - 1, -1, -1):
                    if _v in model_version_history:
                        _prev_vid = _v
                        break
                if _prev_vid is None and 0 in model_version_history:
                    _prev_vid = 0

                state_snapshot = {
                    'state_obs': env.state_to_observation(state_before_step),
                    'state': _serialize_paratransit_state(state_before_step),
                    'epoch': decision_epoch,
                    'model_version_at_t': current_model_version,
                    'prev_version_id': _prev_vid,
                    'env_params': {
                        'n_vehicles': n_vehicles,
                        'n_requests': n_requests,
                        'traffic_condition': traffic_condition,
                        'seed': seed,
                        'data_path': data_path,
                        'fixed_request_ids': fixed_request_ids,
                        'requests_csv_path': requests_csv_path,
                        'congestion_epoch': congestion_epoch,
                        'congestion_traffic_level': congestion_traffic_level,
                        'max_time': env.max_time,
                    },
                    'env_history': {
                        'pickup_times': dict(getattr(env, 'pickup_times', {})),
                        'dropoff_times': dict(getattr(env, 'dropoff_times', {})),
                        'request_status': dict(getattr(env, 'request_status', {})),
                        'request_assignments': dict(getattr(env, 'request_assignments', {})),
                        'original_request_ids': [r.original_id for r in getattr(env, 'requests', [])],
                    },
                }

                mdp_t_metrics = {
                    'assigned_vehicle': extract_pctl_metrics_for_vehicle(mcts, best_action, env, decision_epoch),
                }
                if closest_vehicle is not None:
                    mdp_t_metrics['closest_vehicle'] = extract_pctl_metrics_for_vehicle(mcts, closest_vehicle, env, decision_epoch)

                assignment_info = {
                    'assigned_vehicle': best_action,
                    'closest_vehicle': closest_vehicle,
                    'assigned_distance': detail.get('chosen_vehicle_distance'),
                    'closest_distance': detail.get('closest_distance'),
                    'assigned_pending': detail.get('chosen_vehicle_pending'),
                    'closest_pending': None,
                }
                if closest_vehicle is not None and closest_vehicle < len(vehicle_snapshot):
                    closest_snap = vehicle_snapshot[closest_vehicle]
                    assignment_info['closest_pending'] = len(set(req_id for req_id, _ in closest_snap['route']))

                mdp_t_metrics['assignment_info'] = assignment_info

                is_trigger_active = (decision_epoch >= trigger_epoch)

                if is_trigger_active:
                    if decision_epoch == trigger_epoch:
                        print(f"  [CONGESTION] Epoch {decision_epoch}: Citywide congestion active (traffic_level -> {congestion_traffic_level})")

                    event_info = {
                        'trigger_epoch': trigger_epoch,
                        'mdp_t_assignment': best_action,
                        'mdp_t_minus_n_assignment': None,
                        'assignment_changed': None,
                        'current_epoch': decision_epoch,
                        'model_version': current_model_version,
                        'closest_vehicle': closest_vehicle,
                        'assigned_distance': detail.get('chosen_vehicle_distance'),
                        'closest_distance': detail.get('closest_distance'),
                        'congestion_epoch': congestion_epoch,
                        'base_traffic_level': traffic_condition,
                        'congestion_traffic_level': congestion_traffic_level,
                    }

                    mdp_comparisons[decision_epoch] = {
                        'mdp_t': mdp_t_metrics,
                        'mdp_t_minus_n': None,
                        'state_snapshot': state_snapshot,
                        'event_info': event_info,
                    }
                else:
                    mdp_comparisons[decision_epoch] = {
                        'mdp_t': mdp_t_metrics,
                        'mdp_t_minus_n': None,
                        'state_snapshot': state_snapshot,
                    }
            else:
                request_id = decision_epoch

            recommended_assignments.append((request_id, best_action))
            total_reward += reward

            if assignment_details:
                last_detail = assignment_details[-1]
                earliest = last_detail.get('earliest_pickup')
                latest = last_detail.get('latest_dropoff')
                pickup_node = last_detail.get('pickup_node')
                dropoff_node = last_detail.get('dropoff_node')
                tl = last_detail.get('traffic_level', '?')
                info_parts = [f"Request {request_id} -> Vehicle {best_action}"]
                if pickup_node is not None and dropoff_node is not None:
                    info_parts.append(f"({pickup_node}->{dropoff_node})")
                if earliest is not None and latest is not None:
                    info_parts.append(f"[window:{earliest:.0f}-{latest:.0f}]")
                actual_pickup = env.pickup_times.get(request_id)
                actual_dropoff = env.dropoff_times.get(request_id)
                if actual_pickup is not None or actual_dropoff is not None:
                    pickup_str = f"{actual_pickup:.0f}" if actual_pickup is not None else "?"
                    dropoff_str = f"{actual_dropoff:.0f}" if actual_dropoff is not None else "?"
                    info_parts.append(f"[actual:{pickup_str}/{dropoff_str}]")
                info_parts.append(f"tl:{tl}")
                info_parts.append(f"reward:{reward:.2f}")
                print(f"  Epoch {decision_epoch}: {' '.join(info_parts)}")
            else:
                print(f"  Epoch {decision_epoch}: Request {request_id} -> Vehicle {best_action} (reward: {reward:.2f})")

            # Training trigger (unchanged; fires AFTER the probe has been recorded)
            should_update = (step_idx % N_interval == 0 and
                             len(episode_buffer) >= N_min)

            if should_update:
                try:
                    train_result = _train_model_from_buffers(
                        hipmdp1, hipmdp2, weight_set1, weight_set2,
                        episode_buffer,
                        seed, domain, ada_mcts_path,
                        best_network_error, best_latent_error, local_converge_count, Nu,
                        decision_epoch, current_model_version, model_update_history,
                        verbose=False,
                    )
                    weight_set1 = train_result['weight_set1']
                    best_network_error = train_result['best_network_error']
                    best_latent_error = train_result['best_latent_error']
                    latent_mean = train_result['latent_mean']
                    latent_std = train_result['latent_std']
                    current_model_version = train_result['current_model_version']
                    model_version_history[current_model_version] = train_result['version_history_entry']

                    _needed = {0, current_model_version}
                    for _comp in mdp_comparisons.values():
                        _ss = _comp.get('state_snapshot', {})
                        _needed.add(_ss.get('model_version_at_t', 0))
                        _pv = _ss.get('prev_version_id')
                        if _pv is not None:
                            _needed.add(_pv)
                    _to_drop = [v for v in model_version_history if v not in _needed]
                    for v in _to_drop:
                        del model_version_history[v]

                except Exception as e:
                    if DEBUG_MODE:
                        print(f"        [WARNING] train_model failed: {e}")

            decision_epoch += 1

        print(f"\n[COMPLETE] Total reward: {total_reward:.2f}")

        traffic_level_history = getattr(env, 'traffic_level_history', {})

        env_data = {
            'n_vehicles': n_vehicles,
            'n_requests': n_requests,
            'traffic_level': traffic_condition,
            'recommended_assignments': recommended_assignments,
            'assignment_details': assignment_details,
            'total_reward': total_reward,
            'final_state': str(state),
            'violations': getattr(env, 'violations', []),
            'seed': seed,
            'domain': 'paratransit',
            'scenario': 'controlled_adaptation',
            'traffic_level_history': traffic_level_history,
        }

        planned_index_map = {f"ep_1|{req_id}": f"ep_1|{req_id}" for req_id, _ in recommended_assignments}

        _needed_versions = {0}
        for _comp in mdp_comparisons.values():
            _ss = _comp.get('state_snapshot', {})
            _needed_versions.add(_ss.get('model_version_at_t', 0))
            _pv = _ss.get('prev_version_id')
            if _pv is not None:
                _needed_versions.add(_pv)
        model_version_history = {v: model_version_history[v]
                                 for v in _needed_versions if v in model_version_history}

        result = {
            'env_data': env_data,
            'per_step_trees': per_step_trees,
            'planned_index_map': planned_index_map,
            'tree_data': {
                'recommended_assignments': recommended_assignments,
                'n_epochs': decision_epoch
            },
            'bnn_signature': bnn_signature,
            'reward': total_reward,
            'scenario_data': {
                'type': 'controlled_adaptation',
                'mdp_comparisons': mdp_comparisons,
                'model_version_history': model_version_history,
                'final_model_version': current_model_version,
                'model_update_history': model_update_history,
                'total_experience_collected_Db': len(episode_buffer),
                'congestion_config': {
                    'congestion_epoch': congestion_epoch,
                    'congestion_traffic_level': congestion_traffic_level,
                    'base_traffic_level': traffic_condition,
                },
                'trigger_epoch': trigger_epoch,
                'assignment_changed': None,
                'mdp_t_minus_n_assignment': None,
                'mdp_t_assignment': None,
                'comparison_epoch': None,
                'dpas_history': dpas_history,
                'episode_boundary_epoch': episode_boundary_epoch,
                'num_episodes': current_episode,
                # Plural probe schema (runtime only; split into singular on disk)
                'probe_results_by_candidate': probe_results_by_candidate,
                'probe_configs': dict(probe_configs),
            }
        }

        return result

    except Exception as e:
        print(f"[ERROR] Scenario 0 run failed: {e}")
        traceback.print_exc()
        raise


def main():
    """Main entry point for subprocess execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Run ADA-MCTS for Paratransit')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--n_vehicles', type=int, default=5, help='Number of vehicles')
    parser.add_argument('--n_requests', type=int, default=10, help='Number of requests')
    parser.add_argument('--max_iterations', type=int, default=3000, help='MCTS iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    print(f"[PARATRANSIT RUNNER] Starting ADA-MCTS")
    print(f"  Vehicles: {args.n_vehicles}")
    print(f"  Requests: {args.n_requests}")
    print(f"  Iterations: {args.max_iterations}")
    print(f"  Seed: {args.seed}")

    # Run ADA-MCTS
    result = run_ada_mcts_paratransit(
        n_vehicles=args.n_vehicles,
        n_requests=args.n_requests,
        max_iterations=args.max_iterations,
        seed=args.seed
    )

    # Wrap result for adapter
    output_data = {
        'mcts_data': result,
        'reward': result['reward']
    }

    # Write to output file
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"[PARATRANSIT RUNNER] Completed successfully")
    print(f"[PARATRANSIT RUNNER] Total reward: {result['reward']:.2f}")
    print(f"[PARATRANSIT RUNNER] Output written to: {args.output}")


if __name__ == '__main__':
    main()
