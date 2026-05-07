import math
import random
#from nsbridge_simulator.nsbridge_v0 import NSBridgeV0 as model
#from nsfrozenlake.nsfrozenlake_v0 import NSFrozenLakeV0 as model
import pickle
from BNN.BayesianNeuralNetwork import *
import autograd.numpy as np
import utils.distribution as distribution
from collections import Counter
# import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from HiPMDP import HiPMDP
import logging
from numbers import Integral
from nsparatransit.nsparatransit_v0 import ParatransitState, VehicleState


def fast_clone_paratransit_state(state):
    """
    PERFORMANCE: Fast shallow clone for ParatransitState.

    Only clones mutable parts (vehicles and their routes) that get modified during MCTS.
    Immutable parts (PassengerRequest) are shared via reference.

    This is 10-100x faster than deepcopy while maintaining correctness.
    """
    # Clone vehicles with new route lists
    cloned_vehicles = []
    for v in state.vehicles:
        cloned_vehicles.append(VehicleState(
            vehicle_id=v.vehicle_id,
            current_location=v.current_location,
            current_time=v.current_time,
            current_occupancy=v.current_occupancy,
            capacity=v.capacity,
            route=list(v.route),  # New list (shallow copy of tuples is fine)
            next_time=v.next_time
        ))

    # Return new ParatransitState with cloned vehicles
    # PassengerRequest can be shared (immutable in MCTS context)
    return ParatransitState(
        decision_epoch=state.decision_epoch,
        current_request=state.current_request,  # Shared reference (not modified)
        vehicles=cloned_vehicles,
        traffic_level=state.traffic_level
    )


def compute_aleatoric_over_actions(state, actions, BNN, weight_set, task):
    """
    PAPER-COMPLIANT FIX (Issue 2): Compute aleatoric uncertainty over a subset of actions.

    Paper Equation 8: VarA(M̂k; S') = (1/|S'|) * sum_{(s,a) in S'} [ (1/N) * sum_i σ²_i(s,a; W_i) ]

    This function computes aleatoric uncertainty averaged over all valid actions at state s.

    Args:
        state: Current state
        actions: List of valid actions at this state
        BNN: Bayesian Neural Network (HiPMDP)
        weight_set: Latent weights
        task: Task environment
        
    Returns:
        Mean aleatoric uncertainty across all (s,a) pairs where a ∈ actions
    """
    task_name = type(task).__name__

    # Encode state ONCE (same for all actions)
    if task_name == 'NSFrozenLakeV0':
        state_obs = BNN.task._NSFrozenLakeV0__encode_state(state)
        n_actions = 4  # Frozen lake
    elif task_name == 'NSParatransitV0':
        state_obs = BNN.task.state_to_observation(state)
        n_actions = task.nA  # Total actions (5 vehicles)
    else:
        raise ValueError(f"Unknown task type: {task_name}")

    state_obs_1d = np.ravel(state_obs)
    weight_set_1d = np.ravel(weight_set)
    _agg = getattr(MCTS, 'uncertainty_aggregation', 'mean') if 'MCTS' in dir() else 'mean'

    # Precise cache key: full state_obs + actions set + BNN identity + weight_set + aggregation
    # id(BNN.network) distinguishes BNN1 vs BNN2; cache is cleared on every model update
    # so id stability within a cache lifetime is guaranteed.
    cache_key = (
        tuple(state_obs_1d),
        tuple(sorted(actions)),
        id(BNN.network),
        tuple(weight_set_1d),
        _agg,
    )
    if cache_key in Node.aleatoric_cache:
        return Node.aleatoric_cache[cache_key]

    # Build batched augmented input: one row per action, sharing state_obs & weight_set
    # CRITICAL: Use total action space size for one-hot, NOT len(valid actions)!
    n = len(actions)
    action_one_hots = np.zeros((n, n_actions))
    for i, action in enumerate(actions):
        action_one_hots[i, action] = 1

    state_tile = np.tile(state_obs_1d, (n, 1))       # (n, obs_dim)
    weight_tile = np.tile(weight_set_1d, (n, 1))      # (n, weight_dim)
    aug_states = np.hstack([state_tile, action_one_hots, weight_tile])  # (n, input_dim)

    # Single batched BNN forward pass (replaces n separate calls)
    # feed_forward_distribution / __predict__ support batch inputs natively
    _, _, _, aleatoric_uncertainties = BNN.network.feed_forward_distribution(aug_states)
    # aleatoric_uncertainties shape: (n, output_dims)

    # Guard against non-finite outputs — conservative: treat as high uncertainty
    # +inf ensures non-finite values push DPAS toward pessimistic sampling
    safe_unc = np.nan_to_num(aleatoric_uncertainties, nan=np.inf, neginf=np.inf, posinf=np.inf)

    # Aggregate aleatoric across output dimensions for each (s,a)
    # Uses MCTS.uncertainty_aggregation for consistency with DPAS
    if _agg == 'sum':
        per_action = np.sum(safe_unc, axis=-1)    # (n,)
    elif _agg == 'max':
        per_action = np.max(safe_unc, axis=-1)    # (n,)
    else:
        per_action = np.mean(safe_unc, axis=-1)   # (n,)

    result = float(np.mean(per_action))
    Node.aleatoric_cache[cache_key] = result
    return result


class Node:
    bnn_cache = {}  # Cache BNN predictions for efficiency
    bnn_cache_hits = 0  # DEBUG: Track cache hits
    bnn_cache_misses = 0  # DEBUG: Track cache misses
    aleatoric_cache = {}  # Cache compute_aleatoric_over_actions results (precise key)

    def __init__(self, state, time, task, danger, action=None, parent=None, node_type="decision", rng=None):
        self.state = state
        self.action = action  # Action taken or outcome for chance node
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.type = node_type  # "decision" or "chance"
        self.probabilities = [] if self.type == "chance" else None

        # Design 4: Domain-aware action space with dynamic valid actions
        task_name = type(task).__name__

        if task_name == 'NSParatransitV0':
            # For paratransit, use get_valid_actions for context-dependent actions
            # (capacity pruning + conditional reject)
            self.possible_actions = task.get_valid_actions(state)
        elif task_name == 'NSFrozenLakeV0':
            # Frozen lake: all 4 actions always valid
            self.possible_actions = list(range(task.nA))
        else:
            raise ValueError(f"Unknown task type: {task_name}")

        self.discount_factor = 0.9999
        self.time = time
        # Use shared RNG for reproducibility (passed from MCTS or parent)
        self.np_random = rng if rng is not None else np.random.RandomState()
        self.task = task
        self.danger = danger
        # Store uncertainty metrics for explainability
        self.epistemic_uncertainty = None
        self.aleatoric_uncertainty = None
        # NEW: Store decision rationale for explanation
        self.explain = None
        # Precompute metadata required for explanation and PCTL propositions
        self._initialize_state_metadata(state)

        # For step number tracking (optional, for debugging)
        self.created_at_step = None

        # Trace-based PCTL evaluation: store rollout traces for semantic evaluation
        # Each trace is a list of {ap_name: bool} dicts (one per step in rollout)
        # Used by formula_evaluator.py for AST-based PCTL evaluation
        self.rollout_traces = []

        # NEW: Store the transition AP from parent to this node
        # This is used to build complete traces with prefix path from root
        # transition_ap = {ap_name: bool} for the (parent_state, action, this_state) transition
        self.transition_ap = None

    def get_transition(self,bnn_samples, BNN1, state, action):
        state_count = []
        samples = bnn_samples[:, 0, :]

        # Determine domain from task type
        task_name = type(BNN1.task).__name__

        if task_name == 'NSFrozenLakeV0':
            # Frozen Lake: decode samples back to state indices
            num_dims = 2
            num_states = 16
            for j in samples.reshape(-1, num_dims):
                state_count.append(BNN1.task._NSFrozenLakeV0__decode_state(j, state, action))
        elif task_name == 'NSParatransitV0':
            # PAPER-COMPLIANT: BNN predicts complete next state observation (16D)
            # BNN learns p(s'|s,a) from observed (s,a,s') transitions
            #
            # BNN output: next_state_obs (16D) = [norm_epoch, v0_loc, v0_time, v0_occ, ...]
            # This captures the full state transition including stochastic travel times
            #
            # NOTE: This categorical distribution over decision_epoch is LEGACY CODE
            # and is NO LONGER USED in the main DPAS path (Issues 4 & 7 fixed).
            # For paratransit, DPAS now samples directly from BNN outputs (samples1/samples2)
            # instead of using get_transition() and categorical_sample().
            #
            # This code is kept for backward compatibility with frozen lake.
            num_dims = 16  # BNN output dimension
            num_states = BNN1.task.n_requests + 1  # Allow terminal state

            # For paratransit, next_decision_epoch is deterministic (always current + 1)
            if hasattr(state, 'decision_epoch'):
                next_decision_epoch = state.decision_epoch + 1
            else:
                next_decision_epoch = 0

            # All samples lead to the same next_decision_epoch (deterministic)
            # The stochasticity is in the vehicle states (captured in BNN samples)
            state_count = [next_decision_epoch] * len(samples)
        else:
            raise ValueError(f"Unknown task type: {task_name}")

        # PERFORMANCE: Vectorized computation using bincount
        state_count_array = np.array(state_count, dtype=np.int32)

        # Check for out-of-bounds states
        out_of_bounds = (state_count_array < 0) | (state_count_array >= num_states)
        if np.any(out_of_bounds):
            bad_states = state_count_array[out_of_bounds]
            print(f"[WARNING] get_transition: {len(bad_states)} states out of bounds [0, {num_states}), task={task_name}")
            # Filter to valid range
            state_count_array = state_count_array[~out_of_bounds]

        # Use bincount for fast counting (much faster than Counter)
        counts = np.bincount(state_count_array, minlength=num_states)
        total_transitions = counts.sum()

        # Normalize to get probabilities
        if total_transitions > 0:
            bnn_transitions1 = (counts / total_transitions).tolist()
        else:
            # Edge case: no valid transitions (shouldn't happen)
            bnn_transitions1 = [0.0] * num_states
            print(f"[WARNING] get_transition: No valid transitions, task={task_name}")

        return bnn_transitions1

    def get_bnn_prediction(self, state, action, BNN1, weight_set1, BNN2, weight_set2, isi):
        # Determine domain and encode state appropriately
        task_name = type(BNN1.task).__name__

        if task_name == 'NSFrozenLakeV0':
            state1 = BNN1.task._NSFrozenLakeV0__encode_state(state)
            cache_key = (state, action)  # Frozen Lake: state is integer, use directly
        elif task_name == 'NSParatransitV0':
            # For paratransit, use the node's own state (not global self.task.state)
            # PAPER-COMPLIANT: state_to_observation now returns 16D (no traffic_level)
            state1 = BNN1.task.state_to_observation(state)

            # FIX: Use balanced cache key precision for speed vs. accuracy trade-off
            # The BNN predicts state transitions, which depend primarily on:
            # 1. Current decision_epoch (which request we're assigning)
            # 2. Action (which vehicle to assign)
            # 3. Vehicle states (location, time, occupancy)
            #
            # TRADE-OFF ANALYSIS:
            # - Too fine (×10000): low cache hit rate (~18%), slow but accurate
            # - Too coarse (×10/×100): high cache hit rate (~96%), fast but may lose precision
            # - Balanced (×200/×50): moderate cache hit rate (~45-60%), good balance
            #
            # Chosen precision (OPTIMIZED BALANCED):
            # - Location: ×200 → ~54 nodes per bin (for 10788 nodes)
            #   Nearby nodes have similar travel times, acceptable for BNN prediction
            # - Time: ×50 → ~10 min per bin (for 480 min max_time)
            #   10-minute precision is reasonable for vehicle scheduling
            # - Occupancy: exact (only 4 values: 0,1,2,3)
            decision_epoch = state.decision_epoch if hasattr(state, 'decision_epoch') else 0

            # Extract vehicle state with balanced precision
            # state1 format: [norm_epoch, v0_loc, v0_time, v0_occ, v1_loc, ...]
            coarse_vehicle_states = []
            n_vehicles = BNN1.task.n_vehicles if hasattr(BNN1.task, 'n_vehicles') else 5
            for v_idx in range(n_vehicles):
                offset = 1 + v_idx * 3
                loc_bin = int(state1[offset + 0] * 2000)     # Location: 2000 bins (~5 nodes/bin)
                time_bin = int(state1[offset + 1] * 200)     # Time: 200 bins (~2.5min each)
                occ = int(state1[offset + 2] * 3 + 0.5)      # Occupancy: exact (0-3)
                coarse_vehicle_states.append((loc_bin, time_bin, occ))

            cache_key = (decision_epoch, action, tuple(coarse_vehicle_states))
        else:
            raise ValueError(f"Unknown task type: {task_name}")

        # Check cache AFTER encoding state (so we can use observation for cache key)
        if cache_key in Node.bnn_cache:
            Node.bnn_cache_hits += 1
            return Node.bnn_cache[cache_key]

        # Cache miss - need to compute BNN prediction
        Node.bnn_cache_misses += 1
        confidence_check = True
        aug_state1 = np.hstack([state1, self.__encode_action(action), weight_set1]).reshape((1, -1))
        aug_state2 = np.hstack([state1, self.__encode_action(action), weight_set2]).reshape((1, -1))
        _, samples1, epistemic_uncertainties1, aleatoric_uncertainties1 = BNN1.network.feed_forward_distribution(
            aug_state1)
        _, samples2, epistemic_uncertainties2, aleatoric_uncertainties2 = BNN2.network.feed_forward_distribution(
            aug_state2)

        # Sanitize raw uncertainty vectors: ensure finite, non-negative (variance semantics)
        # Non-finite → 0 (upstream BNN guard is belt, this is suspenders)
        # Negative → 0 (variances must be ≥ 0)
        def _sanitize_unc(arr):
            a = np.asarray(arr, dtype=float)
            a = np.where(np.isfinite(a), a, 0.0)
            return np.clip(a, 0.0, None)
        epistemic_uncertainties1 = _sanitize_unc(epistemic_uncertainties1)
        aleatoric_uncertainties1 = _sanitize_unc(aleatoric_uncertainties1)
        epistemic_uncertainties2 = _sanitize_unc(epistemic_uncertainties2)
        aleatoric_uncertainties2 = _sanitize_unc(aleatoric_uncertainties2)

        # Aggregate uncertainties according to paper's formulation:
        # - Epistemic (Eq. 9): VarE is already computed per (s,a), aggregate across output dims
        # - Aleatoric (Eq. 8): VarA aggregated across output dims
        # Aggregation method is configurable via MCTS.uncertainty_aggregation ('mean'/'sum'/'max')
        _agg = MCTS.uncertainty_aggregation
        if _agg == 'sum':
            _agg_fn = np.sum
        elif _agg == 'max':
            _agg_fn = np.max
        else:  # default 'mean'
            _agg_fn = np.mean
        epistemic_1_total = _agg_fn(epistemic_uncertainties1)
        aleatoric_1_total = _agg_fn(aleatoric_uncertainties1)
        epistemic_2_total = _agg_fn(epistemic_uncertainties2)
        aleatoric_2_total = _agg_fn(aleatoric_uncertainties2)

        transition1 = self.get_transition(samples1, BNN1, state, action)  # M̂k
        transition2 = self.get_transition(samples2, BNN2, state, action)  # M̂k-1
        pessimistic1 = self.pessimistic_sample(transition2, state, action, isi)  # pwc from M̂k-1

        # Store the results in the cache
        Node.bnn_cache[cache_key] = (
            transition1, samples1, pessimistic1, epistemic_1_total, aleatoric_1_total,
            transition2, samples2, pessimistic1, epistemic_2_total, aleatoric_2_total,
            confidence_check)
        return (transition1, samples1, pessimistic1, epistemic_1_total, aleatoric_1_total,
                transition2, samples2, pessimistic1, epistemic_2_total, aleatoric_2_total,
                confidence_check)
        
    def is_decision_node(self):
        return self.type == "decision"

    def is_chance_node(self):
        return self.type == "chance"

    def get_possible_actions(self):
        return self.possible_actions

    def get_child_with_state(self, state):
        """Find child node with matching state (handles both int and ParatransitState)."""
        for child in self.children:
            # For ParatransitState objects, use __eq__ method (defined in nsparatransit_v0.py)
            if hasattr(state, '__eq__') and hasattr(child.state, '__eq__'):
                if child.state == state:
                    return child
            # For numpy arrays or integer states
            elif np.array_equal(child.state, state):
                return child
        return None

    def _initialize_state_metadata(self, raw_state):
        self.state_index = self._normalize_state_index(raw_state)
        self.state_coordinates = self._compute_state_coordinates(self.state_index)
        self.state_reward = self._compute_state_reward(self.state_index)
        self.goal = bool(self.state_reward == 1.0)
        self.hole = bool(self.state_reward == -1.0)
        self.nearH = self._compute_near_hole_flag()
        # Path-based flags are filled in later by higher-level analyzers
        self.on_direct_path = False
        self.on_opt_path_pre = False
        self.on_opt_path_post = False
        self.new_hole = False
        self.removed_hole = False

    def _normalize_state_index(self, state):
        # Paratransit: extract decision_epoch from ParatransitState
        if hasattr(state, 'decision_epoch'):
            return int(state.decision_epoch)

        # Frozen Lake: state is integer
        if isinstance(state, Integral):
            return int(state)

        if hasattr(state, 'index'):
            try:
                return int(state.index)
            except Exception:
                return None
        if isinstance(state, (list, tuple)) and len(state) >= 2 and self.task is not None:
            try:
                return int(self.task.to_s(int(state[0]), int(state[1])))
            except Exception:
                return None
        try:
            return int(state)
        except Exception:
            return None

    def _compute_state_coordinates(self, state_index):
        if self.task is None or state_index is None:
            return None
        try:
            # Check if task has to_m method (Frozen Lake)
            if hasattr(self.task, 'to_m'):
                row, col = self.task.to_m(state_index)
                return (int(row), int(col))
            # For paratransit, return decision epoch as coordinate
            else:
                return (int(state_index), 0)
        except Exception:
            return None

    def _compute_state_reward(self, state_index):
        if self.task is None or state_index is None:
            return 0.0
        try:
            return float(self.task.instant_reward_byindex(state_index))
        except Exception:
            return 0.0

    def _compute_near_hole_flag(self):
        coords = self.state_coordinates
        if self.task is None or coords is None:
            return False
        row, col = coords
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            adj_row = row + dr
            adj_col = col + dc
            if self._is_within_bounds(adj_row, adj_col):
                try:
                    adj_index = self.task.to_s(adj_row, adj_col)
                    reward = self.task.instant_reward_byindex(adj_index)
                except Exception:
                    reward = 0.0
                if reward == -1.0:
                    return True
        return False

    def _is_within_bounds(self, row, col):
        if self.task is None:
            return False
        try:
            return 0 <= row < self.task.nrow and 0 <= col < self.task.ncol
        except Exception:
            return False

    def export_atomic_propositions(self):
        """Return current atomic proposition flags in a single dict."""
        return {
            'goal': bool(getattr(self, 'goal', False)),
            'hole': bool(getattr(self, 'hole', False)),
            'nearH': bool(getattr(self, 'nearH', False)),
            'on_direct_path': bool(getattr(self, 'on_direct_path', False)),
            'on_opt_path_pre': bool(getattr(self, 'on_opt_path_pre', False)),
            'on_opt_path_post': bool(getattr(self, 'on_opt_path_post', False)),
            'new_hole': bool(getattr(self, 'new_hole', False)),
            'removed_hole': bool(getattr(self, 'removed_hole', False)),
        }

    def _estimate_V_hat(self, state_idx):
        """
        PAPER-COMPLIANT: Estimate V̂(s') for pwc construction.

        Paper Eq.(6): pwc selects successor with worst VALUE, not worst instant reward.
        Uses 1-step Bellman estimate: V̂(s) = R(s) + γ * avg_a avg_{s'} R(s')
        for non-terminal states. This distinguishes states near holes/goals
        unlike instant_reward which returns 0 for all non-terminal states.

        Args:
            state_idx: Integer state index (Frozen Lake)

        Returns:
            Estimated state value (float)
        """
        reward = self.task.instant_reward_byindex(state_idx)
        # Terminal states: return immediate reward
        if reward != 0.0:
            return reward

        gamma = 0.99
        neighbor_values = []
        n_actions = self.task.nA if hasattr(self.task, 'nA') else 4

        for a in range(n_actions):
            # Use reachable_states to find neighbors
            if hasattr(self.task, 'reachable_states'):
                rs = np.array(self.task.reachable_states(state_idx, a))
                reachable = np.where(rs == 1)[0]
                for s_prime in reachable:
                    neighbor_values.append(self.task.instant_reward_byindex(s_prime))
            else:
                # Fallback: grid neighbors (Frozen Lake is nrow x ncol)
                if hasattr(self.task, 'nrow') and hasattr(self.task, 'ncol'):
                    row, col = self.task.to_m(state_idx)
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < self.task.nrow and 0 <= nc < self.task.ncol:
                            neighbor_values.append(
                                self.task.instant_reward_byindex(self.task.to_s(nr, nc))
                            )

        if neighbor_values:
            return reward + gamma * np.mean(neighbor_values)
        return reward

    def pessimistic_sample(self, w0, current_state, curren_action, isi):
        """
        PAPER-COMPLIANT: Construct worst-case distribution pwc.

        Paper Eq.(6): pwc selects successor with worst V̂(s') (value-level),
        not worst instant reward. Uses _estimate_V_hat for value estimation.

        For Frozen Lake: select worst-value state from distribution support,
        optionally mix with Wasserstein.
        For Paratransit: fallback to w_worst (handled separately via
        _choose_worst_next_state_sample).
        """
        w0 = np.asarray(w0, dtype=float)
        support = np.where(w0 > 0)[0]

        # If support is empty, return original distribution
        if support.size == 0:
            return w0.tolist()

        # PAPER-COMPLIANT: Find worst state by V̂(s') instead of instant_reward
        v0 = np.array([self._estimate_V_hat(i) for i in support])
        worst_idx_in_support = np.argmin(v0)
        worst_state = support[worst_idx_in_support]

        # Construct one-hot distribution at worst state
        w_worst = np.zeros(len(w0))
        w_worst[worst_state] = 1.0

        # If danger mode is off, or task lacks reachable_states/distances_matrix, return w_worst
        if not self.danger:
            return w_worst.tolist()

        # Check if task has required methods for Wasserstein mixing
        if not (hasattr(self.task, 'reachable_states') and hasattr(self.task, 'distances_matrix')):
            # Paratransit doesn't have these - return w_worst directly
            return w_worst.tolist()

        # Wasserstein-based mixing (Frozen Lake only)
        try:
            c = 1  # Wasserstein radius
            rs = np.array(self.task.reachable_states(current_state, curren_action))
            reachable_states = np.where(rs == 1)[0]
            d = self.task.distances_matrix(reachable_states.tolist())

            w0_dis = w0[reachable_states]
            w_worst_dis = w_worst[reachable_states]

            wass_value = distribution.wass_dual(w0_dis, w_worst_dis, d)
            if wass_value <= c:
                return w_worst.tolist()

            lbd = c / wass_value
            w = (1.0 - lbd) * w0_dis + lbd * w_worst_dis

            new_w = np.zeros(len(w0))
            new_w[reachable_states] = w
            return new_w.tolist()
        except Exception:
            # Fallback if Wasserstein mixing fails
            return w_worst.tolist()


    def _choose_worst_next_state_sample(self, samples, state, action, BNN, weight_set,
                                         tree_children=None):
        """
        Paper Eq.(6)/Alg2: Construct pwc using MCTS tree-backed V, then sample.

        Paper Algorithm 2, Line 58: s' ~ pwc(·|s,a,M̂_{k-1})

        $V(\nu^{s,a}) = R(s,a) + \gamma \mathbb{E}_{s' \sim p_{wc}(\cdot|s,a)} V(\nu^{s'})$

        The pwc distribution concentrates mass on the successor with the
        **lowest** backed-up value — i.e. worst-case for the agent.

        Value source:
        * If ``tree_children`` contains visited decision-node children of
          THIS chance node (s,a), their Q = value/visits IS the search
          value V(ν^{s'}).  Each BNN sample is mapped to its nearest
          tree child (observation-space L2) and scored by that Q.
        * When no tree children are available (first expansion, or during
          rollout where no tree exists for the current state), the
          1-step reward r(s,a,s') is used as a conservative fallback.

        Args:
            samples: BNN samples of next_state_obs  [N, 1, output_dims]
            state: Current ParatransitState (the 's' in p(s'|s,a))
            action: Action taken
            BNN: HiPMDP whose task.state_to_observation is used
            weight_set: Latent weights (kept for call-site compat)
            tree_children: List of Node children of the chance node for
                           (state, action).  Pass ``self.children`` from
                           expand(); pass ``[]`` or ``None`` from rollout
                           (where no tree exists for the current state).

        Returns:
            Index into ``samples`` drawn from the pwc distribution.
        """
        if tree_children is None:
            tree_children = []

        n_samples = len(samples)

        # ------------------------------------------------------------------
        # Collect tree-backed Q-values from visited children
        # ------------------------------------------------------------------
        child_obs_list = []   # observation vectors of visited children
        child_q_list = []     # backed-up Q = value / visits
        for child in tree_children:
            if child.visits > 0:
                try:
                    obs = BNN.task.state_to_observation(child.state)
                    child_obs_list.append(obs)
                    child_q_list.append(child.value / child.visits)
                except Exception:
                    pass

        has_tree_info = len(child_obs_list) > 0

        # ------------------------------------------------------------------
        # Score each BNN sample
        # ------------------------------------------------------------------
        v_hat_values = np.zeros(n_samples)

        for i in range(n_samples):
            next_obs = samples[i, 0, :]

            if has_tree_info:
                # Map sample → nearest tree child → use that child's Q
                distances = [np.sum((next_obs - co) ** 2) for co in child_obs_list]
                nearest_idx = int(np.argmin(distances))
                v_hat_values[i] = child_q_list[nearest_idx]
            else:
                # Fallback: 1-step reward (no tree info for this state)
                next_epoch = state.decision_epoch + 1
                temp_next_state = self._decode_bnn_output_to_state(
                    next_obs, state, action, next_epoch
                )
                v_hat_values[i] = self.task.compute_rollout_reward(
                    state, action, temp_next_state
                )

        # ------------------------------------------------------------------
        # Construct pwc: one-hot on argmin V̂  (simplest valid pwc)
        # ------------------------------------------------------------------
        worst_idx = int(np.argmin(v_hat_values))
        pwc = np.zeros(n_samples)
        pwc[worst_idx] = 1.0

        # Sample from pwc (paper Line 58: s' ~ pwc)
        return self.np_random.choice(n_samples, p=pwc)

    def _decode_bnn_output_to_state(self, next_obs, current_state, action, next_epoch):
        """
        PAPER-COMPLIANT: Decode BNN output (16D observation) to ParatransitState.

        BNN learns p(s'|s,a) where s' is encoded as 16D observation:
        [norm_epoch, v0.loc, v0.time, v0.occ, ..., v4.loc, v4.time, v4.occ]

        Args:
            next_obs: BNN-predicted next observation (16D numpy array)
            current_state: Current ParatransitState
            action: Action taken
            next_epoch: Next decision epoch (deterministic)

        Returns:
            ParatransitState reconstructed from BNN prediction
        """
        # PERFORMANCE: Fast clone instead of deepcopy
        predicted_state = fast_clone_paratransit_state(current_state)

        # Update decision_epoch (deterministic)
        predicted_state.decision_epoch = next_epoch

        # Decode vehicle states from BNN prediction (16D)
        # Format: [norm_epoch, v0_loc, v0_time, v0_occ, v1_loc, v1_time, v1_occ, ...]
        n_vehicles = self.task.n_vehicles

        for v_idx in range(n_vehicles):
            # Extract this vehicle's features from observation
            # Offset: 1 (epoch) + v_idx * 3 (features per vehicle)
            offset = 1 + v_idx * 3

            norm_loc = next_obs[offset + 0]
            norm_time = next_obs[offset + 1]
            norm_occ = next_obs[offset + 2]

            # Finite check: non-finite norm values → fall back to current state values
            if not np.isfinite(norm_loc):
                norm_loc = current_state.vehicles[v_idx].current_location / max(self.task.n_nodes - 1, 1)
            if not np.isfinite(norm_time):
                norm_time = current_state.vehicles[v_idx].current_time / self.task.max_time if self.task.max_time > 0 else 0.0
            if not np.isfinite(norm_occ):
                norm_occ = current_state.vehicles[v_idx].current_occupancy / max(self.task.vehicle_capacity, 1)

            # Clamp normalized values to [0, 1]
            norm_loc = max(0.0, min(float(norm_loc), 1.0))
            norm_time = max(0.0, min(float(norm_time), 1.0))
            norm_occ = max(0.0, min(float(norm_occ), 1.0))

            # Denormalize to actual values
            predicted_loc = int(norm_loc * (self.task.n_nodes - 1) + 0.5) if self.task.n_nodes > 1 else 0
            predicted_time = norm_time * self.task.max_time
            predicted_occ = int(norm_occ * self.task.vehicle_capacity + 0.5)

            # Clamp to valid ranges
            predicted_loc = max(0, min(predicted_loc, self.task.n_nodes - 1))
            predicted_time = max(0.0, min(predicted_time, self.task.max_time))
            predicted_occ = max(0, min(predicted_occ, self.task.vehicle_capacity))

            # Update vehicle state
            vehicle = predicted_state.vehicles[v_idx]
            vehicle.current_location = predicted_loc
            vehicle.current_time = predicted_time
            vehicle.current_occupancy = predicted_occ

        # Update current_request if within bounds
        if next_epoch < len(self.task.requests):
            predicted_state.current_request = self.task.requests[next_epoch]

        # PAPER-COMPLIANT: traffic_level is HIDDEN from agent
        # Agent doesn't observe or predict it (like Frozen Lake's intend_prob)
        # We keep it unchanged as agent's "belief" (assumes traffic stays constant)
        # BNN learns travel time dynamics WITHOUT knowing this hidden parameter
        # The learned vehicle times implicitly capture traffic effects
        predicted_state.traffic_level = current_state.traffic_level  # Agent's belief (unchanged)

        return predicted_state

    def is_terminal(self, bnn1, state):
        """Check if state is terminal (domain-aware)."""
        # Use task's built-in is_terminal method if available
        if hasattr(self.task, 'is_terminal'):
            return self.task.is_terminal(state)
        # Fallback: use reward-based heuristic for Frozen Lake
        reward = self.task.instant_reward_byindex(state)
        return reward == 1 or reward == -1

    def expand(self, BNN1, weight_set1, BNN2, weight_set2, threshold, training_started, isi):
        if self.is_terminal(BNN1, self.state):
            return self
        if self.is_decision_node():
            for action in self.get_possible_actions():
                # Create chance nodes for each possible action
                child_node = Node(self.state, self.time, self.task, self.danger, action=action, parent=self, node_type="chance", rng=self.np_random)
                self.children.append(child_node)
        else:  # For a chance node
            transition1, samples1, pessimistic1, epistemic_uncertainties1, aleatoric_uncertainties1, transition2, samples2, pessimistic2, epistemic_uncertainties2, aleatoric_uncertainties2, confidence_check = self.get_bnn_prediction(self.state, self.action, BNN1, weight_set1, BNN2, weight_set2, isi)
            
            # Store uncertainty values in the chance node for explainability
            # Use model 1 (base model) uncertainties as the primary values
            self.epistemic_uncertainty = float(epistemic_uncertainties1)
            self.aleatoric_uncertainty = float(aleatoric_uncertainties1)

            # NEW: Record decision rationale for explanation
            branch = None

            # DPAS (Dual-Phase Adaptive Sampling) - Algorithm 2, Line 52-61
            # BNN1 = M̂k (current updated model), BNN2 = M̂k-1 (previous model)
            # transition1 = p(·|s,a,M̂k), transition2 = p(·|s,a,M̂k-1)
            # pessimistic1 = pwc(·|s,a,M̂k-1)

            # Epistemic: per (s,a) is correct (Equation 9)
            delta_E = float(epistemic_uncertainties1) - float(epistemic_uncertainties2)
            # Non-finite δE → +inf (force pessimistic branch as conservative default)
            if not np.isfinite(delta_E):
                delta_E = float('inf')

            # PAPER-COMPLIANT FIX: δA = VarA(M̂k; S') - VarA(M̂k-1; S')
            # Paper Eq. 8: VarA is computed over a SUBSET S' of state-action pairs
            #
            # CRITICAL: Both VarA computations must use the SAME state S' with BOTH models
            # - VarA(M̂k; S'): Aleatoric uncertainty under current model
            # - VarA(M̂k-1; S'): Aleatoric uncertainty under previous episode's model
            # This correctly measures whether the ENVIRONMENT has become more stochastic
            task_name = type(self.task).__name__
            if task_name == 'NSParatransitV0' and self.parent:
                # Get all valid actions at parent state (this is S')
                valid_actions = self.task.get_valid_actions(self.parent.state)

                # VarA(M̂k; S'): Aleatoric using current model
                aleatoric_Mk = compute_aleatoric_over_actions(
                    self.parent.state, valid_actions, BNN1, weight_set1, self.task
                )

                # VarA(M̂k-1; S'): Aleatoric using previous episode's model (SAME state!)
                aleatoric_Mk_minus_1 = compute_aleatoric_over_actions(
                    self.parent.state, valid_actions, BNN2, weight_set2, self.task
                )

                # δA = VarA(M̂k) - VarA(M̂k-1) on SAME state S'
                delta_A = aleatoric_Mk - aleatoric_Mk_minus_1
                # Non-finite δA → +inf (force pessimistic branch as conservative default)
                if not np.isfinite(delta_A):
                    delta_A = float('inf')
            else:
                # For frozen lake, keep original per-(s,a) calculation
                delta_A = float(aleatoric_uncertainties1) - float(aleatoric_uncertainties2)
                if not np.isfinite(delta_A):
                    delta_A = float('inf')

            # DEBUG: Summarize DPAS decisions to avoid noisy per-step logs
            # Per-epoch tracking (reset at each epoch via MCTS._dpas_current_epoch)
            if not hasattr(Node, '_dpas_counts'):
                Node._dpas_counts = {"transition1": 0, "pessimistic1": 0}
                Node._dpas_log_interval = 5000
                Node._dpas_total = 0
                Node._dpas_delta_E_sum = 0.0  # Track average delta_E for diagnostics
                Node._dpas_delta_A_sum = 0.0  # Track average delta_A for diagnostics
                Node._dpas_epoch_counts = {"transition1": 0, "pessimistic1": 0}  # Per-epoch
                Node._dpas_epoch_total = 0
                Node._dpas_epoch_delta_E_sum = 0.0
                Node._dpas_epoch_delta_A_sum = 0.0

            # DPAS Logic (Paper Algorithm 2, Lines 52-61):
            # δE = VarE(M̂k; s,a) - VarE(M̂k-1; s,a) (change in epistemic uncertainty)
            # δA = VarA(M̂k; s,a) - VarA(M̂k-1; s,a) (change in aleatoric uncertainty)
            # If δE ≤ εE and δA ≤ εA: uncertainty didn't increase much → use regular sampling
            # Otherwise: uncertainty increased significantly → use pessimistic sampling (risk-averse)

            # PAPER-COMPLIANT FIX (Issue 4): For paratransit, sample directly from BNN outputs
            # instead of using categorical_sample on decision_epoch distribution
            if task_name == 'NSParatransitV0':
                # Paratransit: Direct BNN sample selection (bypasses categorical_sample)
                if delta_E <= MCTS.epsilon_E and delta_A <= MCTS.epsilon_A:
                    # Regular sampling from M̂k: randomly select a BNN sample
                    sampled_next_obs = samples1[self.np_random.randint(len(samples1)), 0, :]
                    branch = "transition1"
                else:
                    # Paper Eq.(6): pwc over this chance node's tree children
                    worst_idx = self._choose_worst_next_state_sample(
                        samples2, self.state, self.action, BNN2, weight_set2,
                        tree_children=self.children)
                    sampled_next_obs = samples2[worst_idx, 0, :]
                    branch = "pessimistic1"

                # Decode BNN output to next decision_epoch (for child_state index)
                # Since decision_epoch is deterministic (+1), we can compute it directly
                if self.parent and hasattr(self.parent.state, 'decision_epoch'):
                    child_state = self.parent.state.decision_epoch + 1
                else:
                    child_state = 0  # Fallback
            else:
                # Frozen Lake: Keep original categorical_sample logic
                if delta_E <= MCTS.epsilon_E and delta_A <= MCTS.epsilon_A:
                    # Uncertainties stable or decreased → trust M̂k, use regular sampling
                    child_state = self.categorical_sample(transition1, self.np_random)
                    branch = "transition1"
                else:
                    # Uncertainties increased → model uncertain, use M̂k-1 pessimistic sampling
                    child_state = self.categorical_sample(pessimistic1, self.np_random)
                    branch = "pessimistic1"
                sampled_next_obs = None  # Not used for Frozen Lake

            # Update cumulative stats
            Node._dpas_counts[branch] += 1
            Node._dpas_total += 1
            # Only accumulate finite deltas into stats; track invalid separately
            if np.isfinite(delta_E) and np.isfinite(delta_A):
                Node._dpas_delta_E_sum += delta_E
                Node._dpas_delta_A_sum += delta_A
            else:
                Node._dpas_invalid_count = getattr(Node, '_dpas_invalid_count', 0) + 1

            # Update per-epoch stats
            Node._dpas_epoch_counts[branch] += 1
            Node._dpas_epoch_total += 1
            if np.isfinite(delta_E) and np.isfinite(delta_A):
                Node._dpas_epoch_delta_E_sum += delta_E
                Node._dpas_epoch_delta_A_sum += delta_A
                Node._dpas_epoch_finite_count = getattr(Node, '_dpas_epoch_finite_count', 0) + 1
            else:
                Node._dpas_epoch_invalid_count = getattr(Node, '_dpas_epoch_invalid_count', 0) + 1
            # Track EU/AU for epoch-level logging (item 9)
            Node._dpas_epoch_eu_sum = getattr(Node, '_dpas_epoch_eu_sum', 0.0) + float(epistemic_uncertainties1)
            Node._dpas_epoch_au_sum = getattr(Node, '_dpas_epoch_au_sum', 0.0) + float(aleatoric_uncertainties1)

            # Store explanation metadata on the chance node
            # PAPER-COMPLIANT: Field names match M̂k / M̂_{k-1} convention
            # BNN1 = M̂k (current updated model), BNN2 = M̂_{k-1} (previous model)
            self.explain = {
                "branch": branch,
                "eu_mk": float(epistemic_uncertainties1),          # VarE(M̂k; s,a)
                "eu_mk_minus_1": float(epistemic_uncertainties2),  # VarE(M̂_{k-1}; s,a)
                "ale_mk": float(aleatoric_uncertainties1),         # VarA(M̂k; s,a)
                "ale_mk_minus_1": float(aleatoric_uncertainties2), # VarA(M̂_{k-1}; s,a)
                "action": self.action,
                "time": self.time,
                "delta_E": delta_E,
                "delta_A": delta_A
            }

            # MODEL DRIFT DETECTION: Store BNN-predicted vehicle times from BOTH models
            # This enables comparison of MDP_t vs MDP_{t-n} understanding of travel times
            # samples1 = BNN1 (M̂k, current model) predictions, shape (100, 1, 16)
            # samples2 = BNN2 (M̂k-1, previous model) predictions, shape (100, 1, 16)
            if task_name == 'NSParatransitV0' and samples1 is not None and samples2 is not None:
                # Extract mean predicted observation from each model
                mean_pred_bnn1 = np.mean(samples1[:, 0, :], axis=0)  # MDP_t model prediction
                mean_pred_bnn2 = np.mean(samples2[:, 0, :], axis=0)  # MDP_{t-n} model prediction

                # Decode predicted vehicle times (for ETA computation)
                # Observation format: [norm_epoch, v0_loc, v0_time, v0_occ, v1_loc, ...]
                n_vehicles = self.task.n_vehicles if hasattr(self.task, 'n_vehicles') else 5
                max_time = self.task.max_time if hasattr(self.task, 'max_time') else 480.0
                n_nodes = self.task.n_nodes if hasattr(self.task, 'n_nodes') else 10788

                # Decode vehicle states from BNN1 (MDP_t / current model)
                bnn1_vehicle_states = {}
                for v_idx in range(n_vehicles):
                    offset = 1 + v_idx * 3
                    norm_loc = mean_pred_bnn1[offset + 0]
                    norm_time = mean_pred_bnn1[offset + 1]
                    norm_occ = mean_pred_bnn1[offset + 2]
                    # Finite check + [0,1] clamp for explain consistency
                    norm_loc = max(0.0, min(float(norm_loc), 1.0)) if np.isfinite(norm_loc) else 0.0
                    norm_time = max(0.0, min(float(norm_time), 1.0)) if np.isfinite(norm_time) else 0.0
                    norm_occ = max(0.0, min(float(norm_occ), 1.0)) if np.isfinite(norm_occ) else 0.0
                    bnn1_vehicle_states[f"V{v_idx}"] = {
                        "predicted_location": max(0, min(int(norm_loc * (n_nodes - 1) + 0.5), n_nodes - 1)),
                        "predicted_time": max(0.0, min(float(norm_time * max_time), max_time)),
                        "predicted_occupancy": max(0, min(int(norm_occ * 3 + 0.5), 3))  # capacity=3
                    }

                # Decode vehicle states from BNN2 (MDP_{t-n} / previous model)
                bnn2_vehicle_states = {}
                for v_idx in range(n_vehicles):
                    offset = 1 + v_idx * 3
                    norm_loc = mean_pred_bnn2[offset + 0]
                    norm_time = mean_pred_bnn2[offset + 1]
                    norm_occ = mean_pred_bnn2[offset + 2]
                    # Finite check + [0,1] clamp for explain consistency
                    norm_loc = max(0.0, min(float(norm_loc), 1.0)) if np.isfinite(norm_loc) else 0.0
                    norm_time = max(0.0, min(float(norm_time), 1.0)) if np.isfinite(norm_time) else 0.0
                    norm_occ = max(0.0, min(float(norm_occ), 1.0)) if np.isfinite(norm_occ) else 0.0
                    bnn2_vehicle_states[f"V{v_idx}"] = {
                        "predicted_location": max(0, min(int(norm_loc * (n_nodes - 1) + 0.5), n_nodes - 1)),
                        "predicted_time": max(0.0, min(float(norm_time * max_time), max_time)),
                        "predicted_occupancy": max(0, min(int(norm_occ * 3 + 0.5), 3))
                    }

                # Store in explain for model drift ETA comparison
                self.explain["bnn_predicted_states"] = {
                    "mdp_t": bnn1_vehicle_states,      # Current model's prediction
                    "mdp_t_minus_n": bnn2_vehicle_states  # Previous model's prediction
                }

            # PAPER-COMPLIANT: Use BNN-predicted next state for ALL domains
            # BNN learns p(s'|s,a) from observed transitions, without knowing hidden parameters
            if task_name == 'NSParatransitV0' and sampled_next_obs is not None:
                # PAPER-COMPLIANT: Reconstruct next ParatransitState from BNN-predicted observation
                # sampled_next_obs was already selected above via DPAS logic
                if self.parent and hasattr(self.parent.state, 'decision_epoch'):
                    # Decode BNN prediction to ParatransitState
                    predicted_state = self._decode_bnn_output_to_state(
                        sampled_next_obs, self.parent.state, self.action, child_state
                    )
                    child_state = predicted_state
            # else: For Frozen Lake, child_state is already an integer index (correct)

            existing_child = self.get_child_with_state(child_state)
            if existing_child:
                child_node = existing_child
            else:
                child_node = Node(child_state, self.time, self.task, self.danger, parent=self, node_type="decision", rng=self.np_random)
                # Inherit uncertainty from parent chance node
                child_node.epistemic_uncertainty = self.epistemic_uncertainty
                child_node.aleatoric_uncertainty = self.aleatoric_uncertainty

                # NEW: Compute and store transition_ap for this (parent_state, action, child_state) transition
                # This enables building complete traces with prefix path from root
                if task_name == 'NSParatransitV0' and hasattr(self.task, 'evaluate_atomic_props'):
                    # Get parent's state (this chance node's parent is a decision node)
                    if self.parent and hasattr(self.parent, 'state'):
                        parent_state = self.parent.state
                        transition_ap = self.task.evaluate_atomic_props(parent_state, self.action, child_state)
                        child_node.transition_ap = transition_ap

                self.children.append(child_node)
        return child_node

    def categorical_sample(self, prob_n, np_random):
        """
        Sample from categorical distribution
        Each row specifies class probabilities
        """
        prob_n = np.asarray(prob_n)
        csprob_n = np.cumsum(prob_n)
        return (csprob_n > np_random.rand()).argmax()

    """ Simulate until a terminal state """

    def rollout(self, BNN1, weight_set1, BNN2, weight_set2, threshold, training_started, isi):
        # BUG FIX: Use BNN-based rollout for ALL domains (model-based MCTS as per paper)
        # The original paratransit-specific code used real environment, which is incorrect
        task_name = type(self.task).__name__
        current_state = self.state
        cumulative_reward = 0.0
        depth = 0
        done = False
        visited_pairs = set()
        reached_goal = False
        reached_hole = False
        steps_taken = 0

        # Trace-based PCTL evaluation: record AP truth values at each step
        # trace = [{ap_name: bool}, {ap_name: bool}, ...] (one dict per step)
        trace = []

        # Paratransit-specific event tracking (for PCTL reach counts)
        # Path-level OR accumulation: once True, stays True for entire rollout
        events_seen = {
            # Core completion/violation
            'service_complete': False,
            'violation': False,
            'capacity_violation': False,
            'time_window_violation': False,
            # Delay events
            'pickup_delay': False,
            'dropoff_delay': False,
            'any_delay': False,
            # Vehicle state events
            'carpool_active': False,
            'capacity_full': False,
            'any_vehicle_idle': False,
            'all_vehicles_busy': False
        }
        # Track if path is "safe" (no violations so far) for safe_complete
        path_is_safe = True

        # DEBUG: Add safety check for infinite loops
        max_iterations = 100

        while not done and depth < max_iterations:
            #print("current state:", current_state)
            if self.is_terminal(BNN1, current_state):
                reward = self.task.instant_reward_byindex(current_state)
                cumulative_reward += reward
                if reward == 1.0:
                    reached_goal = True
                elif reward == -1.0:
                    reached_hole = True

                # FIX: Record terminal state's APs in trace (otherwise trace would be empty!)
                # This ensures even terminal-starting rollouts produce a valid trace for PCTL
                if task_name == 'NSParatransitV0' and hasattr(self.task, 'evaluate_terminal_props'):
                    # For terminal state, create a "final step" AP record
                    # We need to evaluate APs for the terminal state itself
                    terminal_props = self.task.evaluate_terminal_props(current_state)
                    if terminal_props:
                        trace.append(terminal_props)
                        # Also update events_seen for path-level tracking
                        for k, v in terminal_props.items():
                            if k in events_seen and v:
                                events_seen[k] = True
                        if terminal_props.get('violation', False):
                            path_is_safe = False
                break

            # BUG FIX: Use dynamic action space for paratransit (state-dependent A(s))
            if task_name == 'NSParatransitV0' and hasattr(current_state, 'decision_epoch'):
                possible_actions = self.task.get_valid_actions(current_state)
            else:
                # For Frozen Lake, use static action space
                possible_actions = self.possible_actions

            action = self.np_random.choice(possible_actions)

            transition1, samples1, pessimistic1, epistemic_uncertainties1, aleatoric_uncertainties1, transition2, samples2, pessimistic2, epistemic_uncertainties2,aleatoric_uncertainties2, confidence_check = self.get_bnn_prediction(
                current_state, action, BNN1, weight_set1, BNN2, weight_set2, isi)

            # DPAS (same logic as in expand)
            delta_E = float(epistemic_uncertainties1) - float(epistemic_uncertainties2)
            # Non-finite δE → +inf (force pessimistic branch as conservative default)
            if not np.isfinite(delta_E):
                delta_E = float('inf')

            # PAPER-COMPLIANT FIX: δA = VarA(M̂k; S') - VarA(M̂k-1; S')
            # Both computations use SAME state with BOTH models
            if task_name == 'NSParatransitV0' and hasattr(current_state, 'decision_epoch'):
                # VarA(M̂k; S'): Aleatoric using current model
                aleatoric_Mk = compute_aleatoric_over_actions(
                    current_state, possible_actions, BNN1, weight_set1, self.task
                )
                # VarA(M̂k-1; S'): Aleatoric using previous episode's model (SAME state!)
                aleatoric_Mk_minus_1 = compute_aleatoric_over_actions(
                    current_state, possible_actions, BNN2, weight_set2, self.task
                )
                # δA = VarA(M̂k) - VarA(M̂k-1) on SAME state S'
                delta_A = aleatoric_Mk - aleatoric_Mk_minus_1
                # Non-finite δA → +inf (force pessimistic branch as conservative default)
                if not np.isfinite(delta_A):
                    delta_A = float('inf')
            else:
                # For frozen lake, keep original per-(s,a) calculation
                delta_A = float(aleatoric_uncertainties1) - float(aleatoric_uncertainties2)
                if not np.isfinite(delta_A):
                    delta_A = float('inf')

            # PAPER-COMPLIANT FIX (Issue 4): For paratransit, sample directly from BNN outputs
            if task_name == 'NSParatransitV0' and hasattr(current_state, 'decision_epoch'):
                # Paratransit: Direct BNN sample selection (same logic as expand())
                if delta_E <= MCTS.epsilon_E and delta_A <= MCTS.epsilon_A:
                    # Regular sampling from M̂k
                    sampled_next_obs = samples1[self.np_random.randint(len(samples1)), 0, :]
                    rollout_branch = "transition1"
                else:
                    # Paper Eq.(6): rollout has no tree for current_state → 1-step fallback
                    worst_idx = self._choose_worst_next_state_sample(
                        samples2, current_state, action, BNN2, weight_set2,
                        tree_children=[])  # no tree children during rollout
                    sampled_next_obs = samples2[worst_idx, 0, :]
                    rollout_branch = "pessimistic1"

                # Compute deterministic next decision_epoch
                child_state_idx = current_state.decision_epoch + 1

                # Decode BNN prediction to ParatransitState
                child_state = self._decode_bnn_output_to_state(
                    sampled_next_obs, current_state, action, child_state_idx
                )
            else:
                # Frozen Lake: Keep original categorical_sample logic
                if delta_E <= MCTS.epsilon_E and delta_A <= MCTS.epsilon_A:
                    child_state_idx = self.categorical_sample(transition1, self.np_random)
                    rollout_branch = "transition1"
                else:
                    child_state_idx = self.categorical_sample(pessimistic1, self.np_random)
                    rollout_branch = "pessimistic1"
                child_state = child_state_idx

            # PAPER-COMPLIANT: Track DPAS stats in rollout (Issue 7)
            # Rollout DPAS decisions were previously untracked
            if not hasattr(Node, '_dpas_rollout_counts'):
                Node._dpas_rollout_counts = {"transition1": 0, "pessimistic1": 0}
                Node._dpas_rollout_total = 0
                Node._dpas_rollout_delta_E_sum = 0.0
                Node._dpas_rollout_delta_A_sum = 0.0
                Node._dpas_rollout_epoch_counts = {"transition1": 0, "pessimistic1": 0}
                Node._dpas_rollout_epoch_total = 0
                Node._dpas_rollout_epoch_delta_E_sum = 0.0
                Node._dpas_rollout_epoch_delta_A_sum = 0.0
            Node._dpas_rollout_counts[rollout_branch] += 1
            Node._dpas_rollout_total += 1
            # Only accumulate finite deltas into rollout stats
            if np.isfinite(delta_E) and np.isfinite(delta_A):
                Node._dpas_rollout_delta_E_sum += delta_E
                Node._dpas_rollout_delta_A_sum += delta_A
            else:
                Node._dpas_rollout_invalid_count = getattr(Node, '_dpas_rollout_invalid_count', 0) + 1
            Node._dpas_rollout_epoch_counts[rollout_branch] += 1
            Node._dpas_rollout_epoch_total += 1
            if np.isfinite(delta_E) and np.isfinite(delta_A):
                Node._dpas_rollout_epoch_delta_E_sum += delta_E
                Node._dpas_rollout_epoch_delta_A_sum += delta_A
                Node._dpas_rollout_epoch_finite_count = getattr(Node, '_dpas_rollout_epoch_finite_count', 0) + 1
            else:
                Node._dpas_rollout_epoch_invalid_count = getattr(Node, '_dpas_rollout_epoch_invalid_count', 0) + 1
            # Track EU/AU for epoch-level logging (rollout)
            Node._dpas_rollout_epoch_eu_sum = getattr(Node, '_dpas_rollout_epoch_eu_sum', 0.0) + float(epistemic_uncertainties1)
            Node._dpas_rollout_epoch_au_sum = getattr(Node, '_dpas_rollout_epoch_au_sum', 0.0) + float(aleatoric_uncertainties1)

            # Option D Step 3: Use (s,a,s') based reward for paratransit
            if task_name == 'NSParatransitV0' and hasattr(current_state, 'decision_epoch'):
                # Compute reward based on (state, action, next_state) for accurate timing/capacity
                reward = self.task.compute_rollout_reward(current_state, action, child_state)

                # Trace-based PCTL: evaluate atomic propositions and record in trace
                if hasattr(self.task, 'evaluate_atomic_props'):
                    step_props = self.task.evaluate_atomic_props(
                        current_state, action, child_state
                    )
                    # Record this step's AP values in the trace
                    trace.append(step_props)

                    # Also OR-accumulate into events_seen for legacy reach counting
                    for k, v in step_props.items():
                        if k in events_seen:
                            events_seen[k] |= v  # OR accumulation
                    # Track if path becomes unsafe (for safe_complete)
                    if step_props.get('violation', False):
                        path_is_safe = False
            else:
                # For Frozen Lake, use simple state-based reward
                reward = self.task.instant_reward_byindex(child_state)

            cumulative_reward += pow(self.discount_factor, depth) * reward

            # Check termination conditions (domain-agnostic)
            # 1. Terminal state reached (via is_terminal check)
            if self.is_terminal(BNN1, child_state):
                done = True
                # For frozen lake compatibility: check if goal or hole
                if reward == 1.0:
                    reached_goal = True
                elif reward == -1.0:
                    reached_hole = True
            # 2. Maximum depth reached (prevents infinite loops)
            elif depth >= 100:
                done = True

            depth += 1
            steps_taken += 1
            current_state = child_state

        # Compute safe_complete: service_complete AND path stayed safe (no violations)
        events_seen['safe_complete'] = events_seen['service_complete'] and path_is_safe

        # Return dict with rollout statistics for PCTL evaluation
        result_dict = {
            'reward': cumulative_reward,
            'reached_goal': reached_goal,
            'reached_hole': reached_hole,
            'reached_terminal': reached_goal or reached_hole,
            'steps': steps_taken,
            'trace': trace,  # Trace-based PCTL: list of {ap_name: bool} per step
        }
        # Add paratransit events (path-level: True if event occurred anywhere in rollout)
        result_dict.update(events_seen)
        # DEBUG: Write to file for debugging
        if not hasattr(Node, '_rollout_debug_count'):
            Node._rollout_debug_count = 0
        if Node._rollout_debug_count < 20:
            with open('/tmp/rollout_debug.txt', 'a') as f:
                f.write(f"Rollout #{Node._rollout_debug_count}: reached_goal={reached_goal}, reached_hole={reached_hole}, steps={steps_taken}, trace_len={len(trace)}, reward={cumulative_reward:.4f}, task={task_name}\n")
            Node._rollout_debug_count += 1
        return result_dict

    def backpropagate(self, result, is_rollout_origin=True):
        """
        Backpropagate simulation result up the tree.

        Args:
            result: Can be either:
                - float: legacy cumulative reward (for backwards compatibility)
                - dict: rollout statistics including 'reward', 'reached_goal', 'reached_hole', 'steps', 'trace'
            is_rollout_origin: True only for the node where rollout started (leaf node).
                               All nodes now store traces with accumulated prefix.

        NEW DESIGN for trace prefix handling:
        - Trace is passed up with accumulated prefix
        - Each decision node prepends its transition_ap before passing to parent
        - When storing, each node stores the trace AS RECEIVED (already has correct prefix)
        - This ensures every node has traces representing paths from THAT node onwards
        """
        if isinstance(result, dict):
            # New format: extract statistics for PCTL evaluation
            reward = result['reward']
            self.visits += 1
            self.value += reward

            # Store the trace as received (already has accumulated prefix from descendants)
            trace = result.get('trace', None)
            if trace is not None and len(trace) > 0:
                self.rollout_traces.append(list(trace))  # Copy to avoid mutation

            # Continue backpropagation to parent
            # For decision nodes with transition_ap: prepend our transition_ap to the trace
            # This builds up the complete prefix path as we go up the tree
            if self.parent:
                result = dict(result)  # Copy to avoid mutation
                if self.is_decision_node():
                    result['reward'] = reward * self.discount_factor
                    # Prepend this node's transition_ap to build complete trace from parent's perspective
                    if self.transition_ap is not None and result.get('trace'):
                        result['trace'] = [self.transition_ap] + list(result['trace'])
                self.parent.backpropagate(result, is_rollout_origin=False)
        else:
            # Legacy format: just a reward value
            reward = result
            self.visits += 1
            self.value += reward
            if self.parent:
                if self.is_decision_node:
                    reward = reward * self.discount_factor
                self.parent.backpropagate(reward)


    def uct_value(self, parent_visits, exploration_constant=math.sqrt(2)):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + exploration_constant * math.sqrt(math.log(parent_visits) / self.visits)

    def best_child(self, exploration_constant=math.sqrt(2)):
        return max(self.children, key=lambda c: c.uct_value(self.visits, exploration_constant))

    def __encode_action(self, action):
        # Determine number of actions from task
        if self.task is not None:
            if hasattr(self.task, 'num_actions'):
                n_actions = self.task.num_actions
            elif hasattr(self.task, 'nA'):
                n_actions = self.task.nA
            else:
                n_actions = 4  # Default to frozen lake
        else:
            n_actions = 4  # Default to frozen lake

        a = np.array([0] * n_actions)
        a[action] = 1
        return a

class MCTS:
    # DPAS thresholds for uncertainty-based sampling strategy (Algorithm 2, Line 56)
    # Paper "Act as You Learn" Section 4 (Hyper-parameters):
    # "we set εE to 0.02 and εA to 0 for all the ADA-MCTS experiments"
    #
    # δE = VarE(M̂k) - VarE(M̂k-1): positive if new model is MORE uncertain
    # δA = VarA(M̂k) - VarA(M̂k-1): positive if environment seems MORE stochastic
    # If δE ≤ εE AND δA ≤ εA → regular sampling (reward maximizing)
    # Otherwise → pessimistic sampling (risk averse)
    #
    # εA = 0 means: ANY positive increase in aleatoric uncertainty → pessimistic
    # This is strict by design — after environment change, even small δA > 0 triggers caution
    epsilon_E = -0.002
    epsilon_A = 0.02

    # PAPER-COMPLIANT (Issue 8): Uncertainty aggregation method across output dimensions
    # Paper Eq.(8)/(9): uncertainties are per-output-dimension vectors.
    # Aggregation to scalar for DPAS comparison. Options: 'mean', 'sum', 'max'
    # Default 'mean' matches original behavior; 'sum'/'max' are more conservative
    # (less likely to smooth out local spikes, improving DPAS sensitivity).
    uncertainty_aggregation = 'mean'

    # Class-level tracking for uncertainty values (for XAI)
    previous_epistemic_uncertainty = None
    previous_aleatoric_uncertainty = None

    @staticmethod
    def reset_epoch_dpas_stats():
        """Reset per-epoch DPAS statistics. Call at start of each decision epoch."""
        # Expand stats
        Node._dpas_epoch_counts = {"transition1": 0, "pessimistic1": 0}
        Node._dpas_epoch_total = 0
        Node._dpas_epoch_delta_E_sum = 0.0
        Node._dpas_epoch_delta_A_sum = 0.0
        Node._dpas_epoch_finite_count = 0
        Node._dpas_epoch_invalid_count = 0
        Node._dpas_epoch_eu_sum = 0.0
        Node._dpas_epoch_au_sum = 0.0
        # Rollout stats (Issue 7: previously untracked)
        Node._dpas_rollout_epoch_counts = {"transition1": 0, "pessimistic1": 0}
        Node._dpas_rollout_epoch_total = 0
        Node._dpas_rollout_epoch_delta_E_sum = 0.0
        Node._dpas_rollout_epoch_delta_A_sum = 0.0
        Node._dpas_rollout_epoch_finite_count = 0
        Node._dpas_rollout_epoch_invalid_count = 0
        Node._dpas_rollout_epoch_eu_sum = 0.0
        Node._dpas_rollout_epoch_au_sum = 0.0

    @staticmethod
    def clear_bnn_cache():
        """Clear BNN prediction cache. Call after model weights are updated."""
        Node.bnn_cache = {}
        Node.bnn_cache_hits = 0
        Node.bnn_cache_misses = 0
        Node.aleatoric_cache = {}

    @staticmethod
    def get_epoch_dpas_stats():
        """Get per-epoch DPAS statistics for logging.

        PAPER-COMPLIANT (Issue 7): Returns separate expand and rollout stats.
        Previously only expand was tracked, missing rollout DPAS decisions.
        """
        expand_total = getattr(Node, '_dpas_epoch_total', 0)
        rollout_total = getattr(Node, '_dpas_rollout_epoch_total', 0)

        if expand_total == 0 and rollout_total == 0:
            return None

        # Expand stats (use finite_count for delta averages)
        e_reg = Node._dpas_epoch_counts.get("transition1", 0) if expand_total > 0 else 0
        e_pess = Node._dpas_epoch_counts.get("pessimistic1", 0) if expand_total > 0 else 0
        e_finite = getattr(Node, '_dpas_epoch_finite_count', 0)
        e_avg_dE = Node._dpas_epoch_delta_E_sum / e_finite if e_finite > 0 else 0.0
        e_avg_dA = Node._dpas_epoch_delta_A_sum / e_finite if e_finite > 0 else 0.0

        # Rollout stats (Issue 7)
        r_reg = Node._dpas_rollout_epoch_counts.get("transition1", 0) if rollout_total > 0 else 0
        r_pess = Node._dpas_rollout_epoch_counts.get("pessimistic1", 0) if rollout_total > 0 else 0
        r_finite = getattr(Node, '_dpas_rollout_epoch_finite_count', 0)
        r_avg_dE = Node._dpas_rollout_epoch_delta_E_sum / r_finite if r_finite > 0 else 0.0
        r_avg_dA = Node._dpas_rollout_epoch_delta_A_sum / r_finite if r_finite > 0 else 0.0

        # Combined (backwards-compatible: 'total'/'regular'/'pessimistic' now include both)
        combined_total = expand_total + rollout_total
        combined_reg = e_reg + r_reg
        combined_pess = e_pess + r_pess
        combined_finite = e_finite + r_finite
        combined_invalid = getattr(Node, '_dpas_epoch_invalid_count', 0) + getattr(Node, '_dpas_rollout_epoch_invalid_count', 0)

        # Epoch-level EU/AU averages (for log output)
        combined_eu_sum = getattr(Node, '_dpas_epoch_eu_sum', 0.0) + getattr(Node, '_dpas_rollout_epoch_eu_sum', 0.0)
        combined_au_sum = getattr(Node, '_dpas_epoch_au_sum', 0.0) + getattr(Node, '_dpas_rollout_epoch_au_sum', 0.0)

        return {
            # Combined (backwards-compatible)
            'total': combined_total,
            'regular': combined_reg,
            'pessimistic': combined_pess,
            'regular_pct': combined_reg / combined_total if combined_total > 0 else 0.0,
            'pessimistic_pct': combined_pess / combined_total if combined_total > 0 else 0.0,
            'avg_delta_E': (Node._dpas_epoch_delta_E_sum + Node._dpas_rollout_epoch_delta_E_sum) / combined_finite if combined_finite > 0 else 0.0,
            'avg_delta_A': (Node._dpas_epoch_delta_A_sum + Node._dpas_rollout_epoch_delta_A_sum) / combined_finite if combined_finite > 0 else 0.0,
            'invalid_count': combined_invalid,
            # Epoch-level EU/AU (current model's uncertainty, averaged over all DPAS calls)
            'epoch_eu': combined_eu_sum / combined_total if combined_total > 0 else 0.0,
            'epoch_au': combined_au_sum / combined_total if combined_total > 0 else 0.0,
            # Separate expand/rollout stats (Issue 7)
            'expand_dpas': {
                'total': expand_total,
                'regular': e_reg,
                'pessimistic': e_pess,
                'avg_delta_E': e_avg_dE,
                'avg_delta_A': e_avg_dA,
            },
            'rollout_dpas': {
                'total': rollout_total,
                'regular': r_reg,
                'pessimistic': r_pess,
                'avg_delta_E': r_avg_dE,
                'avg_delta_A': r_avg_dA,
            },
        }

    def compute_visit_weighted_dpas(self):
        """Compute visit-weighted δE/δA from the MCTS tree after search.

        Returns two sets of metrics:
        1. Root-Weighted: Only root's chance children (one per action).
           avgδE_root = Σ v_i·δE_i / Σ v_i   for i ∈ C_root
        2. All-Chance: Every chance node in the tree.
           avgδE_all  = Σ v_i·δE_i / Σ v_i   for i ∈ C_all

        These give a more meaningful picture than the flat per-call averages,
        because high-visit branches (MCTS's preferred actions) get higher weight.
        """
        root_wdE, root_wdA, root_wsum = 0.0, 0.0, 0.0
        all_wdE, all_wdA, all_wsum = 0.0, 0.0, 0.0

        # --- Root-level chance children ---
        for child in self.root.children:
            if child.is_chance_node() and child.explain and child.visits > 0:
                v = child.visits
                dE = child.explain.get('delta_E', 0.0)
                dA = child.explain.get('delta_A', 0.0)
                if np.isfinite(dE) and np.isfinite(dA):
                    root_wdE += v * dE
                    root_wdA += v * dA
                    root_wsum += v

        # --- All chance nodes (BFS) ---
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node.is_chance_node() and node.explain and node.visits > 0:
                v = node.visits
                dE = node.explain.get('delta_E', 0.0)
                dA = node.explain.get('delta_A', 0.0)
                if np.isfinite(dE) and np.isfinite(dA):
                    all_wdE += v * dE
                    all_wdA += v * dA
                    all_wsum += v
            queue.extend(node.children)

        return {
            'root_weighted_dE': root_wdE / root_wsum if root_wsum > 0 else 0.0,
            'root_weighted_dA': root_wdA / root_wsum if root_wsum > 0 else 0.0,
            'root_total_visits': int(root_wsum),
            'all_weighted_dE': all_wdE / all_wsum if all_wsum > 0 else 0.0,
            'all_weighted_dA': all_wdA / all_wsum if all_wsum > 0 else 0.0,
            'all_total_visits': int(all_wsum),
        }

    @staticmethod
    def store_epoch_uncertainties(epistemic, aleatoric):
        """Store current epoch's uncertainties for next epoch's comparison."""
        MCTS.previous_epistemic_uncertainty = epistemic
        MCTS.previous_aleatoric_uncertainty = aleatoric

    def compute_root_aleatoric(self):
        """Compute aleatoric uncertainty at root state over all valid actions."""
        task_name = type(self.task).__name__
        if task_name != 'NSParatransitV0':
            return None

        state = self.root.state
        valid_actions = self.task.get_valid_actions(state)
        return compute_aleatoric_over_actions(
            state, valid_actions, self.BNN1, self.weight_set1, self.task
        )

    def __init__(self, initial_state_coordinate, initial_state_index, bnn1, bnn2, ws1, ws2, time, task, threshold, training_started, danger, seed=None):
        # Create shared RNG for reproducibility
        self.rng = np.random.RandomState(seed)
        self.root = Node(initial_state_index, time, task, danger, rng=self.rng)
        self.task = task
        self.BNN1 = bnn1
        self.BNN2 = bnn2
        self.weight_set1 = ws1
        self.weight_set2 = ws2
        self.time = time
        self.threshold = threshold
        self.training_started = training_started
        self.isi = initial_state_index
        Node.bnn_cache = {}  # Clear BNN cache for this MCTS run
        Node.aleatoric_cache = {}  # Clear aleatoric cache for this MCTS run

    def is_terminal(self, state):
        """Check if state is terminal (domain-aware)."""
        # Use task's built-in is_terminal method if available
        if hasattr(self.task, 'is_terminal'):
            return self.task.is_terminal(state)
        # Fallback: use reward-based heuristic for Frozen Lake
        reward = self.task.instant_reward_byindex(state)
        return reward == 1 or reward == -1

    def search(self, iterations):
        # Print progress for paratransit (long-running)
        task_name = type(self.task).__name__ if self.task else None
        show_progress = (task_name == 'NSParatransitV0' and iterations > 100)
        progress_interval = 500  # Print every 1000 iterations

        # Store reference to this MCTS instance in task for uncertainty tracking
        self.task._current_mcts = self

        # DEBUG: Print before first iteration
        if show_progress:
            print(f"        MCTS: Starting {iterations} iterations...", flush=True)

        for i in range(iterations):
            if show_progress and (i + 1) % progress_interval == 0:
                print(f"        MCTS: {i+1}/{iterations} iterations...", end='\r', flush=True)

            leaf = self.traverse(self.root)  # Traverse till you reach a leaf
            expanded_node = leaf.expand(self.BNN1, self.weight_set1, self.BNN2, self.weight_set2, self.threshold, self.training_started, self.isi)
            if expanded_node.is_chance_node():
                # If it's a chance node, expand again to get a decision node for rollout
                expanded_node = expanded_node.expand(self.BNN1, self.weight_set1, self.BNN2, self.weight_set2, self.threshold, self.training_started, self.isi)
            result = expanded_node.rollout(self.BNN1, self.weight_set1, self.BNN2, self.weight_set2, self.threshold, self.training_started, self.isi)
            expanded_node.backpropagate(result)

        if show_progress:
            print()  # New line after progress updates
            # DEBUG: Print cache statistics
            total_requests = Node.bnn_cache_hits + Node.bnn_cache_misses
            if total_requests > 0:
                hit_rate = Node.bnn_cache_hits / total_requests
                print(f"        BNN Cache: {Node.bnn_cache_hits} hits / {total_requests} total ({hit_rate:.1%} hit rate)")

    def traverse(self, node):
        while node.children:
            if node.is_decision_node():
                if self.training_started:
                    node = node.best_child(math.sqrt(2))
                else:
                    node = node.best_child(math.sqrt(2))
            elif self.is_terminal(node.state):
                return node
            else: # chance node
                node = node.expand(self.BNN1, self.weight_set1, self.BNN2, self.weight_set2, self.threshold, self.training_started, self.isi)
        return node

    def root_policy(self):
        """PAPER-COMPLIANT (Issue 6): Return visit-based policy distribution π(a|s0).

        Paper Algorithm 2, Line 12: π(a|s0) ← N(ν^{s0,a}) / N(s0)
        Returns dict {action: probability} based on visit counts.
        Execution can still take argmax, but the full distribution is preserved.
        """
        if not self.root.children:
            return {}
        total_visits = sum(c.visits for c in self.root.children)
        if total_visits == 0:
            # Uniform if no visits
            n_children = len(self.root.children)
            return {c.action: 1.0 / n_children for c in self.root.children}
        return {c.action: c.visits / total_visits for c in self.root.children}

    def best_action(self):
        """Return the best action (argmax of visit-based policy).

        Also computes and stores the full policy distribution π(a|s0)
        in self.last_policy for downstream analysis.
        """
        # Compute and store full policy distribution (Issue 6)
        self.last_policy = self.root_policy()

        # Debug: print action statistics
        if self.root.children:
            print(f"\n    Action Statistics (after {sum(c.visits for c in self.root.children)} total visits):")
            for child in sorted(self.root.children, key=lambda c: c.action):
                avg_value = child.value / child.visits if child.visits > 0 else 0
                pi = self.last_policy.get(child.action, 0.0)
                print(f"      Choice {child.action}: visits={child.visits:4d}, value={child.value:8.2f}, avg={avg_value:6.2f}, π={pi:.4f}")
        return max(self.root.children, key=lambda c: c.visits).action

    def __encode_action(self, action):
        # Determine number of actions from task
        if self.task is not None:
            if hasattr(self.task, 'num_actions'):
                n_actions = self.task.num_actions
            elif hasattr(self.task, 'nA'):
                n_actions = self.task.nA
            else:
                n_actions = 4  # Default to frozen lake
        else:
            n_actions = 4  # Default to frozen lake

        a = np.array([0] * n_actions)
        a[action] = 1
        return a
