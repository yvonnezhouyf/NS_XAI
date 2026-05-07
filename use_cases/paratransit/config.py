"""
Paratransit Use Case Configuration
Defines all paratransit-specific components for the NS-XAI system.
"""

from typing import Dict, Any, Optional, List
from .pa_pctl_mapping import PCTL_CTL_TEMPLATES


# ============================================================================
# Tree Structure Printing (for user study baseline)
# ============================================================================

def print_tree_structure(tree_node, indent: int = 0, max_depth: int = 2) -> str:
    """
    Print MCTS tree structure with aggregated summary per vehicle.

    Shows:
    - Decision(root) with total visits
    - Chance nodes (V0, V1, ...) with aggregated statistics

    Args:
        tree_node: AdaptedMCTSNode or dict representing tree node
        indent: Current indentation level (unused, kept for compatibility)
        max_depth: Maximum depth (unused, we only show root + immediate children)

    Returns:
        String representation of tree structure
    """
    if tree_node is None:
        return "(empty tree)"

    lines = []

    # Get root node attributes
    def get_attr(node, attr, default=None):
        if hasattr(node, attr):
            return getattr(node, attr, default)
        elif isinstance(node, dict):
            return node.get(attr, default)
        return default

    root_visits = get_attr(tree_node, 'visits', 0)
    root_type = get_attr(tree_node, 'type', 'decision')
    root_value = get_attr(tree_node, 'value', 0.0)

    lines.append(f"Decision(root): visits={root_visits}, value={root_value:.3f}, type={root_type}")

    # Get children (Chance nodes for each vehicle)
    children = get_attr(tree_node, 'children', [])

    if not children:
        lines.append("  (no children)")
        return "\n".join(lines)

    for child in children:
        action = get_attr(child, 'action')
        visits = get_attr(child, 'visits', 0)
        q_value = get_attr(child, 'q_value', 0.0)
        value = get_attr(child, 'value', 0.0)
        node_type = get_attr(child, 'type', 'chance')

        # Get grandchildren count (Decision nodes under this Chance node)
        grandchildren = get_attr(child, 'children', [])
        num_grandchildren = len(grandchildren) if grandchildren else 0

        # Aggregate grandchildren stats
        if grandchildren:
            gc_visits = sum(get_attr(gc, 'visits', 0) for gc in grandchildren)
            gc_values = [get_attr(gc, 'value', 0.0) for gc in grandchildren]
            avg_gc_value = sum(gc_values) / len(gc_values) if gc_values else 0.0
        else:
            gc_visits = 0
            avg_gc_value = 0.0

        action_str = f"V{action}" if action is not None else "?"
        lines.append(
            f"  ├─ Chance({action_str}): visits={visits}, Q={q_value:.3f}, "
            f"value={value:.3f}, children={num_grandchildren}, "
            f"child_visits={gc_visits}, avg_child_value={avg_gc_value:.3f}"
        )

    return "\n".join(lines)


def get_tree_for_epoch(mcts_data: Dict, epoch: int):
    """
    Get the MCTS tree for a specific epoch.

    Args:
        mcts_data: Full MCTS data with per_step_trees
        epoch: The epoch to get tree for

    Returns:
        Tree node (AdaptedMCTSNode or dict), or None if not found
    """
    per_step_trees = mcts_data.get('per_step_trees', {})
    tree_key = f"ep_1|{epoch}"
    return per_step_trees.get(tree_key)


# ============================================================================
# Scenario Definitions
# ============================================================================

class ScenarioType:
    """Enum-like class for scenario types."""
    COUNTER_INTUITIVE = "counter_intuitive_assignment"
    # Event-based scenario (node-based congestion)
    EVENT_ASSIGNMENT_CHANGE = "event_assignment_change"
    # Citywide congestion scenario (global traffic_level jump)
    CITYWIDE_CONGESTION = "citywide_congestion"
    # Scenario 0: real Case-3 adaptation run + fixed probe snapshot shadow timeline
    CONTROLLED_ADAPTATION = "controlled_adaptation"


class ParatransitConfig:
    """Configuration for Paratransit use case."""

    # Use case identifier
    USE_CASE_NAME = "paratransit"

    # Selected request indices for user study (subset of 30 requests shown to users)
    # User study subsets (10 selected requests per case):
    # COUNTER_INTUITIVE: [0, 5, 8, 12, 14, 17, 21, 22, 24, 26]
    # EVENT_ASSIGNMENT_CHANGE: [0, 6, 8, 10, 14, 16, 22, 24, 26, 27]
    USER_STUDY_SELECTED_INDICES = {
        ScenarioType.COUNTER_INTUITIVE: [0, 6, 7, 10, 11, 13, 21, 24, 25, 26],
        ScenarioType.EVENT_ASSIGNMENT_CHANGE: [0, 5, 9, 16, 18, 19, 20, 21, 23, 26],
        ScenarioType.CITYWIDE_CONGESTION: [2, 3, 8, 10, 11, 13, 20, 21, 26, 28],
        # Scenario 0 (Case 0): probe-only 10-request view. Display ids 1..10 map to these probe epochs.
        # Phases: pre = [5,6,7], mid = [10,14,19], post = [20,26,28,29].
        ScenarioType.CONTROLLED_ADAPTATION: [5, 6, 7, 10, 14, 19, 20, 26, 28, 29],
    }

    @classmethod
    def get_request_id_mapping(cls, scenario_type: str) -> Optional[Dict[int, int]]:
        """Get original_index -> display_id (1-based) mapping for user study renumbering."""
        indices = cls.USER_STUDY_SELECTED_INDICES.get(scenario_type)
        if indices is None:
            return None
        return {orig: display + 1 for display, orig in enumerate(indices)}

    @classmethod
    def get_reverse_request_id_mapping(cls, scenario_type: str) -> Optional[Dict[int, int]]:
        """Get display_id (1-based) -> original_index mapping for user study renumbering."""
        indices = cls.USER_STUDY_SELECTED_INDICES.get(scenario_type)
        if indices is None:
            return None
        return {display + 1: orig for display, orig in enumerate(indices)}

    # Valid atomic propositions for this domain (for PCTL formula validation)
    AP_WHITELIST = {
        # Core success/failure
        'service_complete',  # All requests fulfilled without violation
        'violation',         # Any timing/capacity constraint violated
        # Specific violation types
        'capacity_violation',      # Vehicle exceeds capacity
        'time_window_violation',   # Dropoff exceeds latest_dropoff (actual deadline miss)
        # Timing performance
        'pickup_delay',      # Pickup time > earliest_pickup
        'dropoff_delay',     # Dropoff time > latest_dropoff
        'any_delay',         # pickup_delay | dropoff_delay
        # Delay threshold APs (5/15/30/60 minutes)
        'pickup_delay_ge_5',
        'pickup_delay_ge_15',
        'pickup_delay_ge_30',
        'pickup_delay_ge_60',
        'dropoff_delay_ge_5',
        'dropoff_delay_ge_15',
        'dropoff_delay_ge_30',
        'dropoff_delay_ge_60',
        # Vehicle state
        'carpool_active',    # Any vehicle carries 2+ passengers
        'capacity_full',     # Any vehicle exactly at capacity
        'any_vehicle_idle',  # Any vehicle has no pending route
        'all_vehicles_busy', # All vehicles have pending routes
        # Assignment path APs
        'vehicle_busy_before_assignment',  # Action vehicle has pending route before this assignment
        'clear_current_route_ge_15',       # Clearing existing route takes >= 15 min
        'clear_current_route_ge_30',       # Clearing existing route takes >= 30 min
        'deadhead_to_pickup_ge_15',        # Deadhead from route end to pickup >= 15 min
        'deadhead_to_pickup_ge_30',        # Deadhead from route end to pickup >= 30 min
        'dropoff_slack_le_15',             # Dropoff slack (deadline - eta) <= 15 min
        'dropoff_slack_le_30',             # Dropoff slack (deadline - eta) <= 30 min
        'service_path_event_affected',     # Service path crosses event-affected nodes
        'service_path_bridge_affected',    # Service path crosses bridge S→N
    }

    # LLM Prompt IDs (OpenAI platform)
    QUERY_CLASSIFICATION_PROMPT_ID = "pmpt_6923ff4547a48193abdca0ce7badbc160e6e9bf7de64f644"
    EXPLANATION_GENERATION_PROMPT_ID = "pmpt_6924044ca09081959632fe166b26e2bd05f29b8e9936c7f9"
    # Stationary explanation prompt (uses only MDP_t, no comparison with MDP_{t-n})
    STATIONARY_EXPLANATION_PROMPT_ID = "pmpt_697d91eff8c081938bf1f8fe68fbf5900cdddd29465ab0c7"

    # Domain-specific PCTL templates
    PCTL_TEMPLATES = PCTL_CTL_TEMPLATES

    # NOTE: QUERY_TYPE_EVALUATION has been removed.
    # Evaluation level (MICRO, SEQUENCE, etc.) is now determined directly by LLM
    # classification output format: {"type_id": int, "level": "MICRO"|"SEQUENCE"}

    # Rebuild configuration for lazy MDP_{t-n} comparison
    REBUILD_MCTS_ITERATIONS = 3000  # Same as main flow for accuracy
    REBUILD_VERBOSE = True  # Print progress during rebuild

    # Available scenarios for paratransit domain
    SCENARIOS = {
        ScenarioType.COUNTER_INTUITIVE: {
            'name': 'Counter-Intuitive Vehicle Assignment',
            'description': 'Demonstrates why ADA-MCTS may choose a farther vehicle over the closest one',
            'default_params': {
                'n_vehicles': 5,
                'n_requests': 10,
                'max_iterations': 3000,
            }
        },
        ScenarioType.EVENT_ASSIGNMENT_CHANGE: {
            'name': 'Event-Based Non-Stationarity',
            'description': 'A sudden event at specific nodes increases travel time; compares old BNN vs updated BNN showing both assignment changes and PCTL shifts',
            'default_params': {
                'n_vehicles': 5,
                'n_requests': 30,
                'max_iterations': 3000,
                'event_epoch': 10,
                'event_multiplier': 3.0,
                'time_window_scale': 2.0,
            }
        },
        ScenarioType.CITYWIDE_CONGESTION: {
            'name': 'Citywide Congestion',
            'description': 'System-wide congestion increases travel times across all routes; compares old BNN vs updated BNN under global dynamics shift',
            'default_params': {
                'n_vehicles': 5,
                'n_requests': 30,
                'max_iterations': 3000,
                'congestion_epoch': 10,
                'congestion_traffic_level': 1.0,
            }
        },
        ScenarioType.CONTROLLED_ADAPTATION: {
            'name': 'Controlled Adaptation (Scenario 0)',
            'description': 'Real Case-3 congestion adaptation run with a fixed probe snapshot re-evaluated each epoch under the current BNN and current traffic level',
            'default_params': {
                'n_vehicles': 5,
                'n_requests': 30,
                'max_iterations': 3000,
                'congestion_epoch': 10,
                'congestion_traffic_level': 1.0,
            }
        },
    }

    @staticmethod
    def load_ada_mcts_tree(mcts_data: Dict):
        """
        Load ADA-MCTS tree for Paratransit domain.

        Args:
            mcts_data: Dictionary containing MCTS execution data

        Returns:
            Tuple of (mcts_root, env)
        """
        from .ada_mcts_adapter import load_ada_mcts_tree
        return load_ada_mcts_tree(mcts_data)

    @staticmethod
    def format_environment_context(mcts_data: Dict, query: str = None) -> str:
        """
        Format environment information for LLM context (Paratransit specific).

        Args:
            mcts_data: Dictionary containing MCTS execution data
            query: Optional query string to extract epoch for filtering

        Returns:
            Formatted context string for the LLM
        """
        lines = ["ENVIRONMENT CONTEXT:"]

        env_data = mcts_data.get('env_data', {})
        scenario_data = mcts_data.get('scenario_data', {})
        scenario_type = scenario_data.get('type')

        # Get renumber mapping for user study
        id_map = ParatransitConfig.get_request_id_mapping(scenario_type)

        # Basic info — show selected count if renumbering is active
        n_requests = env_data.get('n_requests', env_data.get('num_requests', 0))
        n_vehicles = env_data.get('n_vehicles', env_data.get('num_vehicles', 0))

        if n_requests:
            display_n = len(id_map) if id_map else n_requests
            lines.append(f"Total Passenger Requests: {display_n}")
        if n_vehicles:
            lines.append(f"Available Vehicles: {n_vehicles}")

        # Add scenario-specific context
        if scenario_type == ScenarioType.COUNTER_INTUITIVE:
            # Extract epoch from query if provided
            queried_epoch = None
            if query:
                queried_epoch = ParatransitConfig.extract_epoch_from_query(query, scenario_type)
            lines.append(ParatransitConfig._format_counter_intuitive_context(mcts_data, queried_epoch, id_map))
        elif scenario_type in [ScenarioType.EVENT_ASSIGNMENT_CHANGE,
                               'event_assignment_change', 'event_no_assignment_change',
                               'event_shared']:
            # Event context (node-based congestion)
            queried_epoch = None
            if query:
                queried_epoch = ParatransitConfig.extract_epoch_from_query(query, scenario_type)
            lines.append(ParatransitConfig._format_event_context(mcts_data, queried_epoch, id_map))
        elif scenario_type in [ScenarioType.CITYWIDE_CONGESTION, 'citywide_congestion']:
            queried_epoch = None
            if query:
                queried_epoch = ParatransitConfig.extract_epoch_from_query(query, scenario_type)
            lines.append(ParatransitConfig._format_congestion_context(mcts_data, queried_epoch, id_map))
        elif scenario_type in [ScenarioType.CONTROLLED_ADAPTATION, 'controlled_adaptation']:
            queried_epoch = None
            if query:
                queried_epoch = ParatransitConfig.extract_epoch_from_query(query, scenario_type)
            lines.append(ParatransitConfig._format_controlled_adaptation_context(mcts_data, queried_epoch, id_map))

        # Add time window information if available
        if 'time_windows' in env_data:
            lines.append(f"\nTime Windows: {env_data['time_windows']}")

        # Add congestion information if available
        if 'congestion_map' in env_data:
            lines.append(f"\nCongestion Status: {env_data['congestion_map']}")

        return "\n".join(lines)

    @staticmethod
    def _format_counter_intuitive_context(mcts_data: Dict, queried_epoch: Optional[int] = None,
                                          id_map: Optional[Dict[int, int]] = None) -> str:
        """
        Format context for counter-intuitive assignment scenario.

        Args:
            mcts_data: MCTS execution data
            queried_epoch: If specified, only show this epoch's details (already in original index)
            id_map: Optional original_index -> display_id mapping for user study renumbering
        """
        def _display(epoch):
            return id_map[epoch] if id_map and epoch in id_map else epoch

        lines = ["\n--- COUNTER-INTUITIVE ASSIGNMENT ANALYSIS ---"]

        scenario_data = mcts_data.get('scenario_data', {})
        ci_epochs = scenario_data.get('counter_intuitive_epochs', [])
        mdp_comparisons = scenario_data.get('mdp_comparisons', {})

        # Filter to only selected requests if renumbering is active
        if id_map:
            ci_epochs = [e for e in ci_epochs if e in id_map]

        if not ci_epochs:
            lines.append("No counter-intuitive assignments detected in this run.")
            return "\n".join(lines)

        # If a specific epoch is queried, only show that one
        if queried_epoch is not None:
            if queried_epoch in ci_epochs:
                epochs_to_show = [queried_epoch]
            else:
                lines.append(f"Request {_display(queried_epoch)} was NOT a counter-intuitive assignment.")
                return "\n".join(lines)
        else:
            epochs_to_show = ci_epochs
            lines.append(f"Counter-intuitive epochs: {[_display(e) for e in ci_epochs]}")

        for epoch in epochs_to_show:
            comparison = mdp_comparisons.get(epoch, {})
            if not comparison:
                continue

            ci_info = comparison.get('counter_intuitive_info', {})

            assigned_v = ci_info.get('assigned_vehicle', '?')
            closest_v = ci_info.get('closest_vehicle', '?')

            lines.append(f"Request {_display(epoch)}: Vehicle V{assigned_v} chosen over closer Vehicle V{closest_v}")

        return "\n".join(lines)

    @staticmethod
    def _format_event_context(mcts_data: Dict, queried_epoch: Optional[int] = None,
                              id_map: Optional[Dict[int, int]] = None) -> str:
        """
        Format context for event-caused assignment change scenario.

        Args:
            mcts_data: MCTS execution data
            queried_epoch: If specified, only show this epoch's details (already in original index)
            id_map: Optional original_index -> display_id mapping for user study renumbering
        """
        def _display(epoch):
            return id_map[epoch] if id_map and epoch in id_map else epoch

        scenario_data = mcts_data.get('scenario_data', {})

        event_config = scenario_data.get('event_config', {})
        mdp_comparisons = scenario_data.get('mdp_comparisons', {})
        trigger_epoch = scenario_data.get('trigger_epoch', None)

        # Infer event-affected epochs from trigger_epoch and mdp_comparisons
        event_affected_epochs = []
        if trigger_epoch is not None and mdp_comparisons:
            all_epochs = sorted(mdp_comparisons.keys())
            event_affected_epochs = [e for e in all_epochs if e >= trigger_epoch]

        # Filter to only selected requests if renumbering is active
        if id_map:
            event_affected_epochs = [e for e in event_affected_epochs if e in id_map]

        # Event mode (node-based congestion)
        lines = []

        event_epoch_raw = event_config.get('event_epoch', '?')
        lines.append(f"Event occurred at epoch: {_display(event_epoch_raw) if isinstance(event_epoch_raw, int) else event_epoch_raw}")

        if not event_affected_epochs:
            lines.append("\nNo event-affected epochs detected.")
            return "\n".join(lines)

        # If a specific epoch is queried, only show that one
        if queried_epoch is not None:
            if queried_epoch in event_affected_epochs:
                epochs_to_show = [queried_epoch]
                lines.append(f"\nFor epoch {_display(queried_epoch)}:")
            else:
                lines.append(f"\nEpoch {_display(queried_epoch)} is NOT in event-affected window.")
                lines.append(f"Event-affected epochs: {[_display(e) for e in event_affected_epochs]}")
                return "\n".join(lines)
        else:
            # Show first and last affected epoch for brevity
            epochs_to_show = [event_affected_epochs[0]] if event_affected_epochs else []
            if len(event_affected_epochs) > 1:
                epochs_to_show.append(event_affected_epochs[-1])

        # Prefer _runtime enriched event_info (query path), fall back to scenario_data (precompute path)
        rt_event_info = mcts_data.get('_runtime', {}).get('enriched_event_info', {})

        for epoch in epochs_to_show:
            comparison = mdp_comparisons.get(epoch, {})
            if not comparison:
                continue

            event_info = rt_event_info.get(epoch) or comparison.get('event_info', {})
            mdp_t = comparison.get('mdp_t', {})
            state_snapshot = comparison.get('state_snapshot', {})

            mdp_t_assignment = event_info.get('mdp_t_assignment', '?')
            mdp_tn_assignment = event_info.get('mdp_t_minus_n_assignment')
            assignment_changed = event_info.get('assignment_changed')
            model_version = state_snapshot.get('model_version_at_t', 0)
            
            lines.append(f"  MDP_t (updated BNN) assigned: V{mdp_t_assignment}")
            if mdp_tn_assignment is not None:
                lines.append(f"  MDP_{{t-n}} (old BNN) would assign: V{mdp_tn_assignment}")
                if assignment_changed is True:
                    lines.append(f"  → Assignment CHANGED (non-stationarity caused different decision)")
                elif assignment_changed is False:
                    lines.append(f"  → Assignment unchanged (V{mdp_t_assignment} still optimal despite environment change)")

        return "\n".join(lines)

    @staticmethod
    def _format_congestion_context(mcts_data: Dict, queried_epoch: Optional[int] = None,
                                   id_map: Optional[Dict[int, int]] = None) -> str:
        """
        Format context for citywide congestion scenario (Case 3).

        Uses the same event_info structure as Case 2, with additional
        congestion-specific fields (congestion_epoch, base/congestion_traffic_level).
        """
        def _display(epoch):
            return id_map[epoch] if id_map and epoch in id_map else epoch

        scenario_data = mcts_data.get('scenario_data', {})
        congestion_config = scenario_data.get('congestion_config', {})
        mdp_comparisons = scenario_data.get('mdp_comparisons', {})
        trigger_epoch = scenario_data.get('trigger_epoch', None)

        # Infer congestion-affected epochs
        congestion_affected_epochs = []
        if trigger_epoch is not None and mdp_comparisons:
            all_epochs = sorted(mdp_comparisons.keys())
            congestion_affected_epochs = [e for e in all_epochs if e >= trigger_epoch]

        if id_map:
            congestion_affected_epochs = [e for e in congestion_affected_epochs if e in id_map]

        lines = []
        cong_epoch = congestion_config.get('congestion_epoch', '?')
        cong_level = congestion_config.get('congestion_traffic_level', '?')
        lines.append(f"Citywide congestion occurred at epoch: {_display(cong_epoch) if isinstance(cong_epoch, int) else cong_epoch}")
        lines.append(f"System-wide congestion has increased travel times across the network (traffic_level → {cong_level})")

        if not congestion_affected_epochs:
            lines.append("\nNo congestion-affected epochs detected.")
            return "\n".join(lines)

        if queried_epoch is not None:
            if queried_epoch in congestion_affected_epochs:
                epochs_to_show = [queried_epoch]
                lines.append(f"\nFor epoch {_display(queried_epoch)}:")
            else:
                lines.append(f"\nEpoch {_display(queried_epoch)} is before congestion onset.")
                return "\n".join(lines)
        else:
            epochs_to_show = [congestion_affected_epochs[0]] if congestion_affected_epochs else []
            if len(congestion_affected_epochs) > 1:
                epochs_to_show.append(congestion_affected_epochs[-1])

        rt_event_info = mcts_data.get('_runtime', {}).get('enriched_event_info', {})

        for epoch in epochs_to_show:
            comparison = mdp_comparisons.get(epoch, {})
            if not comparison:
                continue

            event_info = rt_event_info.get(epoch) or comparison.get('event_info', {})

            mdp_t_assignment = event_info.get('mdp_t_assignment', '?')
            mdp_tn_assignment = event_info.get('mdp_t_minus_n_assignment')
            assignment_changed = event_info.get('assignment_changed')

            lines.append(f"  MDP_t (updated BNN) assigned: V{mdp_t_assignment}")
            if mdp_tn_assignment is not None:
                lines.append(f"  MDP_{{t-n}} (old BNN) would assign: V{mdp_tn_assignment}")
                if assignment_changed is True:
                    lines.append(f"  → Assignment CHANGED (congestion caused different decision)")
                elif assignment_changed is False:
                    lines.append(f"  → Assignment unchanged (V{mdp_t_assignment} still optimal despite congestion)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Legacy controlled-adaptation formatter (candidate-era 30-epoch layout).
    # Retired when Scenario 0 was finalized into the probe-only Case 0.
    # Kept as a comment block for history / easy revert.
    # ------------------------------------------------------------------
    # @staticmethod
    # def _format_controlled_adaptation_context(mcts_data, queried_epoch=None, id_map=None):
    #     """Old formatter: Case-3 congestion narrative + FIXED PROBE SNAPSHOT appendix."""
    #     lines = [ParatransitConfig._format_congestion_context(mcts_data, queried_epoch, id_map)]
    #     scenario_data = mcts_data.get('scenario_data', {})
    #     probe_config = scenario_data.get('probe_config', {})
    #     probe_results = scenario_data.get('probe_results', {})
    #     if not probe_config or not probe_results:
    #         return "\n".join(lines)
    #     lines.append("\n--- FIXED PROBE SNAPSHOT ---")
    #     pickup = probe_config.get('pickup', '?')
    #     dropoff = probe_config.get('dropoff', '?')
    #     veh_locs = probe_config.get('vehicle_locations', [])
    #     dp = probe_config.get('delta_pickup')
    #     dd = probe_config.get('delta_dropoff')
    #     lines.append(f"Probe dispatch: pickup node {pickup} → dropoff node {dropoff}")
    #     lines.append(f"Fixed idle fleet locations (V0..V{len(veh_locs) - 1}): {veh_locs}")
    #     if dp is not None and dd is not None:
    #         lines.append(
    #             f"Fixed time-window offsets (minutes): earliest_pickup = request_time + {dp}, "
    #             f"latest_dropoff = request_time + {dd}"
    #         )
    #     lines.append(
    #         "Same pickup/dropoff/fleet/offsets at every probe evaluation; "
    #         "absolute request_time inherits the real epoch's request_time."
    #     )
    #     epochs_sorted = sorted(probe_results.keys())
    #     if queried_epoch is not None:
    #         entry = probe_results.get(queried_epoch)
    #         if entry is None:
    #             lines.append(f"\nNo probe entry recorded for epoch {queried_epoch}.")
    #             return "\n".join(lines)
    #         assigned_v = entry.get('assigned_vehicle', '?')
    #         tl = entry.get('traffic_level', '?')
    #         mv = entry.get('model_version', '?')
    #         lines.append(
    #             f"\nProbe at epoch {queried_epoch}: V{assigned_v} "
    #             f"(traffic_level={tl}, model_version={mv})"
    #         )
    #         return "\n".join(lines)
    #     pre_phase = [e for e in epochs_sorted if probe_results[e].get('traffic_level') is not None
    #                  and probe_results[e].get('traffic_level') < 0.5]
    #     post_phase = [e for e in epochs_sorted if e not in pre_phase]
    #     def _summarize(phase_name, epochs):
    #         if not epochs:
    #             return f"{phase_name}: (no epochs)"
    #         vehicles = [probe_results[e].get('assigned_vehicle') for e in epochs]
    #         unique_v = sorted(set(v for v in vehicles if v is not None))
    #         return (f"{phase_name} (epochs {epochs[0]}–{epochs[-1]}): probe chose "
    #                 + ", ".join(f"V{v}" for v in unique_v))
    #     lines.append("")
    #     lines.append(_summarize("Pre-change phase", pre_phase))
    #     lines.append(_summarize("Adaptation / post-adaptation phase", post_phase))
    #     if epochs_sorted:
    #         for ep in (epochs_sorted[0], epochs_sorted[-1]):
    #             entry = probe_results.get(ep, {})
    #             lines.append(
    #                 f"  Epoch {ep}: V{entry.get('assigned_vehicle', '?')} "
    #                 f"(traffic_level={entry.get('traffic_level', '?')}, "
    #                 f"model_version={entry.get('model_version', '?')})"
    #             )
    #     return "\n".join(lines)

    @staticmethod
    def _format_controlled_adaptation_context(mcts_data: Dict, queried_epoch: Optional[int] = None,
                                              id_map: Optional[Dict[int, int]] = None) -> str:
        """
        Case 0 probe-only context formatter (the only active CONTROLLED_ADAPTATION formatter).

        Scenario 0's user-facing surface is now a curated probe-only 10-request view. All
        narrative is probe-centric; the real Case-3 backbone is not described here.

        When `queried_epoch` maps to a post-congestion probe epoch, the formatter surfaces the
        probe's MDP_t vs MDP_{t-n} comparison (both sides built from probe trees, one under the
        updated BNN, one rebuilt under the previous BNN) from `scenario_data['mdp_comparisons']`.
        """
        def _display(epoch):
            return id_map[epoch] if id_map and epoch in id_map else epoch

        scenario_data = mcts_data.get('scenario_data', {})
        probe_config = scenario_data.get('probe_config', {}) or {}
        probe_results = scenario_data.get('probe_results', {}) or {}
        mdp_comparisons = scenario_data.get('mdp_comparisons', {}) or {}
        congestion_config = scenario_data.get('congestion_config', {}) or {}
        trigger_epoch = scenario_data.get('trigger_epoch')

        selected = ParatransitConfig.USER_STUDY_SELECTED_INDICES.get(
            ScenarioType.CONTROLLED_ADAPTATION, []
        )
        # Phase partitioning based on trigger_epoch if known; else fall back to the canonical
        # 3/3/4 split used by Case 0 (pre=[5,6,7], mid=[10,14,19], post=[20,26,28,29]).
        if trigger_epoch is not None and selected:
            pre = [e for e in selected if e < trigger_epoch]
            non_pre = [e for e in selected if e >= trigger_epoch]
            # mid = first 3 post-trigger selected epochs, post = the rest (preserves 3/3/4 layout)
            mid, post = non_pre[:3], non_pre[3:]
        else:
            pre, mid, post = selected[:3], selected[3:6], selected[6:]

        lines = []

        cong_epoch = congestion_config.get('congestion_epoch')
        if cong_epoch is not None:
            cong_level = congestion_config.get('congestion_traffic_level')
            base_level = congestion_config.get('base_traffic_level')
            lines.append(
                f"Congestion onset at epoch {cong_epoch} (display id "
                f"{_display(cong_epoch)}): traffic_level {base_level} → {cong_level}"
            )

        # Per-epoch detail or phase summary
        if queried_epoch is not None:
            entry = probe_results.get(queried_epoch)
            if entry is None:
                lines.append(
                    f"\nNo probe entry recorded for epoch {queried_epoch} "
                    f"(display id {_display(queried_epoch)})."
                )
                return "\n".join(lines)

            assigned_v = entry.get('assigned_vehicle', '?')
            closest_v = entry.get('closest_vehicle')

            closest_str = f"V{closest_v}" if closest_v is not None else "N/A"
            lines.append(f"\nAssigned Vehicle: V{assigned_v}, Closest Vehicle: {closest_str}")

            comparison = mdp_comparisons.get(queried_epoch) or {}
            event_info = comparison.get('event_info') or {}
            mdp_tn = event_info.get('mdp_t_minus_n_assignment')
            mdp_t = event_info.get('mdp_t_assignment', assigned_v)
            assignment_changed = event_info.get('assignment_changed')

            if mdp_tn is not None:
                lines.append(f"  MDP_t (probe under updated BNN) chose: V{mdp_t}")
                lines.append(f"  MDP_{{t-n}} (probe under previous BNN) would have chosen: V{mdp_tn}")
                if assignment_changed is True:
                    lines.append("  → Probe assignment CHANGED (non-stationarity affected the probe decision)")
                elif assignment_changed is False:
                    lines.append(f"  → Probe assignment unchanged (V{mdp_t} still optimal under the previous BNN)")
            elif trigger_epoch is not None and queried_epoch < trigger_epoch:
                lines.append("  (pre-congestion probe — no MDP_{t-n} comparison applicable)")

            return "\n".join(lines)

        # No specific epoch: phase-level summary of probe choices
        def _phase_vehicles(name, epochs):
            if not epochs:
                return f"  {name}: (none)"
            veh = [probe_results[e].get('assigned_vehicle') for e in epochs if e in probe_results]
            unique_v = sorted({v for v in veh if v is not None})
            return f"  {name}: probe chose {[f'V{v}' for v in unique_v] or 'N/A'}"

        lines.append("")
        lines.append(_phase_vehicles("Pre-change phase   (requests 1-3)", pre))
        lines.append(_phase_vehicles("Mid / adaptation   (requests 4-6)", mid))
        lines.append(_phase_vehicles("Post-adaptation    (requests 7-10)", post))

        return "\n".join(lines)

    @staticmethod
    def build_tree_from_data(tree_data):
        """
        Build MCTS tree node from serialized data.

        Args:
            tree_data: Serialized tree data (dict format)

        Returns:
            AdaptedMCTSNode object for paratransit
        """
        from .ada_mcts_adapter import build_tree_from_data
        return build_tree_from_data(tree_data)

    @staticmethod
    def check_atomic_proposition(node, env, prop: str) -> bool:
        """
        Check if an atomic proposition is satisfied by the given node.

        Uses _cached_props computed during MCTS execution by
        compute_and_cache_atomic_propositions() in adamcts_runner.py,
        which delegates to env.evaluate_atomic_props() for consistency.

        Paratransit atomic propositions (from nsparatransit_v0.py):
        - service_complete: All requests fulfilled without violation
        - violation: Any constraint violated (timing/capacity)
        - capacity_violation: Vehicle exceeds capacity
        - time_window_violation: Dropoff exceeds latest_dropoff
        - pickup_delay: Pickup time > earliest_pickup
        - dropoff_delay: Dropoff time > latest_dropoff
        - pickup_delay_ge_5/15/30/60: Pickup delay >= threshold minutes
        - dropoff_delay_ge_5/15/30/60: Dropoff delay >= threshold minutes
        - any_delay: pickup_delay OR dropoff_delay
        - carpool_active: Vehicle has 2+ passengers
        - capacity_full: Vehicle exactly at capacity
        - any_vehicle_idle: Any vehicle has no pending route
        - all_vehicles_busy: All vehicles have pending routes

        Args:
            node: MCTS node (AdaptedMCTSNode or dict)
            env: Environment adapter (ParatransitEnvAdapter) - unused but kept for interface consistency
            prop: Atomic proposition name

        Returns:
            Boolean result
        """
        # Check node's cached atomic props (computed by adamcts_runner.py)
        if hasattr(node, '_cached_props') and prop in node._cached_props:
            return node._cached_props[prop]

        # Fallback: check if node has get_atomic_prop method (AdaptedMCTSNode)
        if hasattr(node, 'get_atomic_prop'):
            return node.get_atomic_prop(prop)

        # Unknown proposition or no cached props
        return False

    @staticmethod
    def get_n_actions(mcts_data: dict) -> int:
        """
        Get the number of actions (vehicles) for this domain.

        Args:
            mcts_data: MCTS data containing env_data

        Returns:
            Number of vehicles (actions)
        """
        env_data = mcts_data.get('env_data', {})
        return env_data.get('n_vehicles', 5)

    @staticmethod
    def action_to_name(action: int) -> str:
        """
        Convert action number to human-readable name.

        Args:
            action: Vehicle ID (0 to n_vehicles-1)

        Returns:
            String like "V0", "V1", etc.
        """
        return f"V{action}"

    @staticmethod
    def extract_epoch_from_query(query: str, scenario_type: str = None) -> Optional[int]:
        """
        Extract decision epoch / request number from query.

        When scenario_type is provided and a user study renumber mapping exists,
        the user's display ID (1-based) is reverse-mapped to the original index.

        Supported patterns:
        - Synonym + digit: "epoch 5", "trip 3", "request 5", "at decision epoch 5",
          "for ride 2", "booking 1", "pickup 4", "step 7", "round 3", etc.
        - Hash / number prefix: "#5", "no. 5", "number 5"
        - Ordinal digit: "5th request", "3rd trip", "the 1st epoch"
        - Spelled-out cardinal (0-10): "trip five", "epoch two", "request ten"
        - Spelled-out ordinal (1st-10th): "fifth trip", "the third request"

        Args:
            query: User query string
            scenario_type: Optional scenario type for reverse-mapping display IDs

        Returns:
            Extracted epoch/request number (original index), or None if not found
        """
        import re
        query_lower = query.lower()

        # Reverse mapping: display_id -> original_index
        reverse_map = None
        if scenario_type:
            reverse_map = ParatransitConfig.get_reverse_request_id_mapping(scenario_type)

        # Spelled-out numbers (0-10) and their ordinal forms
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10,
        }
        ordinal_to_num = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
            'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
        }

        # Synonyms users might use for epoch/request in paratransit context
        epoch_synonyms = (
            r'(?:decision\s+)?epoch|request(?:\s+id)?|'
            r'trip|ride|call|booking|pickup|order|stop|'
            r'(?:time\s+)?step|stage|iteration|round|'
            r'(?:decision\s+)?point|time(?:\s+period)?|period'
        )

        word_num_pattern = '|'.join(word_to_num.keys())
        ordinal_pattern = '|'.join(ordinal_to_num.keys())

        # --- Digit-based patterns ---
        patterns_digit = [
            # "epoch 5", "trip 5", "request 5", "at decision epoch 5", etc.
            rf'(?:at\s+|for\s+)?(?:{epoch_synonyms})\s+#?\s*(\d+)',
            # "#5", "no. 5", "number 5", "no 5"
            r'(?:#|no\.?\s*|number\s+)(\d+)',
            # Ordinal digit: "5th request", "the 5th"
            rf'(\d+)(?:st|nd|rd|th)\s*(?:{epoch_synonyms})?',
        ]

        extracted = None

        for pattern in patterns_digit:
            match = re.search(pattern, query_lower)
            if match:
                extracted = int(match.group(1))
                break

        # --- Spelled-out cardinal: "trip five", "epoch two" ---
        if extracted is None:
            match = re.search(
                rf'(?:at\s+|for\s+)?(?:{epoch_synonyms})\s+({word_num_pattern})',
                query_lower,
            )
            if match:
                extracted = word_to_num[match.group(1)]

        # --- Spelled-out ordinal: "fifth trip", "the third request" ---
        if extracted is None:
            match = re.search(
                rf'(?:the\s+)?({ordinal_pattern})\s+(?:{epoch_synonyms})',
                query_lower,
            )
            if match:
                extracted = ordinal_to_num[match.group(1)]

        if extracted is None:
            return None

        # Reverse-map display ID to original index if renumbering is active
        if reverse_map and extracted in reverse_map:
            return reverse_map[extracted]

        return extracted

    @staticmethod
    def extract_vehicles_from_query(query: str, n_vehicles: int = 5) -> List[Dict]:
        """
        Extract vehicle references from query in mention order.

        Returns ordered list of raw references. Each entry is one of:
            {'type': 'concrete', 'id': int}
            {'type': 'role', 'role': 'assigned'|'closest'}

        Supported concrete aliases: v, veh, vehicle, car, van, shuttle, bus
        Supported concrete forms: v3, v-3, veh3, vehicle 3, car3, car-3, etc.
        Supported assigned roles: assigned, chosen, selected, recommended
        Supported closest roles: closest, nearest
        Role nouns: vehicle, car, van, shuttle, bus

        Bare numbers (e.g., "3") are NOT matched — they overlap with epoch/request refs.

        Args:
            query: User query string
            n_vehicles: Total number of vehicles (for range validation)

        Returns:
            List of reference dicts in mention order, deduped by value
        """
        import re
        query_lower = query.lower()

        # Collect (offset, ref_dict) tuples, then sort by offset
        hits = []

        # Concrete vehicle pattern: alias + optional separator + digit(s)
        _ALIASES = r'(?:v|veh|vehicle|car|van|shuttle|bus)'
        concrete_re = re.compile(
            rf'{_ALIASES}[\s\-]?(\d+)', re.IGNORECASE
        )
        for m in concrete_re.finditer(query_lower):
            v_id = int(m.group(1))
            if 0 <= v_id < n_vehicles:
                hits.append((m.start(), {'type': 'concrete', 'id': v_id}))

        # Role pattern: adjective + noun
        _ASSIGNED_ADJ = r'(?:assigned|chosen|selected|recommended)'
        _CLOSEST_ADJ = r'(?:closest|nearest)'
        _NOUNS = r'(?:vehicle|car|van|shuttle|bus)'

        for m in re.finditer(rf'{_ASSIGNED_ADJ}\s+{_NOUNS}', query_lower):
            hits.append((m.start(), {'type': 'role', 'role': 'assigned'}))
        for m in re.finditer(rf'{_CLOSEST_ADJ}\s+{_NOUNS}', query_lower):
            hits.append((m.start(), {'type': 'role', 'role': 'closest'}))

        # Sort by offset (mention order)
        hits.sort(key=lambda x: x[0])

        # Dedup preserving first-mention order
        seen = set()
        refs = []
        for _, ref in hits:
            key = ('concrete', ref['id']) if ref['type'] == 'concrete' else ('role', ref['role'])
            if key not in seen:
                seen.add(key)
                refs.append(ref)

        return refs

    @staticmethod
    def resolve_vehicle_references(
        refs: List[Dict], assigned_id: int, closest_id: int
    ) -> List[int]:
        """
        Resolve raw vehicle references to concrete vehicle IDs.

        Called after epoch and assignment info are known.

        Args:
            refs: Output of extract_vehicles_from_query (ordered raw references)
            assigned_id: Assigned vehicle ID for the current epoch
            closest_id: Closest vehicle ID for the current epoch

        Returns:
            Ordered list of unique vehicle IDs
        """
        seen = set()
        result = []
        for ref in refs:
            if ref['type'] == 'concrete':
                v_id = ref['id']
            elif ref['role'] == 'assigned':
                v_id = assigned_id
            elif ref['role'] == 'closest':
                v_id = closest_id
            else:
                continue
            if v_id not in seen:
                seen.add(v_id)
                result.append(v_id)
        return result
