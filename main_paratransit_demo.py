#!/usr/bin/env python3
"""
NS-XAI Interactive Demo for Paratransit
Runs ADA-MCTS on paratransit domain and provides an interactive Q&A session for explaining dispatcher behavior.

Supports multiple scenarios:
1. Counter-Intuitive Vehicle Assignment - Demonstrates non-stationarity through
   comparing MDP_{t-n} vs MDP_t when a farther vehicle is chosen over the closest one.
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
import traceback

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

from copy import deepcopy

from use_cases.paratransit.ada_mcts_adapter import check_conda_environment
from use_cases.paratransit.config import ParatransitConfig, ScenarioType
from use_cases.paratransit.adamcts_runner import (
    run_counter_intuitive_scenario, run_event_scenario, run_congestion_scenario,
    run_scenario0_scenario,
)

from core.orchestrator import NSXAIOrchestrator, prepare_runtime_artifacts
from core.formula_evaluator import evaluate_mdp_comparison, set_active_config as set_fe_config
from core.pctl_checking import set_active_config as set_pctl_config

_DATA_DIR = os.path.join(os.path.dirname(__file__), "use_cases", "paratransit", "data")

# ── Scenario Registry ────────────────────────────────────────────────
# Each entry provides the minimal per-scenario config; the template
# functions _load_or_generate / _run_scenario_demo handle the rest.
SCENARIO_REGISTRY = {
    0: {
        'saved_file': 'saved_scenario0_results.pkl',
        'label': 'Scenario 0 — Controlled Adaptation (probe-only 10-request view, Case 0)',
        'scenario_type': ScenarioType.CONTROLLED_ADAPTATION,
        'requests_csv': os.path.join(_DATA_DIR, 'case3_chains.csv'),
        'runner': None,
        'runner_kwargs': None,
        'probe_config': {
            'pickup': 7309, 'dropoff': 6106,
            'delta_pickup': 21.75, 'delta_dropoff': 37.7667,
            'vehicle_locations': [4025, 8958, 175, 454, 3482],
        },
        'precompute': None,
        'banner': [
            "Scenario 0: probe-only 10-request view derived from the real Case-3 adaptation run.",
            "Probe (pickup 7309 → dropoff 6106, fixed idle fleet) at 10 selected epochs.",
            "MDP_t (probe under updated BNN) vs MDP_{t-n} (probe under previous BNN).",
        ],
    },
    1: {
        'saved_file': 'saved_counter_intuitive_assignment_results.pkl',
        'label': 'Counter-Intuitive Vehicle Assignment (Case 1)',
        'scenario_type': ScenarioType.COUNTER_INTUITIVE,
        'requests_csv': os.path.join(_DATA_DIR, 'case1_chains.csv'),
        'runner': run_counter_intuitive_scenario,
        'runner_kwargs': {
            'n_vehicles': 5, 'n_requests': 30, 'max_iterations': 3000, 'seed': 42,
            'bridge_accident_epoch': 10, 'bridge_accident_multiplier': 2.0,
        },
        'precompute': None,
        'banner': [
            "This scenario demonstrates ADA-MCTS's adaptation to non-stationarity.",
            "When a farther vehicle is chosen over the closest one, we compare",
            "MDP_{t-n} vs MDP_t to show how the environment understanding changed.",
        ],
    },
    2: {
        'saved_file': 'saved_event_scenario_results.pkl',
        'label': 'Event-Based Non-Stationarity (Case 2)',
        'scenario_type': ScenarioType.EVENT_ASSIGNMENT_CHANGE,
        'requests_csv': os.path.join(_DATA_DIR, 'case2_chains.csv'),
        'runner': run_event_scenario,
        'runner_kwargs': {
            'n_vehicles': 5, 'n_requests': 30, 'max_iterations': 3000, 'seed': 42,
            'event_epoch': 10, 'event_nodes': [225], 'event_multiplier': 3.0,
        },
        'precompute': 'precompute_event_comparisons',  # resolved lazily
        'banner': [
            "This scenario demonstrates non-stationarity caused by a sudden event.",
            "At a specific epoch, congestion at certain nodes increases travel time.",
            "Compares OLD BNN vs UPDATED BNN decisions — some epochs show assignment",
            "changes, others show PCTL shifts with the same vehicle still optimal.",
        ],
    },
    3: {
        'saved_file': 'saved_congestion_scenario_results.pkl',
        'label': 'Citywide Congestion (Case 3)',
        'scenario_type': ScenarioType.CITYWIDE_CONGESTION,
        'requests_csv': os.path.join(_DATA_DIR, 'case3_chains.csv'),
        'runner': run_congestion_scenario,
        'runner_kwargs': {
            'n_vehicles': 5, 'n_requests': 30, 'max_iterations': 3000, 'seed': 42,
            'congestion_epoch': 10, 'congestion_traffic_level': 1.0,
        },
        'precompute': None,
        'banner': [
            "This scenario demonstrates non-stationarity from citywide congestion.",
            "At epoch 10, system-wide congestion increases travel times across all routes.",
            "Compares OLD BNN vs UPDATED BNN decisions — adaptation to global dynamics shift.",
        ],
    },
    # NOTE: Cases 4/5/6 (candidate1/2/3 probe scan) are retired from the user-facing
    # surface. Case 0 is the single final Scenario 0 product. The candidate entries are
    # preserved as comments so the candidate research flow can be resurrected if needed.
    # 4: {
    #     'saved_file': 'saved_scenario0_candidate1_results.pkl',
    #     'label': 'Scenario 0 — Controlled Adaptation, C1 (probe req26: 4649→4292)',
    #     'scenario_type': ScenarioType.CONTROLLED_ADAPTATION,
    #     'requests_csv': os.path.join(_DATA_DIR, 'case3_chains.csv'),
    #     'runner': run_scenario0_scenario,
    #     'runner_kwargs': {
    #         'n_vehicles': 5, 'n_requests': 30, 'max_iterations': 3000, 'seed': 42,
    #         'congestion_epoch': 10, 'congestion_traffic_level': 1.0,
    #     },
    #     'probe_config': {
    #         'pickup': 4649, 'dropoff': 4292,
    #         'delta_pickup': 17.7, 'delta_dropoff': 39.7167,
    #         'vehicle_locations': [4025, 8958, 175, 454, 3482],
    #     },
    #     'precompute': None,
    #     'banner': [
    #         "Scenario 0: real Case-3 adaptation run + fixed probe snapshot.",
    #         "Probe C1 (req 26) — pickup 4649 → dropoff 4292, shared synthetic idle fleet.",
    #         "Probe is re-evaluated at every decision epoch under current BNN and traffic.",
    #     ],
    # },
    # 5: {
    #     'saved_file': 'saved_scenario0_candidate2_results.pkl',
    #     'label': 'Scenario 0 — Controlled Adaptation, C2 (probe req27: 4649→769)',
    #     'scenario_type': ScenarioType.CONTROLLED_ADAPTATION,
    #     'requests_csv': os.path.join(_DATA_DIR, 'case3_chains.csv'),
    #     'runner': run_scenario0_scenario,
    #     'runner_kwargs': {
    #         'n_vehicles': 5, 'n_requests': 30, 'max_iterations': 3000, 'seed': 42,
    #         'congestion_epoch': 10, 'congestion_traffic_level': 1.0,
    #     },
    #     'probe_config': {
    #         'pickup': 4649, 'dropoff': 769,
    #         'delta_pickup': 16.4833, 'delta_dropoff': 49.0333,
    #         'vehicle_locations': [4025, 8958, 175, 454, 3482],
    #     },
    #     'precompute': None,
    #     'banner': [
    #         "Scenario 0: real Case-3 adaptation run + fixed probe snapshot.",
    #         "Probe C2 (req 27) — pickup 4649 → dropoff 769, shared synthetic idle fleet.",
    #         "Probe is re-evaluated at every decision epoch under current BNN and traffic.",
    #     ],
    # },
    # 6: {
    #     'saved_file': 'saved_scenario0_candidate3_results.pkl',
    #     'label': 'Scenario 0 — Controlled Adaptation, C3 (probe req10: 7309→6106)',
    #     'scenario_type': ScenarioType.CONTROLLED_ADAPTATION,
    #     'requests_csv': os.path.join(_DATA_DIR, 'case3_chains.csv'),
    #     'runner': run_scenario0_scenario,
    #     'runner_kwargs': {
    #         'n_vehicles': 5, 'n_requests': 30, 'max_iterations': 3000, 'seed': 42,
    #         'congestion_epoch': 10, 'congestion_traffic_level': 1.0,
    #     },
    #     'probe_config': {
    #         'pickup': 7309, 'dropoff': 6106,
    #         'delta_pickup': 21.75, 'delta_dropoff': 37.7667,
    #         'vehicle_locations': [4025, 8958, 175, 454, 3482],
    #     },
    #     'precompute': None,
    #     'banner': [
    #         "Scenario 0: real Case-3 adaptation run + fixed probe snapshot.",
    #         "Probe C3 (req 10) — pickup 7309 → dropoff 6106, shared synthetic idle fleet.",
    #         "Probe is re-evaluated at every decision epoch under current BNN and traffic.",
    #     ],
    # },
}

# Retired candidate-scan bookkeeping (Cases 4/5/6). Kept commented for history only.
# SCENARIO0_CASES = (4, 5, 6)
# SCENARIO0_CANDIDATE_NAMES = ('candidate1', 'candidate2', 'candidate3')
# SCENARIO0_CASE_TO_CANDIDATE = dict(zip(SCENARIO0_CASES, SCENARIO0_CANDIDATE_NAMES))


def check_python_version():
    """Check if we're running in Python 3.9 (XAI environment)."""
    if sys.version_info[:2] != (3, 9):
        print(f"[!] WARNING: Expected Python 3.9, got {sys.version_info[:2]}")
        print("Make sure you're in the 'xai39' conda environment:")
        print("    conda activate xai39")
        return False
    return True


def check_environment_setup():
    """Verify that both conda environments exist and are properly configured."""
    print("➤ Checking environment setup...")

    # Check adamcts38 environment
    if check_conda_environment("adamcts38"):
        print("[OK] ADA-MCTS environment (adamcts38) found")
    else:
        print("[x] ADA-MCTS environment (adamcts38) not found")
        print("   Create it with: conda create -n adamcts38 python=3.8")
        return False

    # Check xai39 environment
    if check_conda_environment("xai39"):
        print("[OK] NS-XAI environment (xai39) found")
    else:
        print("[x] NS-XAI environment (xai39) not found")
        print("   Create it with: conda create -n xai39 python=3.9")
        return False

    # Check OpenAI API key
    if HAS_DOTENV:
        load_dotenv()

    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key and len(openai_key) > 20:
        print("[OK] OpenAI API key configured")
    else:
        print("[x] OpenAI API key not found")
        print("   Add it to .env file: OPENAI_API_KEY=your_key_here")
        return False

    print("\n[OK] Environment setup verification complete!\n")
    return True


def precompute_event_comparisons(mcts_data):
    """
    Precompute MDP_{t-n} best-action comparison for event-affected epochs.

    Only rebuilds old-model tree and computes best action (no full PCTL formula evaluation).
    PCTL evaluation is left to warmup or first query time.

    Stores comparison_results in scenario_data with:
    - mdp_t_minus_n_assignment: vehicle chosen by old BNN
    - mdp_t_assignment: vehicle chosen by updated BNN
    - assignment_changed: True/False
    """
    from core.formula_evaluator import (
        _rebuild_mdp_for_comparison, _compute_best_action_from_tree,
        _get_run_id, _MDP_REBUILD_CACHE,
        set_active_config as fe_set_config,
    )

    scenario_data = mcts_data.get('scenario_data', {})
    mdp_comparisons = scenario_data.get('mdp_comparisons', {})
    model_version_history = scenario_data.get('model_version_history', {})
    trigger_epoch = scenario_data.get('trigger_epoch', None)

    # Infer event-affected epochs: all epochs from trigger_epoch onwards
    event_affected_epochs = []
    if trigger_epoch is not None and mdp_comparisons:
        all_epochs = sorted(mdp_comparisons.keys())
        event_affected_epochs = [e for e in all_epochs if e >= trigger_epoch]

    if not event_affected_epochs:
        print("[PRECOMPUTE] No event-affected epochs found, skipping precomputation")
        return

    print(f"\n[PRECOMPUTE] Rebuilding MDP_{{t-n}} for {len(event_affected_epochs)} event-affected epochs: {event_affected_epochs}")

    cfg = ParatransitConfig()
    fe_set_config(cfg)

    comparison_results = scenario_data.setdefault('comparison_results', {})

    for epoch in event_affected_epochs:
        print(f"  [PRECOMPUTE] Rebuilding MDP_{{t-n}} for epoch {epoch}...")
        try:
            comparison_data = mdp_comparisons.get(epoch, {})
            state_snapshot = comparison_data.get('state_snapshot', {})

            # Use pre-computed prev_version_id (falls back to scan for legacy pkl)
            prev_version_id = state_snapshot.get('prev_version_id')
            if prev_version_id is None:
                model_version_at_t = state_snapshot.get('model_version_at_t', 0)
                if model_version_history and model_version_at_t > 0:
                    for v_id in range(model_version_at_t - 1, -1, -1):
                        if v_id in model_version_history:
                            prev_version_id = v_id
                            break
                elif model_version_history and model_version_at_t == 0 and 0 in model_version_history:
                    prev_version_id = 0

            if prev_version_id is None:
                continue

            # Rebuild MDP_{t-n} tree (with pkl caching)
            run_id = _get_run_id(mcts_data)
            cache_key = (run_id, epoch, prev_version_id)
            tree_t_minus_n = _MDP_REBUILD_CACHE.get(cache_key)

            if tree_t_minus_n is None:
                tree_dict = _rebuild_mdp_for_comparison(
                    state_snapshot, model_version_history, prev_version_id, mcts_data)
                if tree_dict is not None:
                    tree_t_minus_n = cfg.build_tree_from_data(tree_dict)
                    if tree_t_minus_n is not None:
                        _MDP_REBUILD_CACHE[cache_key] = tree_t_minus_n

            if tree_t_minus_n is None:
                continue

            # Compute best action from old BNN tree
            mdp_t_minus_n_assignment = _compute_best_action_from_tree(tree_t_minus_n)
            event_info = comparison_data.get('event_info', {})
            mdp_t_assignment = event_info.get('mdp_t_assignment')

            if mdp_t_assignment is not None and mdp_t_minus_n_assignment is not None:
                true_changed = (mdp_t_assignment != mdp_t_minus_n_assignment)
                comparison_results[epoch] = {
                    'assignment_changed': true_changed,
                    'mdp_t_minus_n_assignment': mdp_t_minus_n_assignment,
                    'mdp_t_assignment': mdp_t_assignment,
                    'comparison_epoch': epoch,
                }

        except Exception as e:
            print(f"  [PRECOMPUTE] Warning: Failed to rebuild epoch {epoch}: {e}")

    changed_count = sum(1 for r in comparison_results.values() if r.get('assignment_changed') is True)
    unchanged_count = sum(1 for r in comparison_results.values() if r.get('assignment_changed') is False)

    print(f"[PRECOMPUTE] Complete:")
    print(f"  - Epochs where assignment changed: {changed_count}")
    print(f"  - Epochs where assignment unchanged: {unchanged_count}")


def precompute_congestion_comparisons(mcts_data):
    """
    Precompute MDP_{t-n} best-action comparison for congestion-affected epochs.

    Same logic as precompute_event_comparisons but reads event_info
    (which Case 3 also uses for enrichment chain compatibility).
    """
    from core.formula_evaluator import (
        _rebuild_mdp_for_comparison, _compute_best_action_from_tree,
        _get_run_id, _MDP_REBUILD_CACHE,
        set_active_config as fe_set_config,
    )

    scenario_data = mcts_data.get('scenario_data', {})
    mdp_comparisons = scenario_data.get('mdp_comparisons', {})
    model_version_history = scenario_data.get('model_version_history', {})
    trigger_epoch = scenario_data.get('trigger_epoch', None)

    congestion_affected_epochs = []
    if trigger_epoch is not None and mdp_comparisons:
        all_epochs = sorted(mdp_comparisons.keys())
        congestion_affected_epochs = [e for e in all_epochs if e >= trigger_epoch]

    if not congestion_affected_epochs:
        print("[PRECOMPUTE] No congestion-affected epochs found, skipping precomputation")
        return

    print(f"\n[PRECOMPUTE] Rebuilding MDP_{{t-n}} for {len(congestion_affected_epochs)} congestion-affected epochs: {congestion_affected_epochs}")

    cfg = ParatransitConfig()
    fe_set_config(cfg)

    comparison_results = scenario_data.setdefault('comparison_results', {})

    for epoch in congestion_affected_epochs:
        print(f"  [PRECOMPUTE] Rebuilding MDP_{{t-n}} for epoch {epoch}...")
        try:
            comparison_data = mdp_comparisons.get(epoch, {})
            state_snapshot = comparison_data.get('state_snapshot', {})

            prev_version_id = state_snapshot.get('prev_version_id')
            if prev_version_id is None:
                model_version_at_t = state_snapshot.get('model_version_at_t', 0)
                if model_version_history and model_version_at_t > 0:
                    for v_id in range(model_version_at_t - 1, -1, -1):
                        if v_id in model_version_history:
                            prev_version_id = v_id
                            break
                elif model_version_history and model_version_at_t == 0 and 0 in model_version_history:
                    prev_version_id = 0

            if prev_version_id is None:
                continue

            run_id = _get_run_id(mcts_data)
            cache_key = (run_id, epoch, prev_version_id)
            tree_t_minus_n = _MDP_REBUILD_CACHE.get(cache_key)

            if tree_t_minus_n is None:
                tree_dict = _rebuild_mdp_for_comparison(
                    state_snapshot, model_version_history, prev_version_id, mcts_data)
                if tree_dict is not None:
                    tree_t_minus_n = cfg.build_tree_from_data(tree_dict)
                    if tree_t_minus_n is not None:
                        _MDP_REBUILD_CACHE[cache_key] = tree_t_minus_n

            if tree_t_minus_n is None:
                continue

            mdp_t_minus_n_assignment = _compute_best_action_from_tree(tree_t_minus_n)
            event_info = comparison_data.get('event_info', {})
            mdp_t_assignment = event_info.get('mdp_t_assignment')

            if mdp_t_assignment is not None and mdp_t_minus_n_assignment is not None:
                true_changed = (mdp_t_assignment != mdp_t_minus_n_assignment)
                comparison_results[epoch] = {
                    'assignment_changed': true_changed,
                    'mdp_t_minus_n_assignment': mdp_t_minus_n_assignment,
                    'mdp_t_assignment': mdp_t_assignment,
                    'comparison_epoch': epoch,
                }

        except Exception as e:
            print(f"  [PRECOMPUTE] Warning: Failed to rebuild epoch {epoch}: {e}")

    changed_count = sum(1 for r in comparison_results.values() if r.get('assignment_changed') is True)
    unchanged_count = sum(1 for r in comparison_results.values() if r.get('assignment_changed') is False)

    print(f"[PRECOMPUTE] Complete:")
    print(f"  - Epochs where assignment changed: {changed_count}")
    print(f"  - Epochs where assignment unchanged: {unchanged_count}")


# ============================================================================
# Case 0 — probe-only view built from the final Scenario 0 pkl.
# ============================================================================

# Fixed probe configuration for Case 0 (same as the retired candidate3 probe).
_CASE0_PROBE_CONFIG = {
    'pickup': 7309,
    'dropoff': 6106,
    'delta_pickup': 21.75,
    'delta_dropoff': 37.7667,
    'vehicle_locations': [4025, 8958, 175, 454, 3482],
}


def _transform_to_case0(mcts_data):
    """Mutate `mcts_data` in place into the Case 0 probe-only runtime view.

    The caller must pass a deep copy — this function does not protect the pkl.
    See Case0_xai_plan.md Step 4 and the approved plan Step 4a.
    """
    scenario_data = mcts_data.setdefault('scenario_data', {})
    env_data = mcts_data.setdefault('env_data', {})

    # 1) Normalize scenario type on both the outer record and scenario_data.
    mcts_data['scenario_type'] = ScenarioType.CONTROLLED_ADAPTATION
    scenario_data['type'] = ScenarioType.CONTROLLED_ADAPTATION

    # 2) Strip candidate-era residue from the in-memory copy (pkl on disk untouched).
    scenario_data.pop('probe_results_by_candidate', None)
    scenario_data.pop('probe_configs', None)
    scenario_data.pop('active_candidate', None)
    for key in list(scenario_data.keys()):
        if isinstance(key, str) and 'candidate' in key:
            scenario_data.pop(key, None)

    # Ensure probe_config reflects the final Case 0 probe identity.
    probe_cfg = dict(_CASE0_PROBE_CONFIG)
    existing_cfg = scenario_data.get('probe_config')
    if isinstance(existing_cfg, dict):
        probe_cfg.update({k: existing_cfg.get(k, v) for k, v in probe_cfg.items()})
    scenario_data['probe_config'] = probe_cfg

    probe_results = scenario_data.get('probe_results', {}) or {}

    # 3) Drop backbone trees and re-key probe trees as `ep_1|<epoch>`.
    new_trees = {}
    for key, val in mcts_data.get('per_step_trees', {}).items():
        if not isinstance(key, str):
            continue
        if key.startswith('probe|candidate'):
            # Defensive: pre-split pkls may still carry per-candidate keys.
            tail = key.split('|')[-1]
            try:
                ep = int(tail)
            except ValueError:
                continue
            new_trees[f'ep_1|{ep}'] = val
        elif key.startswith('probe|'):
            try:
                ep = int(key.split('|', 1)[1])
            except ValueError:
                continue
            new_trees[f'ep_1|{ep}'] = val
        # else: drop existing ep_1|* (backbone) and any other keys
    mcts_data['per_step_trees'] = new_trees

    # 4) Rebuild env_data['assignment_details'] as a 30-slot list indexed by epoch.
    n_slots = int(env_data.get('n_requests', 30) or 30)
    # Case 0 always operates on a 30-slot timeline regardless of n_requests drift.
    n_slots = max(n_slots, 30)
    details: list = [None] * n_slots

    for ep, entry in probe_results.items():
        if not isinstance(ep, int) or ep < 0 or ep >= n_slots:
            continue
        rt = entry.get('request_time')
        if rt is None:
            snap = (scenario_data.get('mdp_comparisons', {}).get(ep, {})
                    .get('state_snapshot', {}).get('state', {}))
            current_req = snap.get('current_request', {})
            rt = current_req.get('request_time', 0.0)
        earliest = entry.get('earliest_pickup')
        if earliest is None:
            earliest = rt + probe_cfg['delta_pickup']
        latest = entry.get('latest_dropoff')
        if latest is None:
            latest = rt + probe_cfg['delta_dropoff']
        details[ep] = {
            'request_id': ep,
            'decision_epoch': ep,
            'request_time': rt,
            'pickup_node': probe_cfg['pickup'],
            'dropoff_node': probe_cfg['dropoff'],
            'earliest_pickup': earliest,
            'latest_dropoff': latest,
            'assigned_vehicle': entry.get('assigned_vehicle'),
            'closest_vehicle': entry.get('closest_vehicle'),
            'closest_distance': entry.get('closest_distance'),
            'chosen_vehicle_distance': entry.get('chosen_vehicle_distance'),
            'chosen_vehicle_pending': 0,
            'chosen_vehicle_completed': 0,
            'traffic_level': entry.get('traffic_level'),
            'vehicle_options': entry.get('vehicle_options', []),
        }
    env_data['assignment_details'] = details

    # Contract: every USER_STUDY_SELECTED_INDICES[CONTROLLED_ADAPTATION] epoch must
    # have a non-None entry. Fail loudly if the source pkl is missing a selected probe.
    selected = ParatransitConfig.USER_STUDY_SELECTED_INDICES.get(
        ScenarioType.CONTROLLED_ADAPTATION, [])
    missing = [ep for ep in selected if ep >= n_slots or details[ep] is None]
    if missing:
        raise ValueError(
            f"Case 0 source pkl is missing probe_results for selected epochs: {missing}. "
            f"Cannot build the probe-only view."
        )

    # 5) Cosmetic env_data patches (leave n_vehicles alone).
    env_data['n_requests'] = n_slots

    # 6) Patch mdp_comparisons so state_snapshot.state and event_info describe the
    #    probe at every selected epoch. Case 0 is probe-only: there is no pre/post-
    #    trigger distinction at the runtime view — the backbone Case-3 run is just
    #    how the pkl was produced, not part of Case 0's surface.
    mdp_comparisons = scenario_data.get('mdp_comparisons', {}) or {}

    for ep, comp in mdp_comparisons.items():
        if not isinstance(ep, int):
            continue
        probe_entry = probe_results.get(ep)
        if not probe_entry:
            continue
        state_snapshot = comp.setdefault('state_snapshot', {})
        state = state_snapshot.setdefault('state', {})
        rt = probe_entry.get('request_time', 0.0)
        state['current_request'] = {
            'request_id': 100000 + ep,
            'pickup_node': probe_cfg['pickup'],
            'dropoff_node': probe_cfg['dropoff'],
            'request_time': rt,
            'earliest_pickup': rt + probe_cfg['delta_pickup'],
            'latest_dropoff': rt + probe_cfg['delta_dropoff'],
        }
        state['vehicles'] = [
            {
                'vehicle_id': i,
                'current_location': probe_cfg['vehicle_locations'][i],
                'current_time': rt,
                'current_occupancy': 0,
                'capacity': 3,
                'route': [],
                'next_time': rt,
            }
            for i in range(len(probe_cfg['vehicle_locations']))
        ]
        state['traffic_level'] = probe_entry.get('traffic_level', state.get('traffic_level'))
        state_snapshot['state_obs'] = None

        event_info = comp.setdefault('event_info', {})
        event_info['mdp_t_assignment'] = probe_entry.get('assigned_vehicle')
        event_info['closest_vehicle'] = probe_entry.get('closest_vehicle')
        event_info['closest_distance'] = probe_entry.get('closest_distance')
        event_info['chosen_vehicle_distance'] = probe_entry.get('chosen_vehicle_distance')
        event_info['mdp_t_minus_n_assignment'] = None
        event_info.pop('assignment_changed', None)

        comp.pop('mdp_t_minus_n_tree', None)
        comp.pop('mdp_t_minus_n', None)
        comp.pop('comparison_results', None)

    # 7) Clear scenario-level cached comparison results. Case 0 intentionally does
    #    NOT precompute MDP_{t-n} here — the probe-patched state_snapshots are ready,
    #    and the standard query-time rebuild path populates comparisons lazily when a
    #    query references a specific post-trigger probe epoch.
    scenario_data['comparison_results'] = {}

    return mcts_data


def load_scenario0_final_data(use_saved=None):
    """Load the final Scenario 0 pkl and return the probe-only Case 0 runtime view.

    The on-disk pkl (`saved_scenario0_results.pkl`) is never mutated. Instead of
    deepcopying the whole pkl (which `_transform_to_case0` then throws most of
    away), we build a targeted Case 0 runtime dict: reference-share the read-only
    pieces (probe_results, probe_config, model_version_history, selected probe
    trees, env_data scalars) and deepcopy only the 10 mdp_comparisons entries
    that `_transform_to_case0` mutates in place.

    If the pkl is missing, returns None — Case 0 cannot regenerate the source
    pkl (candidate generator retired).
    """
    saved_file = 'saved_scenario0_results.pkl'
    loaded = _try_load_saved(saved_file, use_saved)
    if loaded is None or loaded[0] is None:
        print(f"✖ Case 0 requires {saved_file} in pkl_cache/. Not found.")
        return None

    src = loaded[0]
    src_sd = src.get('scenario_data', {}) or {}
    src_ed = src.get('env_data', {}) or {}
    src_trees = src.get('per_step_trees', {}) or {}
    selected = ParatransitConfig.USER_STUDY_SELECTED_INDICES.get(
        ScenarioType.CONTROLLED_ADAPTATION, [])

    mcts_data = {
        'scenario_type': ScenarioType.CONTROLLED_ADAPTATION,
        'scenario_data': {
            'type': ScenarioType.CONTROLLED_ADAPTATION,
            'probe_results': src_sd.get('probe_results', {}),
            'probe_config': src_sd.get('probe_config', {}),
            'model_version_history': src_sd.get('model_version_history', []),
            'trigger_epoch': src_sd.get('trigger_epoch'),
            'dpas_history': src_sd.get('dpas_history', {}),             # read-only
            # Only the 10 selected entries are deepcopied because _transform_to_case0
            # mutates state_snapshot / event_info in place. Non-selected entries are
            # intentionally omitted — Case 0 runtime never reads them.
            'mdp_comparisons': {
                ep: deepcopy(entry)
                for ep, entry in (src_sd.get('mdp_comparisons') or {}).items()
                if ep in selected
            },
            'comparison_results': {},
        },
        'env_data': {
            # Transform rewrites assignment_details wholesale; shallow-copy the rest.
            **{k: v for k, v in src_ed.items() if k != 'assignment_details'},
        },
        # Pass `probe|<ep>` keys so _transform_to_case0's re-keying loop at step 3
        # does its normal job (probe|<ep> → ep_1|<ep>). Entries are reference-shared
        # — tree dicts are read-only during XAI (serialized, never mutated).
        'per_step_trees': {
            f'probe|{ep}': src_trees[f'probe|{ep}']
            for ep in selected
            if f'probe|{ep}' in src_trees
        },
    }
    for k in ('timestamp', 'run_metadata', 'config'):
        if k in src:
            mcts_data[k] = src[k]

    try:
        _transform_to_case0(mcts_data)
    except Exception as e:
        print(f"✖ Case 0 transform failed: {e}")
        traceback.print_exc()
        return None
    return mcts_data


def display_paratransit_scenario(mcts_data):
    """
    Display the paratransit scenario information.
    
    Args:
        mcts_data: MCTS execution data
    """
    env_data = mcts_data.get('env_data', {})
    scenario_data = mcts_data.get('scenario_data', {})
    scenario_type = scenario_data.get('type', 'default')
    
    # Compute changed/unchanged epochs dynamically from comparison_results
    # Check both _runtime (query-time) and scenario_data (precompute-time)
    comparison_results = scenario_data.get('comparison_results', {})
    rt_comparison = mcts_data.get('_runtime', {}).get('comparison_results', {})
    merged_comparison = {**comparison_results, **rt_comparison}
    changed_epochs = []   # Assignment changed (MDP_{t-n} != MDP_t)
    unchanged_epochs = []  # Assignment unchanged (MDP_{t-n} == MDP_t)

    for epoch, result in merged_comparison.items():
        assignment_changed = result.get('assignment_changed')
        if assignment_changed is True:
            changed_epochs.append(epoch)
        elif assignment_changed is False:
            unchanged_epochs.append(epoch)
    
    changed_epochs = sorted(changed_epochs)
    unchanged_epochs = sorted(unchanged_epochs)

    print("\n" + "=" * 80)
    if scenario_type == ScenarioType.COUNTER_INTUITIVE:
        print("SCENARIO: COUNTER-INTUITIVE VEHICLE ASSIGNMENT (Case 1)")
        # Display bridge accident info
        bridge_epoch = scenario_data.get('bridge_accident_epoch')
        bridge_mult = scenario_data.get('bridge_accident_multiplier')
        if bridge_epoch is not None:
            print(f"  Bridge accident at epoch {bridge_epoch}: S→N travel time x{bridge_mult}")
    elif scenario_type == ScenarioType.EVENT_ASSIGNMENT_CHANGE:
        print("SCENARIO: EVENT-BASED NON-STATIONARITY ANALYSIS (Case 2)")
        if changed_epochs:
            print(f"  Epochs where assignment changed: {changed_epochs}")
        if unchanged_epochs:
            print(f"  Epochs where assignment unchanged: {unchanged_epochs}")
    elif scenario_type == ScenarioType.CITYWIDE_CONGESTION:
        print("SCENARIO: CITYWIDE CONGESTION ANALYSIS (Case 3)")
        congestion_config = scenario_data.get('congestion_config', {})
        cong_epoch = congestion_config.get('congestion_epoch')
        cong_level = congestion_config.get('congestion_traffic_level')
        base_level = congestion_config.get('base_traffic_level')
        if cong_epoch is not None:
            print(f"  Citywide congestion at epoch {cong_epoch}: traffic_level {base_level} -> {cong_level}")
        if changed_epochs:
            print(f"  Epochs where assignment changed: {changed_epochs}")
        if unchanged_epochs:
            print(f"  Epochs where assignment unchanged: {unchanged_epochs}")
    elif scenario_type == ScenarioType.CONTROLLED_ADAPTATION:
        # Case 0 banner — probe-only 10-request view.
        print("SCENARIO 0: CONTROLLED ADAPTATION — CASE 0 (10 selected probe epochs)")
        probe_config = scenario_data.get('probe_config', {})
        congestion_config = scenario_data.get('congestion_config', {})
        cong_epoch = congestion_config.get('congestion_epoch') or scenario_data.get('trigger_epoch')
        cong_level = congestion_config.get('congestion_traffic_level')
        base_level = congestion_config.get('base_traffic_level')
        if probe_config:
            print(f"  Probe: pickup {probe_config.get('pickup')} → dropoff {probe_config.get('dropoff')}")
            print(f"  Fixed idle fleet: {probe_config.get('vehicle_locations')}")
        if cong_epoch is not None and cong_level is not None:
            print(f"  Congestion at epoch {cong_epoch}: traffic_level {base_level} → {cong_level}")
        selected = ParatransitConfig.USER_STUDY_SELECTED_INDICES.get(
            ScenarioType.CONTROLLED_ADAPTATION, [])
        if len(selected) == 10:
            pre_map = [(i + 1, selected[i]) for i in range(3)]
            mid_map = [(i + 1, selected[i]) for i in range(3, 6)]
            post_map = [(i + 1, selected[i]) for i in range(6, 10)]
            print("  Selected requests (display → original epoch):")
            print("    pre:  " + ", ".join(f"{d}→{ep}" for d, ep in pre_map))
            print("    mid:  " + ", ".join(f"{d}→{ep}" for d, ep in mid_map))
            print("    post: " + ", ".join(f"{d}→{ep}" for d, ep in post_map))
    else:
        print("PARATRANSIT SCENARIO - VEHICLE ASSIGNMENT ANALYSIS")
    print("=" * 80)

    # Display event info if applicable
    if scenario_type == ScenarioType.EVENT_ASSIGNMENT_CHANGE:
        event_config = scenario_data.get('event_config', {})
        event_epoch = event_config.get('event_epoch', 'N/A')
        event_nodes = event_config.get('event_nodes', [])
        event_multiplier = event_config.get('event_multiplier', 'N/A')
        comparison_results = scenario_data.get('comparison_results', {})
        
        print(f"\nEVENT CONFIGURATION:")
        print(f"   Event occurred at epoch: {event_epoch}")
        print(f"   Affected nodes: {event_nodes}")
        print(f"   Travel time multiplier: {event_multiplier}x")
        print(f"   Note: Any route with pickup/dropoff at these nodes is affected")
        
        # Show per-epoch comparison results if available
        if comparison_results:
            print(f"\n   MDP Comparison Results (old BNN vs updated BNN):")
            
            for comp_epoch in sorted(comparison_results.keys()):
                result = comparison_results[comp_epoch]
                changed = result.get('assignment_changed')
                mdp_tn = result.get('mdp_t_minus_n_assignment')
                mdp_t = result.get('mdp_t_assignment')
                
                status = "CHANGED" if changed else "NO CHANGE"
                print(f"     Epoch {comp_epoch}: {status}")
                if mdp_tn is not None and mdp_t is not None:
                    print(f"       Old BNN → V{mdp_tn}, Updated BNN → V{mdp_t}")

    # Display congestion info if applicable
    if scenario_type == ScenarioType.CITYWIDE_CONGESTION:
        congestion_config = scenario_data.get('congestion_config', {})
        cong_epoch = congestion_config.get('congestion_epoch', 'N/A')
        cong_level = congestion_config.get('congestion_traffic_level', 'N/A')
        base_level = congestion_config.get('base_traffic_level', 'N/A')
        comparison_results = scenario_data.get('comparison_results', {})

        print(f"\nCONGESTION CONFIGURATION:")
        print(f"   Congestion begins at epoch: {cong_epoch}")
        print(f"   Traffic level: {base_level} -> {cong_level}")
        print(f"   Note: All routes affected (global travel time interpolation change)")

        if comparison_results:
            print(f"\n   MDP Comparison Results (old BNN vs updated BNN):")
            for comp_epoch in sorted(comparison_results.keys()):
                result = comparison_results[comp_epoch]
                changed = result.get('assignment_changed')
                mdp_tn = result.get('mdp_t_minus_n_assignment')
                mdp_t = result.get('mdp_t_assignment')
                status = "CHANGED" if changed else "NO CHANGE"
                print(f"     Epoch {comp_epoch}: {status}")
                if mdp_tn is not None and mdp_t is not None:
                    print(f"       Old BNN → V{mdp_tn}, Updated BNN → V{mdp_t}")

    # Case 0 — 10-row selected probe table with pre/mid/post phase separators.
    if scenario_type == ScenarioType.CONTROLLED_ADAPTATION:
        probe_results = scenario_data.get('probe_results', {})
        probe_config = scenario_data.get('probe_config', {})
        selected = ParatransitConfig.USER_STUDY_SELECTED_INDICES.get(
            ScenarioType.CONTROLLED_ADAPTATION, [])
        if probe_results and selected:
            print("\nPROBE ASSIGNMENT TABLE — 10 selected requests (same pickup/dropoff/fleet every epoch):")
            print(f"  Pickup: {probe_config.get('pickup')}  Dropoff: {probe_config.get('dropoff')}")
            print(f"  Δpickup: {probe_config.get('delta_pickup')} min  Δdropoff: {probe_config.get('delta_dropoff')} min")
            print("┌──────┬────────┬──────────┬──────────────┬──────────────┬────────────────┐")
            print("│ Req# │ Epoch  │ Traffic  │ Model ver.   │ Probe chose  │ Closest veh.   │")
            print("├──────┼────────┼──────────┼──────────────┼──────────────┼────────────────┤")
            phase_break_after = {3, 6}  # render separator after display 3 (pre|mid) and 6 (mid|post)
            for i, ep in enumerate(selected):
                display_id = i + 1
                entry = probe_results.get(ep, {})
                tl = entry.get('traffic_level', '?')
                mv = entry.get('model_version', '?')
                assigned_v = entry.get('assigned_vehicle')
                closest_v = entry.get('closest_vehicle')
                assigned_str = f"V{assigned_v}" if assigned_v is not None else "N/A"
                closest_str = f"V{closest_v}" if closest_v is not None else "N/A"
                print(f"│ {display_id:>4} │ {ep:>6} │ {tl!s:>8} │ {mv!s:>12} │ {assigned_str:>12} │ {closest_str:>14} │")
                if display_id in phase_break_after:
                    print("├──────┼────────┼──────────┼──────────────┼──────────────┼────────────────┤")
            print("└──────┴────────┴──────────┴──────────────┴──────────────┴────────────────┘")

    # Display detailed assignment table
    if 'assignment_details' in env_data and env_data['assignment_details']:
        details = env_data['assignment_details']
        ci_epochs = scenario_data.get('counter_intuitive_epochs', [])
        comparison_results = scenario_data.get('comparison_results', {})

        # Case 0: filter the 30-slot assignment_details list down to the 10 selected
        # probe epochs (list indexed by original epoch — see _transform_to_case0).
        if scenario_type == ScenarioType.CONTROLLED_ADAPTATION:
            selected = ParatransitConfig.USER_STUDY_SELECTED_INDICES.get(
                ScenarioType.CONTROLLED_ADAPTATION, [])
            details = [details[ep] for ep in selected
                       if 0 <= ep < len(details) and details[ep] is not None]

        # Common request epochs for cross-stage comparison (51→5733)
        common_request_epochs = [8, 14, 24]

        print("\nADA-MCTS Vehicle Assignment Decisions:")
        print("Note: Pend=pending requests; Done=completed requests")
        if scenario_type != ScenarioType.CONTROLLED_ADAPTATION:
            print(f"      * marks common request 51→5733 (epochs: {common_request_epochs})")

        # Enhanced table with request details (removed separate CI column)
        print("┌─────┬──────┬──────────────────────────────────────────┬───────────────────────────┬───────────────────────────┐")
        print("│ Req │ Time │  Request Details                         │  Closest Vehicle          │  Assigned Vehicle         │")
        print("│     │      │  Nodes(pickup→drop)  Time(early→late)    │  ID    Dist  Pend    Done │  ID    Dist  Pend    Done │")
        print("├─────┼──────┼──────────────────────────────────────────┼───────────────────────────┼───────────────────────────┤")

        # For Case 0, map each selected probe epoch to its 1..10 display id.
        case0_id_map = None
        if scenario_type == ScenarioType.CONTROLLED_ADAPTATION:
            case0_id_map = ParatransitConfig.get_request_id_mapping(
                ScenarioType.CONTROLLED_ADAPTATION) or {}

        for detail in details:
            original_req_id = detail.get('request_id', '?')
            decision_epoch = detail.get('decision_epoch', original_req_id)
            # Case 0: print 1..10 display id; for other scenarios keep the original id.
            if case0_id_map is not None and isinstance(original_req_id, int):
                req_id = case0_id_map.get(original_req_id, original_req_id)
            else:
                req_id = original_req_id

            # No filtering - show all requests
            
            assigned_veh = detail.get('assigned_vehicle', '?')
            closest_veh = detail.get('closest_vehicle')
            closest_dist = detail.get('closest_distance')
            chosen_dist = detail.get('chosen_vehicle_distance')
            request_time = detail.get('request_time', 0)

            # Use actual pending and completed recorded at assignment time
            chosen_pending = detail.get('chosen_vehicle_pending', 0)
            chosen_done = detail.get('chosen_vehicle_completed', 0)

            # Request details
            pickup_node = detail.get('pickup_node', '?')
            dropoff_node = detail.get('dropoff_node', '?')
            earliest_pickup = detail.get('earliest_pickup', 0)
            latest_dropoff = detail.get('latest_dropoff', 0)

            # Find closest vehicle info from vehicle_options
            closest_pending = 0
            closest_done = 0
            if 'vehicle_options' in detail and closest_veh is not None:
                for opt in detail['vehicle_options']:
                    if opt['vehicle_id'] == closest_veh:
                        closest_pending = opt.get('pending_requests', 0)
                        closest_done = opt.get('completed_requests', 0)
                        break

            # Marker: * for common request epochs or event-affected epochs.
            # Case 0 suppresses the 51→5733 common-request marker (not meaningful for probes).
            is_common = (scenario_type != ScenarioType.CONTROLLED_ADAPTATION
                         and original_req_id in common_request_epochs)
            is_event_affected = decision_epoch in changed_epochs or decision_epoch in unchanged_epochs

            if is_event_affected:
                req_str = f"*{req_id:>3} "
            elif is_common:
                req_str = f"*{req_id:>3} "
            else:
                req_str = f" {req_id:>3} "
            time_str = f" {request_time:4.0f} "

            # Request details section (41 chars)
            req_details = f"  {pickup_node:>5}→{dropoff_node:<5}        {earliest_pickup:>3.0f}→{latest_dropoff:<6.0f}           "

            # Closest vehicle section (27 chars)
            if closest_veh is not None:
                closest_id = f"V{closest_veh}"
                closest_dist_str = f"{closest_dist:5.1f}" if closest_dist is not None else "  N/A"
                closest_section = f"  {closest_id:<4}  {closest_dist_str:>5}  {closest_pending:>2}      {closest_done:>2}  "
            else:
                closest_section = "  N/A                       "

            # Assigned vehicle section (27 chars)
            assigned_id = f"V{assigned_veh}"
            chosen_dist_str = f"{chosen_dist:5.1f}" if chosen_dist is not None else "  N/A"
            assigned_section = f"  {assigned_id:<4}  {chosen_dist_str:>5}  {chosen_pending:>2}      {chosen_done:>2}  "

            print(f"│{req_str}│{time_str}│{req_details}│{closest_section}│{assigned_section}│")

        print("└─────┴──────┴──────────────────────────────────────────┴───────────────────────────┴───────────────────────────┘")

    # Display execution statistics. Case 0 is a curated probe-only view, so the
    # backbone-level cumulative reward / n_requests stats don't describe it meaningfully;
    # skip them.
    if scenario_type != ScenarioType.CONTROLLED_ADAPTATION:
        print(f"\nEXECUTION STATISTICS:")
        n_requests = env_data.get('n_requests', 0)
        total_reward = env_data.get('total_reward', 0.0)
        violations = env_data.get('violations', [])

        print(f"   Total requests assigned: {n_requests} requests")
        print(f"   Total cumulative reward: {total_reward:.2f}")
        print(f"   Average reward per request: {total_reward/n_requests:.2f}" if n_requests > 0 else "   Average reward: N/A")

        if violations:
            print(f"   Constraint violations: {len(violations)}")
        else:
            print(f"   Constraint violations: None")
    
    # Display scenario-specific information
    if scenario_type == ScenarioType.COUNTER_INTUITIVE:
        ci_epochs = scenario_data.get('counter_intuitive_epochs', [])
        if ci_epochs:
            print(f"\n   Counter-intuitive assignments (closest != assigned) at epochs: {ci_epochs}")
        print(f"   * Common request (51→5733) at epochs: [8, 14, 24]")
    elif scenario_type == ScenarioType.EVENT_ASSIGNMENT_CHANGE:
        if changed_epochs:
            print(f"\n   * Assignment CHANGED at epochs: {changed_epochs}")
            print(f"     Old BNN and updated BNN chose different vehicles")
        if unchanged_epochs:
            print(f"   * Assignment UNCHANGED at epochs: {unchanged_epochs}")
            print(f"     PCTL metrics shifted but same vehicle still optimal")
        print(f"   (MDP comparison data available for PCTL evaluation)")
    elif scenario_type == ScenarioType.CITYWIDE_CONGESTION:
        if changed_epochs:
            print(f"\n   * Assignment CHANGED at epochs: {changed_epochs}")
            print(f"     Old BNN and updated BNN chose different vehicles")
        if unchanged_epochs:
            print(f"   * Assignment UNCHANGED at epochs: {unchanged_epochs}")
            print(f"     Same vehicle still optimal despite congestion")
        print(f"   (MDP comparison data available for PCTL evaluation)")
    elif scenario_type == ScenarioType.CONTROLLED_ADAPTATION:
        # Case 0 — explicit pre / mid / post phase summary using the 10 selected probe epochs.
        probe_results = scenario_data.get('probe_results', {})
        selected = ParatransitConfig.USER_STUDY_SELECTED_INDICES.get(
            ScenarioType.CONTROLLED_ADAPTATION, [])
        phases = [
            ("Pre-change phase   (requests 1-3) ", selected[:3]),
            ("Mid / adaptation   (requests 4-6) ", selected[3:6]),
            ("Post-adaptation    (requests 7-10)", selected[6:10]),
        ]
        print()
        for label, epochs in phases:
            if not epochs:
                continue
            vehs = [probe_results.get(e, {}).get('assigned_vehicle') for e in epochs]
            vehs_str = [f"V{v}" if v is not None else "?" for v in vehs]
            print(f"   {label}  epochs {list(epochs)}  probe chose: {vehs_str}")

    print("=" * 80)
    
    # Note: MDP comparison details are available in scenario_data for PCTL evaluation
    # They are not displayed here - the explanation generation will use them


def save_mcts_results(mcts_data, reward, filename="saved_paratransit_mcts_results.pkl"):
    """Save ADA-MCTS results to file for later use."""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'use_cases', 'paratransit', 'pkl_cache')
    os.makedirs(cache_dir, exist_ok=True)
    save_path = os.path.join(cache_dir, filename)
    data = {
        'mcts_data': mcts_data,
        'reward': reward,
        'timestamp': datetime.now().isoformat(),
    }
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"\n[SAVE] MCTS results saved to: use_cases/paratransit/pkl_cache/{filename}")
    return save_path


def load_mcts_results(filename="saved_paratransit_mcts_results.pkl"):
    """Load previously saved ADA-MCTS results."""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'use_cases', 'paratransit', 'pkl_cache')
    load_path = os.path.join(cache_dir, filename)
    if not os.path.exists(load_path):
        return None

    with open(load_path, 'rb') as f:
        data = pickle.load(f)

    timestamp = data.get('timestamp', 'unknown')
    print(f"\n[LOAD] Loaded MCTS results from: use_cases/paratransit/pkl_cache/{filename}")
    print(f"   Timestamp: {timestamp}")
    return data['mcts_data'], data['reward']


def _try_load_saved(saved_file, use_saved):
    """Check for saved results, optionally prompt user, and load if appropriate.

    Args:
        saved_file: Filename in pkl_cache directory.
        use_saved: True to force load, False to skip, None to auto-detect/prompt.

    Returns:
        (mcts_data, reward) if loaded, else (None, None).
    """
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'use_cases', 'paratransit', 'pkl_cache')
    has_saved = os.path.exists(os.path.join(cache_dir, saved_file))

    if use_saved is None and has_saved:
        if not sys.stdin.isatty():
            print("\n[INFO] Found saved results. (Auto-loading in batch mode)")
            use_saved = True
        else:
            print("\n[INFO] Found saved results for this scenario.")
            choice = input("   Load saved results? (y/n, default=y): ").strip().lower()
            use_saved = (choice != 'n')
    elif use_saved is None:
        use_saved = False

    if use_saved and has_saved:
        print("\nLoading saved scenario results...")
        loaded = load_mcts_results(saved_file)
        if loaded is not None:
            return loaded
    return None, None


# Retired: _ensure_scenario0_pkls + _load_or_generate_scenario0 drove the candidate1/2/3
# scan (Cases 4/5/6) by running the shared Case-3 probe generator and splitting its
# output into per-candidate pkls. Case 0 now repurposes the existing
# saved_scenario0_results.pkl via load_scenario0_final_data + _transform_to_case0 and
# does NOT regenerate. Kept commented for history.
#
# def _ensure_scenario0_pkls(use_saved=None):
#     """Generate (or verify) all Scenario 0 candidate pkls.
#
#     Strict contract (per-call, not per-process):
#     - use_saved is True  → every candidate pkl must already exist, else raise.
#     - use_saved is False → always regenerate when called (rewrites all pkls).
#     - use_saved is None  → regenerate only if any pkl is missing.
#
#     All candidates in ``SCENARIO0_CANDIDATE_NAMES`` share ONE real Case-3 adaptation
#     run; probes for each are computed in the same pass. The shared result is
#     deep-copied and split into N singular-schema pkls (one per candidate).
#
#     Note: CLI batch callers are responsible for invoking this at most once per
#     batch — see ``run_interactive_demo``'s pre-loop hoist.
#     """
#     cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                              'use_cases', 'paratransit', 'pkl_cache')
#     paths = {
#         case_num: os.path.join(cache_dir, SCENARIO_REGISTRY[case_num]['saved_file'])
#         for case_num in SCENARIO0_CASES
#     }
#     missing = [p for p in paths.values() if not os.path.exists(p)]
#
#     if use_saved is True:
#         if missing:
#             raise FileNotFoundError(
#                 f"--use-saved requires all {len(SCENARIO0_CANDIDATE_NAMES)} Scenario 0 pkls "
#                 f"({list(SCENARIO0_CANDIDATE_NAMES)}) to exist; missing: " + ", ".join(missing)
#             )
#         return
#
#     if use_saved is None and not missing:
#         return
#
#     # Regenerate the shared run
#     probe_configs = {
#         SCENARIO0_CASE_TO_CANDIDATE[case_num]: dict(SCENARIO_REGISTRY[case_num]['probe_config'])
#         for case_num in SCENARIO0_CASES
#     }
#     shared_spec = SCENARIO_REGISTRY[SCENARIO0_CASES[0]]
#     runner_kwargs = dict(shared_spec['runner_kwargs'])
#     runner_kwargs['requests_csv_path'] = shared_spec['requests_csv']
#     runner_kwargs['probe_configs'] = probe_configs
#
#     print(
#         f"\n[SCENARIO 0] Launching shared real Case-3 run with "
#         f"{len(SCENARIO0_CANDIDATE_NAMES)} probes: {list(SCENARIO0_CANDIDATE_NAMES)}..."
#     )
#     shared_mcts_data = run_scenario0_scenario(**runner_kwargs)
#
#     # Split into two singular-schema pkls
#     os.makedirs(cache_dir, exist_ok=True)
#     for case_num in SCENARIO0_CASES:
#         candidate_name = SCENARIO0_CASE_TO_CANDIDATE[case_num]
#         spec = SCENARIO_REGISTRY[case_num]
#
#         candidate_data = deepcopy(shared_mcts_data)
#         sdata = candidate_data['scenario_data']
#
#         plural_results = sdata.pop('probe_results_by_candidate', {})
#         sdata['probe_results'] = plural_results.get(candidate_name, {})
#
#         plural_configs = sdata.pop('probe_configs', {})
#         sdata['probe_config'] = plural_configs.get(candidate_name, dict(spec['probe_config']))
#
#         # Re-key per_step_trees probe entries: drop the other candidate, strip prefix
#         new_trees = {}
#         prefix = f"probe|{candidate_name}|"
#         for tree_key, tree_val in candidate_data.get('per_step_trees', {}).items():
#             if tree_key.startswith("probe|"):
#                 if tree_key.startswith(prefix):
#                     new_trees[f"probe|{tree_key[len(prefix):]}"] = tree_val
#                 # else: belongs to another candidate — drop
#             else:
#                 new_trees[tree_key] = tree_val
#         candidate_data['per_step_trees'] = new_trees
#
#         # Also patch tree_key field inside probe_results to match the renamed keys
#         for ep, entry in sdata['probe_results'].items():
#             if isinstance(entry, dict) and 'tree_key' in entry:
#                 entry['tree_key'] = f"probe|{ep}"
#
#         # Run the scenario's standard precompute on the per-candidate copy
#         precompute = spec.get('precompute')
#         if precompute:
#             if isinstance(precompute, str):
#                 precompute = globals()[precompute]
#             print(f"\n[PRECOMPUTE] Building MDP comparisons for {spec['label']}...")
#             precompute(candidate_data)
#             mdp_comparisons = candidate_data.get('scenario_data', {}).get('mdp_comparisons', {})
#             for ep in mdp_comparisons:
#                 mdp_comparisons[ep].pop('mdp_t_minus_n_tree', None)
#
#         save_mcts_results(candidate_data, candidate_data.get('reward', 0.0), spec['saved_file'])
#
#
# def _load_or_generate_scenario0(case_num, use_saved=None):
#     """Scenario 0 bypass: ensure both pkls are fresh, then load the requested one."""
#     _ensure_scenario0_pkls(use_saved)
#     spec = SCENARIO_REGISTRY[case_num]
#     loaded = _try_load_saved(spec['saved_file'], use_saved=True)
#     if loaded is None or loaded[0] is None:
#         print(f"✖ Failed to load Scenario 0 pkl for case {case_num}")
#         return None
#     return loaded[0]


def _load_or_generate(case_num, use_saved=None):
    """Registry-driven template: try loading saved pkl, else generate + save.

    Returns mcts_data or None on failure.
    """
    # Case 0 uses a dedicated probe-only loader (no regeneration path).
    if case_num == 0:
        return load_scenario0_final_data(use_saved)

    spec = SCENARIO_REGISTRY[case_num]
    mcts_data, reward = _try_load_saved(spec['saved_file'], use_saved)

    if mcts_data is None:
        print(f"\nRunning {spec['label']}...")
        try:
            kwargs = dict(spec['runner_kwargs'])
            kwargs['requests_csv_path'] = spec['requests_csv']
            mcts_data = spec['runner'](**kwargs)
            reward = mcts_data.get('reward', 0.0)

            # Run optional post-generation precompute
            precompute = spec.get('precompute')
            if precompute:
                if isinstance(precompute, str):
                    precompute = globals()[precompute]
                print(f"\n[PRECOMPUTE] Building MDP comparisons for event epochs...")
                precompute(mcts_data)
                # Strip rebuild trees to reduce pkl size
                mdp_comparisons = mcts_data.get('scenario_data', {}).get('mdp_comparisons', {})
                for ep in mdp_comparisons:
                    mdp_comparisons[ep].pop('mdp_t_minus_n_tree', None)

            save_mcts_results(mcts_data, reward, spec['saved_file'])

        except Exception as e:
            print(f"✖ Scenario execution failed: {e}")
            traceback.print_exc()
            return None

    return mcts_data


def _run_scenario_demo(case_num, use_saved=None):
    """Registry-driven template: banner → load/generate → display → QA."""
    spec = SCENARIO_REGISTRY[case_num]

    print("\n" + "=" * 60)
    print(f"SCENARIO: {spec['label']}")
    print("=" * 60)
    for line in spec['banner']:
        print(line)

    mcts_data = _load_or_generate(case_num, use_saved)
    if mcts_data is None:
        return False

    display_paratransit_scenario(mcts_data)
    return run_interactive_qa(mcts_data)


# Backward-compatible thin wrappers (load_counter_intuitive_data is imported by tests)

def load_counter_intuitive_data(use_saved=None):
    """Load or generate Case 1 data."""
    return _load_or_generate(1, use_saved)


def run_counter_intuitive_scenario_demo(use_saved=None):
    """Run Case 1 demo."""
    return _run_scenario_demo(1, use_saved)


def run_interactive_qa(mcts_data):
    """Run interactive Q&A session with the given MCTS data."""
    if not sys.stdin.isatty():
        print("\n[INFO] Batch mode - skipping interactive Q&A session.")
        return True
    print("\nSTEP 2: Interactive Q&A Session")
    print("Type your questions below. Type 'exit', 'quit', or 'q' to end.")

    config = ParatransitConfig()
    prepare_runtime_artifacts(config, mcts_data)
    orchestrator = NSXAIOrchestrator(config)

    while True:
        try:
            query = input("\nYour question > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ['exit', 'quit', 'q', 'e']:
            break

        try:
            explanation = orchestrator.explain_query_with_ada_mcts(query, mcts_data)
            # Note: explanation is already printed inside explain_query_with_ada_mcts
        except Exception as e:
            print(f"\nError processing query: {e}\n")
            traceback.print_exc()
            continue

    print("\nExiting interactive session. Goodbye!")
    return True


def load_event_scenario_data(use_saved=None):
    """Load or generate Case 2 data."""
    return _load_or_generate(2, use_saved)


def run_event_scenario_demo(use_saved=None):
    """Run Case 2 demo."""
    return _run_scenario_demo(2, use_saved)


def load_congestion_scenario_data(use_saved=None):
    """Load or generate Case 3 data."""
    return _load_or_generate(3, use_saved)


def run_congestion_scenario_demo(use_saved=None):
    """Run Case 3 demo."""
    return _run_scenario_demo(3, use_saved)


def run_interactive_demo(use_saved=None, cases=None):
    """
    Run an interactive Q&A session after executing ADA-MCTS.

    Allows user to select which scenario to run.

    Args:
        use_saved: If True, load saved results. If False, run new. If None, ask user.
        cases: List of case numbers to run (e.g. [1, 2]). If None, prompt user.
    """
    # If cases specified via CLI, run them sequentially.
    # Case 0 has its own dedicated loader (no shared regeneration); no batch hoist needed.
    if cases:
        all_success = True
        for case_num in cases:
            spec = SCENARIO_REGISTRY[case_num]
            sep = '#' * 60
            print(f"\n{sep}")
            print(f"# Running: {spec['label']}")
            print(sep)
            if not _run_scenario_demo(case_num, use_saved):
                all_success = False
        return all_success

    # Interactive menu
    print("\n" + "=" * 60)
    print("NS-XAI PARATRANSIT DEMO")
    print("=" * 60)
    print("\nSelect a scenario to run:")
    for num, spec in sorted(SCENARIO_REGISTRY.items()):
        print(f"  {num}. {spec['label']}")
        for line in spec['banner']:
            print(f"     {line}")
        print()

    if sys.stdin.isatty():
        choice = input(f"Enter choice ({'/'.join(str(n) for n in SCENARIO_REGISTRY)}, default=0): ").strip()
    else:
        choice = "0"

    case_num = int(choice) if choice.isdigit() and int(choice) in SCENARIO_REGISTRY else 0
    return _run_scenario_demo(case_num, use_saved)


def parse_args():
    parser = argparse.ArgumentParser(description="NS-XAI Paratransit Demo")
    parser.add_argument(
        '--cases', nargs='+', type=int, choices=[0, 1, 2, 3],
        help='Case numbers to run (e.g. --cases 0 1 2 3). Runs sequentially.'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run all cases (0, 1, 2, 3) sequentially.'
    )
    parser.add_argument(
        '--scenario0', action='store_true',
        help='Run final Scenario 0 (Case 0, probe-only 10-request view).'
    )
    parser.add_argument(
        '--use-saved', action='store_true', default=False,
        help='Force load saved pkl results if available (skip prompt).'
    )
    parser.add_argument(
        '--no-saved', action='store_true', default=False,
        help='Force fresh run, ignore saved pkl results.'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Check environment setup
    if not check_environment_setup():
        print("\n✖ Environment check failed. Please fix setup issues before running.")
        sys.exit(1)

    # Determine use_saved
    if args.use_saved:
        use_saved = True
    elif args.no_saved:
        use_saved = False
    else:
        use_saved = None  # Auto-detect / prompt

    # Determine cases to run
    cases = None
    if args.all:
        cases = [0, 1, 2, 3]
    elif args.scenario0:
        cases = [0]
    elif args.cases:
        cases = args.cases

    # Run demo
    success = run_interactive_demo(use_saved=use_saved, cases=cases)
    sys.exit(0 if success else 1)
