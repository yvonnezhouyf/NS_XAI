"""
Formula Evaluator: Evaluate PCTL formulas and derived metrics on MCTS trees.
"""

import hashlib
import logging
import math
import os
import pickle
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d

# Add ADA-MCTS path for NSParatransitV0 import (needed for ETA computation)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ada_mcts_path = os.path.join(project_root, 'algo', 'ADA-MCTS-main')
if ada_mcts_path not in sys.path:
    sys.path.insert(0, ada_mcts_path)


# ============================================================================
# PARATRANSIT DERIVED METRIC CONSTANTS (hard-coded values)
# ============================================================================
PARATRANSIT_DERIVED_CONSTANTS = {
    "N_MIN": 3,                    # Minimum exploration steps before adaptation
    "N_INTERVAL": 2,               # Online update frequency constant
    "ASSIGNED_CAPACITY": 3,        # Per-vehicle capacity (NSParatransitV0 default)
    "EVENT_MULTIPLIER": 3.0,       # Congestion multiplier for event scenarios
    "ACCIDENT_MULTIPLIER": 2.0,    # Travel time multiplier for accident scenarios
    "EVENT_EPOCH": 10,             # Epoch at which event (congestion) occurs
    "ACCIDENT_EPOCH": 10,          # Epoch at which accident occurs
}

# Scenario-specific constants: metric_name -> tuple of allowed scenario_type prefixes.
# Used to filter out formulas that are not applicable to a given scenario.
SCENARIO_SPECIFIC_CONSTANTS = {
    'EVENT_MULTIPLIER': ('event_',),
    'EVENT_EPOCH': ('event_',),
    'ACCIDENT_MULTIPLIER': ('counter_intuitive',),
    'ACCIDENT_EPOCH': ('counter_intuitive',),
}


def filter_formulas_for_scenario(formulas, scenario_type):
    """Remove DERIVED formulas that are not applicable to the given scenario."""
    result = []
    for f in formulas:
        if f.startswith('DERIVED:'):
            metric = f.replace('DERIVED:', '').strip()
            prefixes = SCENARIO_SPECIFIC_CONSTANTS.get(metric)
            if prefixes and not any(scenario_type.startswith(p) for p in prefixes):
                continue
        result.append(f)
    return result


_MCTS_DATA = None  # Full MCTS data including per_step_trees
_ACTIVE_CONFIG = None  # Active domain config (ParatransitConfig)

# Cache for rebuilt MDP_{t-n} trees to avoid re-running MCTS
# Key: (run_id, epoch, prev_version_id) — scoped by run_id for multi-case coexistence
_MDP_REBUILD_CACHE: Dict[Tuple, Any] = {}

# Cache for PCTL formula evaluation results on trees (avoids re-computing deterministic results)
# Key: (run_id, epoch, tree_role, formula_str) — stable logical key
#   run_id: from mcts_data["_runtime"]["run_id"] (unique per session/load)
#   tree_role: "mdp_t" or "mdp_t_minus_n_{version_id}"
# Value: (results_dict, error_note) — the return value of _evaluate_formula_on_tree_children
_FORMULA_ON_TREE_CACHE: Dict[Tuple, Tuple] = {}

# Cache for ETA computation environments (to avoid reloading large CSV files)
# Key: (requests_csv_path, fixed_request_ids tuple, seed, traffic_condition,
#       event_epoch, event_nodes tuple, event_multiplier)
_ETA_ENV_CACHE: Dict[Tuple, Any] = {}

# Directory for MDP_{t-n} pkl cache
_MDP_TN_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'use_cases', 'paratransit', 'pkl_cache', 'mdp_tn'
)

def _get_run_id(mcts_data: Optional[Dict]) -> str:
    """Return the run_id from mcts_data['_runtime'], or a fallback based on id()."""
    if mcts_data is None:
        return '0'
    rt = mcts_data.get('_runtime')
    if rt and 'run_id' in rt:
        return rt['run_id']
    return str(id(mcts_data))


def clear_runtime_caches(run_id=None):
    """Clear formula and rebuild caches.

    Args:
        run_id: If given, only evict entries belonging to this run.
                If None, clear everything.
    """
    global _MDP_REBUILD_CACHE, _FORMULA_ON_TREE_CACHE
    if run_id is None:
        _MDP_REBUILD_CACHE = {}
        _FORMULA_ON_TREE_CACHE = {}
    else:
        _MDP_REBUILD_CACHE = {k: v for k, v in _MDP_REBUILD_CACHE.items() if k[0] != run_id}
        _FORMULA_ON_TREE_CACHE = {k: v for k, v in _FORMULA_ON_TREE_CACHE.items() if k[0] != run_id}

def set_active_config(config):
    """
    Set the active domain configuration for formula evaluation.

    Args:
        config: Domain config object (FrozenLakeConfig or ParatransitConfig)
    """
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = config

def set_evidence_cache(cache: Dict = None, execution_map: Optional[Dict] = None, mcts_data: Optional[Dict] = None):
    """
    Set the active MCTS data for formula evaluation.

    Args:
        cache: Unused (kept for backward compatibility)
        execution_map: Unused (kept for backward compatibility)
        mcts_data: Full MCTS data including per_step_trees (for tree-based PCTL)
    """
    global _MCTS_DATA
    _MCTS_DATA = mcts_data


# ============================================================================
# FORMULA VALIDATION HELPER
# ============================================================================

def _validate_and_filter_formulas(
    formulas: List[str],
    use_case: str
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Validate formulas against AP whitelist and return valid formulas + notes for invalid ones.

    Args:
        formulas: List of PCTL formulas to validate
        use_case: Use case name ('frozen_lake' or 'paratransit')

    Returns:
        Tuple of (valid_formulas, invalid_notes)
        - valid_formulas: List of formulas that passed validation
        - invalid_notes: List of {'note': ...} dicts for invalid formulas
    """
    valid_formulas = []
    invalid_notes = []

    for formula in formulas:
        # Skip DERIVED formulas (they don't have APs to validate)
        if formula.startswith('DERIVED:'):
            valid_formulas.append(formula)
            continue

        is_valid, invalid_props = validate_atomic_propositions(formula, use_case)
        if is_valid:
            valid_formulas.append(formula)
        else:
            invalid_notes.append({
                'note': f'Formula "{formula}" contains invalid atomic propositions: {invalid_props}'
            })

    return valid_formulas, invalid_notes


def evaluate_mdp_comparison(
    formulas: List[str],
    mcts_data: Dict,
    env,
    query: str,
    epoch: int,
    allow_rebuild: bool = True
) -> List[Dict[str, Any]]:
    """
    Evaluate PCTL formulas comparing MDP_t (current model) vs MDP_{t-n} (previous model).

    This is the core evaluation method for demonstrating non-stationarity in paratransit.
    It compares how the current model (after online adaptation) differs from an earlier
    model version for the same state.

    PCTL Evaluation:
    - Uses parse_property_formula() to parse formulas (supports F/G/X/U/F<=k, AND/OR/NOT)
    - Uses eval_property_formula_on_traces() with trace-based evaluation on node.traces
    - Returns proper probability/reward values for all PCTL operators

    Args:
        formulas: List of PCTL formulas to evaluate
        mcts_data: Full MCTS data with per_step_trees, scenario_data, etc.
        env: Environment adapter
        query: User query string
        epoch: Decision epoch to analyze
        allow_rebuild: If True, rebuild MDP_{t-n} on-demand when needed

    Returns:
        List of evaluation results, each with:
        {
            'formula': str,
            'result': {
                'mdp_t': {action: {pctl_value, visits, status}, ...},
                'mdp_t_minus_n': {action: {...}, ...} or None,
                'prev_version_id': int or None,
                'prev_epoch': int or None,
                'counter_intuitive_info': {...} or None,
                'event_info': {...} or None,
                'is_compare_epoch': bool
            }
        }
    """
    global _ACTIVE_CONFIG, _MDP_REBUILD_CACHE

    results = []
    per_step_trees = mcts_data.get('per_step_trees', {})
    scenario_data = mcts_data.get('scenario_data', {})

    # Get tree for this epoch (MDP_t)
    step_key = f"ep_1|{epoch}"
    tree_t = per_step_trees.get(step_key)

    # Build tree node if it's still a dict
    if tree_t is not None and isinstance(tree_t, dict):
        if _ACTIVE_CONFIG and hasattr(_ACTIVE_CONFIG, 'build_tree_from_data'):
            tree_t = _ACTIVE_CONFIG.build_tree_from_data(tree_t)
        else:
            tree_t = None  # Can't build tree without config

    # Get action naming from config
    n_actions = 5  # Default for paratransit
    if _ACTIVE_CONFIG and hasattr(_ACTIVE_CONFIG, 'get_n_actions'):
        n_actions = _ACTIVE_CONFIG.get_n_actions(mcts_data)

    def action_name(a: int) -> str:
        if _ACTIVE_CONFIG and hasattr(_ACTIVE_CONFIG, 'action_to_name'):
            return _ACTIVE_CONFIG.action_to_name(a)
        return f"V{a}"

    # Check if this is a counter-intuitive epoch with comparison data
    mdp_comparisons = scenario_data.get('mdp_comparisons', {})
    model_version_history = scenario_data.get('model_version_history', {})

    ci_info = None
    comparison_data = mdp_comparisons.get(epoch)
    if comparison_data:
        ci_info = comparison_data.get('counter_intuitive_info')

    # Validate formulas against AP whitelist
    use_case = _ACTIVE_CONFIG.USE_CASE_NAME if _ACTIVE_CONFIG and hasattr(_ACTIVE_CONFIG, 'USE_CASE_NAME') else 'paratransit'
    validated_formulas, invalid_notes = _validate_and_filter_formulas(formulas, use_case)
    results.extend(invalid_notes)

    # Separate PCTL and DERIVED formulas
    pctl_formulas = [f for f in validated_formulas if not f.startswith('DERIVED:')]
    derived_formulas = [f for f in validated_formulas if f.startswith('DERIVED:')]

    # Pre-filter derived formulas by scenario type (e.g. skip EVENT_EPOCH for counter_intuitive)
    scenario_type_str = scenario_data.get('type', '')
    derived_formulas = filter_formulas_for_scenario(derived_formulas, scenario_type_str)

    # Early exit only if no formulas at all
    if not pctl_formulas and not derived_formulas:
        if results:  # has invalid_notes
            return results
        return [{'note': 'No PCTL formulas to evaluate'}]

    # Pre-compute MDP_{t-n} tree if needed (shared across all formulas)
    # Now available for ALL epochs since all epochs have state_snapshot
    tree_t_minus_n = None
    prev_version_id = None
    prev_epoch = None
    mdp_t_minus_n_assignment = None  # Best action from old BNN

    if comparison_data:
        state_snapshot = comparison_data.get('state_snapshot', {})

        # Use pre-computed prev_version_id from snapshot (falls back to scan for legacy pkl)
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

        if prev_version_id is not None and prev_version_id in model_version_history:
            prev_epoch = model_version_history[prev_version_id].get('start_epoch', 0)
        else:
            prev_epoch = None
            
        run_id = _get_run_id(mcts_data)

        if prev_version_id is not None:
            cache_key = (run_id, epoch, prev_version_id)
            if cache_key in _MDP_REBUILD_CACHE:
                tree_t_minus_n = _MDP_REBUILD_CACHE[cache_key]
            elif allow_rebuild:
                # Warn if this is user-study mode (rebuild should not happen after warmup)
                rt = mcts_data.get('_runtime', {}) if mcts_data else {}
                if rt.get('warn_on_runtime_rebuild'):
                    prewarmed = rt.get('prewarmed_epochs', set())
                    scenario_type = mcts_data.get('scenario_data', {}).get('type', 'unknown') if mcts_data else 'unknown'
                    if epoch in prewarmed:
                        print(f"[WARNING] Unexpected runtime rebuild after warmup: "
                              f"run_id={run_id}, epoch={epoch}, prev_version={prev_version_id}, "
                              f"scenario={scenario_type} (epoch IS in prewarmed_epochs)", flush=True)
                    else:
                        print(f"[WARNING] Runtime rebuild for non-prewarmed epoch: "
                              f"run_id={run_id}, epoch={epoch}, prev_version={prev_version_id}, "
                              f"scenario={scenario_type} (epoch NOT in prewarmed_epochs)", flush=True)

                # Rebuild and cache (with pkl persistence for cross-run reuse)
                tree_dict = _rebuild_mdp_for_comparison(
                    state_snapshot, model_version_history, prev_version_id, mcts_data
                )
                if tree_dict is not None:
                    # Build node from dict
                    if _ACTIVE_CONFIG and hasattr(_ACTIVE_CONFIG, 'build_tree_from_data'):
                        tree_t_minus_n = _ACTIVE_CONFIG.build_tree_from_data(tree_dict)
                    if tree_t_minus_n is not None:
                        _MDP_REBUILD_CACHE[cache_key] = tree_t_minus_n

            # FIX #1: Compute mdp_t_minus_n_assignment (best action from old BNN)
            # This enables proper "old BNN vs updated BNN" comparison at same epoch
            if tree_t_minus_n is not None:
                mdp_t_minus_n_assignment = _compute_best_action_from_tree(tree_t_minus_n)

                # Compute true assignment_changed and store results
                event_info = comparison_data.get('event_info') if comparison_data else None
                if event_info:
                    mdp_t_assignment = event_info.get('mdp_t_assignment')
                    if mdp_t_assignment is not None:
                        true_assignment_changed = (mdp_t_assignment != mdp_t_minus_n_assignment)

                        comp_result = {
                            'assignment_changed': true_assignment_changed,
                            'mdp_t_minus_n_assignment': mdp_t_minus_n_assignment,
                            'mdp_t_assignment': mdp_t_assignment,
                            'comparison_epoch': epoch,
                        }

                        # Write to _runtime if available (pure query, no scenario_data mutation)
                        rt = mcts_data.get('_runtime')
                        if rt is not None:
                            rt.setdefault('comparison_results', {})[epoch] = comp_result
                        else:
                            # Fallback: write to scenario_data (precompute path)
                            scenario_data.setdefault('comparison_results', {})[epoch] = comp_result

                        # Write enriched event_info to _runtime (no scenario_data mutation)
                        updated_event_info = dict(event_info)
                        updated_event_info['mdp_t_minus_n_assignment'] = mdp_t_minus_n_assignment
                        updated_event_info['assignment_changed'] = true_assignment_changed
                        if rt is not None:
                            rt.setdefault('enriched_event_info', {})[epoch] = updated_event_info
                            # Invalidate cached env_context for this epoch so it picks up enriched info
                            rt.get('env_context_by_epoch', {}).pop(epoch, None)
                        else:
                            # Precompute path: write to scenario_data
                            mdp_comp_epoch = scenario_data.setdefault('mdp_comparisons', {}).setdefault(epoch, {})
                            mdp_comp_epoch['event_info'] = updated_event_info

                        print(f"  [MDP Rebuild @ epoch {epoch}] True assignment_changed: {true_assignment_changed} "
                              f"(old BNN→V{mdp_t_minus_n_assignment} vs updated BNN→V{mdp_t_assignment})")

    # Evaluate each PCTL formula (requires tree data)
    # Build stable cache key prefixes (run_id, epoch, tree_role)
    rt = mcts_data.get('_runtime', {})
    run_id = rt.get('run_id')
    mdp_t_cache_prefix = (run_id, epoch, 'mdp_t') if run_id else None
    tn_role = f'mdp_t_minus_n_{prev_version_id}' if prev_version_id is not None else 'mdp_t_minus_n'
    mdp_tn_cache_prefix = (run_id, epoch, tn_role) if run_id else None

    if tree_t is None:
        # No tree data - PCTL formulas can't be evaluated
        for formula in pctl_formulas:
            results.append({
                'note': f'Formula "{formula}" requires tree data (no tree for epoch {epoch})'
            })
    else:
        for formula in pctl_formulas:
            # Evaluate on MDP_t (current model)
            mdp_t_results, mdp_t_error = _evaluate_formula_on_tree_children(
                formula, tree_t, action_name, cache_key_prefix=mdp_t_cache_prefix)

            # Evaluate on MDP_{t-n} if available
            mdp_t_minus_n_results = None
            mdp_t_minus_n_error = None
            if tree_t_minus_n is not None:
                mdp_t_minus_n_results, mdp_t_minus_n_error = _evaluate_formula_on_tree_children(
                    formula, tree_t_minus_n, action_name, cache_key_prefix=mdp_tn_cache_prefix)

            # If both evaluations failed, surface as note
            if mdp_t_error and (tree_t_minus_n is None or mdp_t_minus_n_error):
                results.append({
                    'note': f'Formula "{formula}" evaluation failed: {mdp_t_error}'
                })
                continue

            result_dict = {
                'mdp_t': mdp_t_results,
                'mdp_t_minus_n': mdp_t_minus_n_results,
                'prev_version_id': prev_version_id,
                'prev_epoch': prev_epoch,
                'is_compare_epoch': comparison_data is not None,
            }

            # Include error info if one side failed but not both
            if mdp_t_error:
                result_dict['mdp_t_error'] = mdp_t_error
            if mdp_t_minus_n_error:
                result_dict['mdp_t_minus_n_error'] = mdp_t_minus_n_error

            if ci_info:
                result_dict['counter_intuitive_info'] = ci_info

            # Include event info if available, enhanced with mdp_t_minus_n_assignment
            event_info = comparison_data.get('event_info') if comparison_data else None
            if event_info:
                # FIX #1 & #3: Create enriched event_info with old BNN assignment
                enriched_event_info = dict(event_info)  # Copy original
                if mdp_t_minus_n_assignment is not None:
                    enriched_event_info['mdp_t_minus_n_assignment'] = mdp_t_minus_n_assignment
                    # Compute assignment_changed: same epoch, old BNN vs updated BNN
                    mdp_t_assignment = event_info.get('mdp_t_assignment')
                    if mdp_t_assignment is not None:
                        enriched_event_info['assignment_changed'] = (mdp_t_assignment != mdp_t_minus_n_assignment)
                result_dict['event_info'] = enriched_event_info

            results.append({
                'formula': formula,
                'result': result_dict
            })

    # Compute DERIVED formulas on both MDP_t and MDP_{t-n} for comparison
    for derived_formula in derived_formulas:
        # Compute on MDP_t (current model)
        derived_result_t = compute_paratransit_derived_metrics(
            derived_formula, mcts_data, env, epoch, tree_t, comparison_data,
            override_assigned_vehicle=None  # Use actual MDP_t assignment
        )

        # Compute on MDP_{t-n} if available
        derived_result_t_minus_n = None
        if comparison_data:  # Can compute derived even without tree_t_minus_n
            # Create comparison_data for MDP_{t-n} with the old model's assignment
            tn_comparison_data = dict(comparison_data)
            tn_comparison_data['is_mdp_t_minus_n'] = True
            # Pass old model's assignment for proper comparison
            derived_result_t_minus_n = compute_paratransit_derived_metrics(
                derived_formula, mcts_data, env, epoch, tree_t_minus_n, tn_comparison_data,
                override_assigned_vehicle=mdp_t_minus_n_assignment
            )

        if derived_result_t is not None or derived_result_t_minus_n is not None:
            result_dict = {
                'mdp_t': derived_result_t,
                'mdp_t_minus_n': derived_result_t_minus_n,
                'prev_version_id': prev_version_id,
                'prev_epoch': prev_epoch,
                'is_compare_epoch': comparison_data is not None,
            }
            if ci_info:
                result_dict['counter_intuitive_info'] = ci_info
            # Include event info
            event_info = comparison_data.get('event_info') if comparison_data else None
            if event_info:
                enriched_event_info = dict(event_info)
                if mdp_t_minus_n_assignment is not None:
                    enriched_event_info['mdp_t_minus_n_assignment'] = mdp_t_minus_n_assignment
                    mdp_t_assignment = event_info.get('mdp_t_assignment')
                    if mdp_t_assignment is not None:
                        enriched_event_info['assignment_changed'] = (mdp_t_assignment != mdp_t_minus_n_assignment)
                result_dict['event_info'] = enriched_event_info

            results.append({
                'formula': derived_formula,
                'result': result_dict
            })
        else:
            results.append({
                'note': f'{derived_formula} could not be computed (missing data)'
            })

    return results


def _evaluate_formula_on_tree_children(
    formula: str,
    tree_root,
    action_name_func,
    cache_key_prefix: Optional[Tuple] = None
) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """
    Evaluate PCTL formula on all action children of a tree root.

    Uses trace-based PCTL evaluation (parse_property_formula + eval_property_formula).

    Args:
        formula: PCTL formula string
        tree_root: MCTS tree root node
        action_name_func: Function to convert action int to name string
        cache_key_prefix: Optional stable prefix (run_id, epoch, tree_role) for cache key.
                          Falls back to id(tree_root) if not provided.

    Returns:
        Tuple of (results_dict, error_note)
        - results_dict: Dict mapping action names to evaluation results
        - error_note: String error message if all evaluations failed, None otherwise
    """
    global _FORMULA_ON_TREE_CACHE

    # Cache lookup using stable logical key when available
    if cache_key_prefix is not None:
        cache_key = cache_key_prefix + (formula,)
    else:
        cache_key = (id(tree_root), formula)
    if cache_key in _FORMULA_ON_TREE_CACHE:
        return _FORMULA_ON_TREE_CACHE[cache_key]

    results = {}
    children = getattr(tree_root, 'children', []) or []
    errors = []

    for child in children:
        action = getattr(child, 'action', None)
        if action is not None and isinstance(action, int):
            eval_result = _evaluate_pctl_policy_conditioned(formula, child)
            results[action_name_func(action)] = eval_result

            # Track errors
            status = eval_result.get('status', '')
            if status in ('parse_error', 'error'):
                errors.append(eval_result.get('error', status))

    # If all evaluations failed with the same error, surface it as a note
    if errors and len(errors) == len(results):
        # All failed - get unique error messages
        unique_errors = list(set(errors))
        if len(unique_errors) == 1:
            result = (results, unique_errors[0])
            _FORMULA_ON_TREE_CACHE[cache_key] = result
            return result
        result = (results, f"Multiple errors: {'; '.join(unique_errors[:3])}")
        _FORMULA_ON_TREE_CACHE[cache_key] = result
        return result

    result = (results, None)
    _FORMULA_ON_TREE_CACHE[cache_key] = result
    return result


def _evaluate_pctl_policy_conditioned(formula: str, action_node) -> Dict[str, Any]:
    """
    Evaluate PCTL formula on an action node using policy-conditioned tree walk.

    Computes P(formula | first action=a, then planner policy) exactly by
    recursively walking the MCTS subtree:
    - At each internal node, weight children by visit proportion N(child)/N(parent)
      (planner policy at decision nodes, transition probability at chance nodes)
    - At leaf nodes, evaluate the formula on (prefix_trace + each rollout trace)
    - Combine via weighted average: P = Σ (path_prob * leaf_satisfaction_rate)

    This replaces the old rollout-trace approach which computed
    P(formula | first action=a, then random rollout), producing
    indistinguishable results across vehicles.

    Args:
        formula: PCTL formula string
        action_node: MCTS child node representing an action (root's child)

    Returns:
        Dict with pctl_value, visits, status, etc.
    """
    visits = getattr(action_node, 'visits', 0)

    try:
        parsed = parse_property_formula(formula)
    except ValueError as e:
        return {'pctl_value': 0.0, 'visits': visits, 'status': 'parse_error', 'error': f'Parse error: {e}'}

    try:
        # Recursively compute policy-conditioned value and evidence coverage
        missing_ap_counter = [0]
        raw_value, evidence_mass = _pctl_tree_walk(
            parsed, action_node, prefix_trace=[], _missing_ap_counter=missing_ap_counter)

        if missing_ap_counter[0] > 0:
            logger.warning(
                "PCTL tree walk encountered %d nodes with missing transition_ap "
                "for formula: %s", missing_ap_counter[0], formula)

        # Compute final pctl_value: value normalized by evidence mass
        if evidence_mass > 0:
            pctl_value = raw_value / evidence_mass
        else:
            pctl_value = 0.0

        result = {
            'pctl_value': pctl_value,
            'visits': visits,
            'status': 'evaluated',
        }

        # Apply threshold comparison for P>=p or P<=p formulas
        if parsed.kind == 'P' and parsed.cmp is not None and parsed.threshold is not None:
            if parsed.cmp == '>=':
                result['threshold_met'] = (pctl_value >= parsed.threshold)
            elif parsed.cmp == '<=':
                result['threshold_met'] = (pctl_value <= parsed.threshold)

        return result

    except Exception as e:
        return {'pctl_value': 0.0, 'visits': visits, 'status': 'error', 'error': str(e)}


def _pctl_tree_walk(parsed_formula, node, prefix_trace: List[Dict],
                    _missing_ap_counter: Optional[List[int]] = None) -> Tuple[float, float]:
    """
    Recursively walk the MCTS subtree to compute policy-conditioned PCTL value.

    Returns (value, evidence_mass) where:
    - value: weighted PCTL value (probability for P, expected steps for R)
    - evidence_mass: fraction of tree weight backed by actual trace evidence (0..1)

    At each internal node:
      value = Σ_child w_i * child_value
      mass  = Σ_child w_i * child_mass
      where w_i = N_child / N_total

    Visit proportion N_child/N_total serves as:
    - Planner policy π(a|s) at decision nodes (which action MCTS prefers)
    - Transition probability P(s'|s,a) at chance nodes (how environment responds)
    Note: for chance nodes this is a Monte Carlo approximation of the true
    transition probability. With sufficient MCTS iterations it converges, but
    could be replaced with exact BNN transition probabilities in the future.

    At leaf nodes with evidence: (satisfaction_rate_or_expected_steps, 1.0)
    At leaf nodes without evidence: (0.0, 0.0)

    When transition_ap is missing on a node, a placeholder step with _missing_ap=True
    is inserted to preserve step count. Traces containing such steps evaluate to
    unknown (NaN) and contribute evidence_mass=0.

    Args:
        parsed_formula: Parsed PCTL formula (PropertyFormula object)
        node: Current tree node
        prefix_trace: Accumulated trace from the action node down to this node
        _missing_ap_counter: Mutable list [count] for aggregating missing_ap warnings

    Returns:
        Tuple[float, float] — (value, evidence_mass)
    """
    if _missing_ap_counter is None:
        _missing_ap_counter = [0]

    children = getattr(node, 'children', []) or []

    if not children:
        # Leaf node: evaluate formula on prefix + rollout traces
        return _eval_formula_at_leaf(parsed_formula, node, prefix_trace)

    # Internal node: weighted average over children by visit proportion
    visits = np.array([max(getattr(c, 'visits', 0), 0) for c in children], dtype=np.float64)
    total = visits.sum()

    if total == 0:
        # No visits on any child — treat as leaf
        return _eval_formula_at_leaf(parsed_formula, node, prefix_trace)

    weighted_value = 0.0
    weighted_mass = 0.0
    for i, child in enumerate(children):
        if visits[i] == 0:
            continue  # Skip unvisited branches

        # Extend prefix trace with child's transition_ap if applicable.
        # Only decision nodes carry transition_ap (state transition semantics).
        # Chance nodes represent action choices — they don't produce a state
        # transition step and should NOT contribute to the trace.
        child_type = getattr(child, 'type', None)
        child_ap = _collect_transition_ap(child)
        if child_ap:
            child_prefix = prefix_trace + [child_ap]
        elif child_type == 'decision':
            # Decision node missing transition_ap — insert placeholder to
            # preserve step count; _missing_ap=True marks this step as unknown
            child_prefix = prefix_trace + [{'_missing_ap': True}]
            _missing_ap_counter[0] += 1
        else:
            # Chance node (or unknown type): no trace step expected
            child_prefix = prefix_trace

        weight = visits[i] / total
        child_value, child_mass = _pctl_tree_walk(
            parsed_formula, child, child_prefix, _missing_ap_counter)
        weighted_value += weight * child_value
        weighted_mass += weight * child_mass

    return weighted_value, weighted_mass


def _eval_formula_at_leaf(parsed_formula, leaf_node,
                          prefix_trace: List[Dict]) -> Tuple[float, float]:
    """
    Evaluate PCTL formula at a leaf node using prefix trace + leaf's rollout traces.

    Returns (value, evidence_mass):
    - Leaf with evidence (rollout traces or non-empty prefix): (result, 1.0)
    - Leaf without any evidence: (0.0, 0.0)
    - Traces containing _missing_ap steps evaluate to NaN and are excluded

    Note: rollout traces use random policy (not planner policy). This is an
    approximation — the tree walk from root to this leaf already captures the
    planner's policy via visit-weighted branching; the rollout traces serve as
    a continuation estimate beyond the tree frontier.

    Args:
        parsed_formula: Parsed PCTL formula
        leaf_node: Leaf tree node
        prefix_trace: Trace accumulated from the action node to this leaf

    Returns:
        Tuple[float, float] — (value, evidence_mass)
    """
    # Get rollout traces from this leaf
    rollout_traces = getattr(leaf_node, 'rollout_traces', None)
    if rollout_traces is None:
        rollout_traces = getattr(leaf_node, 'traces', None)
    if rollout_traces is None:
        rollout_traces = []

    is_reward = (parsed_formula.kind == 'R')

    if not rollout_traces:
        # No rollout data — evaluate on prefix trace alone
        if not prefix_trace:
            # No evidence at all
            return (0.0, 0.0)
        result = _eval_single_trace(parsed_formula, prefix_trace)
        if math.isnan(result):
            return (0.0, 0.0)
        return (result, 1.0)

    # Evaluate on each (prefix + rollout_suffix) combination
    # Traces that return NaN (due to _missing_ap) are excluded
    valid_count = 0
    accumulated = 0.0
    for rt in rollout_traces:
        full_trace = prefix_trace + list(rt)
        result = _eval_single_trace(parsed_formula, full_trace)
        if math.isnan(result):
            continue
        valid_count += 1
        if is_reward:
            accumulated += result  # step count
        else:
            accumulated += 1.0 if result else 0.0  # satisfaction

    if valid_count == 0:
        return (0.0, 0.0)

    n_total = len(rollout_traces)
    # Normalize by TOTAL trace count (not valid_count) to avoid double-normalization.
    # The top-level _evaluate_pctl_policy_conditioned divides value by evidence_mass
    # to get the final estimate: (accumulated/n_total) / (valid_count/n_total)
    # = accumulated/valid_count = correct average over evidence-backed traces.
    value = accumulated / n_total
    evidence_mass = valid_count / n_total
    return (value, evidence_mass)


def _eval_single_trace(parsed_formula, trace: List[Dict]) -> float:
    """
    Evaluate a parsed PCTL formula on a single trace.

    For P formulas: returns 1.0 if satisfied, 0.0 if not.
    For R{"steps"} formulas: returns the step index at which expr is first
    satisfied (0-indexed), or len(trace) if never satisfied.

    Args:
        parsed_formula: Parsed PropertyFormula object
        trace: Single trace (list of AP dicts)

    Returns:
        float — satisfaction (0/1) for P formulas, step count for R formulas.
        Returns float('nan') if trace contains _missing_ap steps (unknown evidence).
    """
    if not trace:
        return 0.0

    # If any step in the trace has _missing_ap, the trace is unreliable
    if any(step.get('_missing_ap', False) for step in trace):
        return float('nan')

    # R{"steps"}=? [F expr] — return number of steps to reach expr
    if parsed_formula.kind == 'R' and parsed_formula.reward_struct == 'steps':
        if parsed_formula.pattern in ('F', 'F_bounded'):
            # F<=k means "within k steps", i.e. check steps 0..k (bound+1 steps)
            max_steps = (parsed_formula.bound + 1) if parsed_formula.pattern == 'F_bounded' else len(trace)
            for step_i in range(min(max_steps, len(trace))):
                if _eval_expr_at_step(parsed_formula.expr, trace, step_i):
                    return float(step_i)
            # Not reached — return trace length as penalty
            return float(len(trace))
        # For other R patterns (G, X, U), fall through to P-style evaluation
        # and return 0/1 (not ideal, but these R-pattern combos are rare)

    if parsed_formula.pattern == 'F':
        return 1.0 if _eval_F_on_trace(parsed_formula.expr, trace) else 0.0
    elif parsed_formula.pattern == 'F_bounded':
        return 1.0 if _eval_F_on_trace(parsed_formula.expr, trace, parsed_formula.bound) else 0.0
    elif parsed_formula.pattern == 'G':
        return 1.0 if _eval_G_on_trace(parsed_formula.expr, trace) else 0.0
    elif parsed_formula.pattern == 'X':
        return 1.0 if _eval_X_on_trace(parsed_formula.expr, trace) else 0.0
    elif parsed_formula.pattern == 'U':
        return 1.0 if _eval_U_on_trace(parsed_formula.expr[0], parsed_formula.expr[1], trace) else 0.0

    return 0.0


def _collect_transition_ap(node) -> Optional[Dict[str, bool]]:
    """
    Collect atomic propositions from a node for one trace step.

    Uses transition_ap only (captures (parent_state, action, this_state) transition semantics).
    Does NOT fall back to _cached_props — that captures state-level APs which would
    conflate step semantics in the trace.

    Args:
        node: MCTS tree node

    Returns:
        Dict of {ap_name: bool} or None if no transition_ap available
    """
    tap = getattr(node, 'transition_ap', None)
    if tap:
        return dict(tap)

    return None


def _compute_best_action_from_tree(tree_root) -> Optional[int]:
    """
    Compute the best action from a rebuilt MCTS tree based on visit counts.

    This follows the same action selection rule as MCTS: choose the action
    with the highest visit count (most explored = most confident).

    Args:
        tree_root: MCTS tree root node (AdaptedMCTSNode or similar)

    Returns:
        Best action (int) or None if no children
    """
    children = getattr(tree_root, 'children', []) or []
    if not children:
        return None

    best_action = None
    best_visits = -1

    for child in children:
        action = getattr(child, 'action', None)
        visits = getattr(child, 'visits', 0)

        if action is not None and isinstance(action, int) and visits > best_visits:
            best_visits = visits
            best_action = action

    return best_action


def _generate_mdp_tn_cache_filename(
    state_snapshot: Dict,
    target_version: int,
    epoch: int,
    scenario_type: str = "counter_intuitive",
    max_iterations: int = 3000
) -> str:
    """
    Generate a unique cache filename for MDP_{t-n} based on:
    - scenario_type (e.g., "counter_intuitive", "event_assignment_change", etc.)
    - seed
    - fixed_request_ids or original_request_ids
    - n_requests, n_vehicles
    - traffic_condition
    - max_iterations (REBUILD_MCTS_ITERATIONS)
    - epoch
    - target_version (t-n)
    - event config (event_epoch, event_nodes, event_multiplier) for Case 2 & 3

    Returns:
        Filename like: mdp_tn_{scenario}_{sig}_ep{epoch}_v{version}_iter{iterations}.pkl
    """
    env_params = state_snapshot.get('env_params', {})
    env_history = state_snapshot.get('env_history', {})

    seed = env_params.get('seed', 0)
    n_requests = env_params.get('n_requests', 0)
    n_vehicles = env_params.get('n_vehicles', 0)
    traffic_condition = env_params.get('traffic_condition', 0.3)

    # Event config for Case 2 & 3 (node-based congestion)
    event_epoch = env_params.get('event_epoch')
    event_nodes = env_params.get('event_nodes', [])
    event_multiplier = env_params.get('event_multiplier', 3.0)
    
    # Build event signature
    if event_epoch is not None and event_nodes:
        event_nodes_str = ','.join(str(n) for n in sorted(event_nodes))
        event_sig = f"event_{event_epoch}|{event_nodes_str}|{event_multiplier}"
    else:
        event_sig = "no_event"

    # Use fixed_request_ids if available, otherwise use original_request_ids
    fixed_ids = env_params.get('fixed_request_ids')
    if fixed_ids is None:
        fixed_ids = env_history.get('original_request_ids', [])

    # Evaluator version tag: invalidates old caches when PCTL semantics change
    # pc_policy_v1 = policy-conditioned tree walk (replaces rollout-trace evaluation)
    # pc_policy_v2 = removed _cached_props fallback, added P>=/<= threshold + R{"steps"} support
    # pc_policy_v3 = evidence_mass tracking, missing_ap handling, F<=k bound fix
    # pc_policy_v4 = fix chance node false _missing_ap, fix double normalization
    evaluator_version = "pc_policy_v4"
    tree_depth = 8  # Must match serialize_node_for_json max_depth

    # Create signature from key parameters (includes evaluator version + depth)
    max_time = env_params.get('max_time', 480)
    sig_str = f"{seed}|{sorted(fixed_ids) if fixed_ids else 'none'}|{n_requests}|{n_vehicles}|{traffic_condition}|{max_iterations}|{event_sig}|{evaluator_version}|d{tree_depth}|mt{max_time}"
    sig_hash = hashlib.md5(sig_str.encode()).hexdigest()[:8]

    # Short scenario name for filename
    scenario_short = scenario_type[:8] if len(scenario_type) > 8 else scenario_type

    return f"mdp_tn_{scenario_short}_{sig_hash}_ep{epoch}_v{target_version}.pkl"


def _rebuild_mdp_for_comparison(
    state_snapshot: Dict,
    model_version_history: Dict,
    target_version: int,
    mcts_data: Optional[Dict] = None
) -> Optional[Any]:
    """
    Rebuild MCTS tree using old model weights for MDP_{t-n} comparison.

    This is the lazy rebuild mechanism - only triggered when user queries
    about a specific counter-intuitive epoch.

    Includes pkl caching to avoid repeated expensive rebuilds:
    1. Check if tree is already in mcts_data (in-memory cache from this run)
    2. Check if pkl cache file exists
    3. If not, rebuild and save to both caches

    Uses config values for rebuild iterations and verbosity.

    Args:
        state_snapshot: State snapshot from CI detection
        model_version_history: Dict of model versions with weights
        target_version: Which version to rebuild with
        mcts_data: Optional MCTS data for additional context

    Returns:
        tree_dict (serialized MCTS tree) or None if rebuild fails
    """
    global _ACTIVE_CONFIG, _MDP_TN_CACHE_DIR

    if target_version not in model_version_history:
        return None

    epoch = state_snapshot.get('epoch', 0)

    # Compute epoch gap: how many epochs back does this version correspond to
    target_start_epoch = model_version_history[target_version].get('start_epoch', 0)
    epoch_gap = epoch - target_start_epoch

    # Get rebuild config from active config (needed for cache filename)
    max_iterations = 3000
    verbose = True
    if _ACTIVE_CONFIG:
        max_iterations = getattr(_ACTIVE_CONFIG, 'REBUILD_MCTS_ITERATIONS', 3000)
        verbose = getattr(_ACTIVE_CONFIG, 'REBUILD_VERBOSE', True)

    # Get scenario type from mcts_data for cache filename differentiation
    scenario_type = "counter_intuitive"  # Default
    if mcts_data is not None:
        scenario_data = mcts_data.get('scenario_data', {})
        scenario_type = scenario_data.get('type', 'counter_intuitive')

    # ========================================================================
    # Step 1: Check in-memory cache in mcts_data (avoid re-reading pkl)
    # ========================================================================
    if mcts_data is not None:
        scenario_data = mcts_data.get('scenario_data', {})
        mdp_comparisons = scenario_data.get('mdp_comparisons', {})
        comparison_data = mdp_comparisons.get(epoch, {})

        cached_tree = comparison_data.get('mdp_t_minus_n_tree')
        if cached_tree is not None:
            print(f"[CACHE] Using in-memory cached MDP_{{t-n}} for epoch {epoch} (n={epoch_gap} epochs, version={target_version})")
            return cached_tree

    # ========================================================================
    # Step 2: Check pkl file cache
    # ========================================================================
    # Normalize event scenario types to share cache
    # Case 1 (counter_intuitive) keeps its own separate cache
    cache_scenario_type = scenario_type
    if scenario_type and scenario_type.startswith("event_"):
        cache_scenario_type = "event_shared"
    
    cache_filename = _generate_mdp_tn_cache_filename(
        state_snapshot, target_version, epoch,
        scenario_type=cache_scenario_type,
        max_iterations=max_iterations
    )
    cache_path = os.path.join(_MDP_TN_CACHE_DIR, cache_filename)

    if os.path.exists(cache_path):
        # Ask user if they want to reload (only in interactive mode)
        use_cached = True
        if sys.stdin.isatty() and not os.environ.get('NS_XAI_NON_INTERACTIVE'):
            print(f"\n[INFO] Found cached MDP_{{t-n}} for epoch {epoch} (n={epoch_gap} epochs, version={target_version})")
            print(f"       File: {cache_filename}")
            choice = input("       Load from cache? (y/n, default=y): ").strip().lower()
            use_cached = (choice != 'n')

        if use_cached:
            try:
                with open(cache_path, 'rb') as f:
                    tree_dict = pickle.load(f)
                print(f"[CACHE] Loaded MDP_{{t-n}} from pkl cache (n={epoch_gap} epochs, version={target_version})")

                # Also store in mcts_data for subsequent queries in this run
                if mcts_data is not None:
                    _store_tree_in_mcts_data(mcts_data, epoch, tree_dict)

                return tree_dict
            except Exception as e:
                print(f"[WARNING] Failed to load cache file: {e}")
                # Continue to rebuild

    # ========================================================================
    # Step 3: Rebuild MDP_{t-n}
    # ========================================================================
    version_info = model_version_history[target_version]

    # Get weights for this version
    network_weights = version_info.get('network_weights')
    latent_weights = version_info.get('latent_weights')
    baseline_network_weights = version_info.get('baseline_network_weights')
    baseline_latent_weights = version_info.get('baseline_latent_weights')

    if network_weights is None or latent_weights is None:
        return None

    # Note: max_iterations and verbose already retrieved above for cache filename

    try:
        # Import rebuild function from adamcts_runner
        from use_cases.paratransit.adamcts_runner import rebuild_mcts_with_old_weights

        # Use explicit None checks to avoid numpy array truth value errors
        base_net = baseline_network_weights if baseline_network_weights is not None else network_weights
        base_latent = baseline_latent_weights if baseline_latent_weights is not None else latent_weights

        tree_dict = rebuild_mcts_with_old_weights(
            state_snapshot,
            network_weights,
            latent_weights,
            base_net,
            base_latent,
            max_iterations=max_iterations,
            verbose=verbose
        )

        if tree_dict is None:
            return None

        # ====================================================================
        # Step 4: Save to both caches
        # ====================================================================
        # Save to pkl file
        try:
            os.makedirs(_MDP_TN_CACHE_DIR, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(tree_dict, f)
            print(f"[CACHE] Saved MDP_{{t-n}} to: {cache_filename} (n={epoch_gap} epochs, version={target_version})")
        except Exception as e:
            print(f"[WARNING] Failed to save cache file: {e}")

        # Save to mcts_data for subsequent queries in this run
        if mcts_data is not None:
            _store_tree_in_mcts_data(mcts_data, epoch, tree_dict)

        return tree_dict

    except Exception as e:
        print(f"[WARNING] MDP rebuild failed: {e}")
        traceback.print_exc()
        return None


def _store_tree_in_mcts_data(mcts_data: Dict, epoch: int, tree_dict: Any) -> None:
    """Store rebuilt tree in mcts_data for in-memory caching."""
    try:
        if 'scenario_data' not in mcts_data:
            mcts_data['scenario_data'] = {}
        if 'mdp_comparisons' not in mcts_data['scenario_data']:
            mcts_data['scenario_data']['mdp_comparisons'] = {}
        if epoch not in mcts_data['scenario_data']['mdp_comparisons']:
            mcts_data['scenario_data']['mdp_comparisons'][epoch] = {}

        mcts_data['scenario_data']['mdp_comparisons'][epoch]['mdp_t_minus_n_tree'] = tree_dict
    except Exception:
        pass  # Non-critical, just skip in-memory caching


# ============================================================================
# TURNING_INTERVAL: Recovery Interval Detection
# ============================================================================

def _map_to_nearest_selected_floor(epoch: int, selected_indices: List[int]) -> int:
    """Map epoch to nearest selected index <= epoch (floor), or nearest overall."""
    candidates = [s for s in selected_indices if s <= epoch]
    if candidates:
        return max(candidates)
    return min(selected_indices, key=lambda s: abs(s - epoch))


def _map_to_nearest_selected_ceil(epoch: int, selected_indices: List[int]) -> int:
    """Map epoch to nearest selected index >= epoch (ceil), or nearest overall."""
    candidates = [s for s in selected_indices if s >= epoch]
    if candidates:
        return min(candidates)
    return min(selected_indices, key=lambda s: abs(s - epoch))


def _compute_turning_interval(mcts_data: Dict, current_epoch: int,
                              scenario_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Find the recovery interval where ADA-MCTS shifts from mostly pessimistic
    to mostly regular sampling after the scenario change (epoch 10).

    Returns an interval (pair of consecutive epochs) rather than a single
    curvature point, with request-ID and user-study display mappings.
    """
    scenario_data = mcts_data.get('scenario_data', {})
    dpas_history = scenario_data.get('dpas_history') or mcts_data.get('env_data', {}).get('dpas_history', {})

    if not dpas_history:
        return None

    epochs = sorted(dpas_history.keys())
    if len(epochs) < 3:
        return {'turning_points': [], 'note': 'Need at least 3 epochs'}

    regular_pcts = [dpas_history[e].get('regular_pct', 0.5) for e in epochs]

    # Compute smoothed curve for visualization
    y = np.array(regular_pcts, dtype=float)
    y_smooth = gaussian_filter1d(y, sigma=max(1.0, len(epochs) / 10), mode='nearest')
    smoothed_curve = [{'epoch': e, 'raw': regular_pcts[i], 'smoothed': round(y_smooth[i], 4)}
                      for i, e in enumerate(epochs)]

    # Scenario change epoch is always 10 for all paratransit cases
    event_epoch = 10

    def find_recovery_interval(idx_list, threshold=0.50):
        """Find first consecutive epoch pair where regular_pct crosses threshold."""
        if len(idx_list) < 2:
            return None
        # Primary: first threshold crossing (below -> above)
        for j in range(len(idx_list) - 1):
            i_cur, i_nxt = idx_list[j], idx_list[j + 1]
            if regular_pcts[i_cur] < threshold and regular_pcts[i_nxt] >= threshold:
                return {
                    'start_epoch': epochs[i_cur],
                    'end_epoch': epochs[i_nxt],
                    'start_regular_pct': round(regular_pcts[i_cur], 4),
                    'end_regular_pct': round(regular_pcts[i_nxt], 4),
                    'delta_regular_pct': round(regular_pcts[i_nxt] - regular_pcts[i_cur], 4),
                }
        # Fallback: largest single-step increase
        best = None
        best_delta = -float('inf')
        for j in range(len(idx_list) - 1):
            i_cur, i_nxt = idx_list[j], idx_list[j + 1]
            delta = regular_pcts[i_nxt] - regular_pcts[i_cur]
            if delta > best_delta:
                best_delta = delta
                best = (i_cur, i_nxt)
        if best and best_delta > 0:
            i_cur, i_nxt = best
            return {
                'start_epoch': epochs[i_cur],
                'end_epoch': epochs[i_nxt],
                'start_regular_pct': round(regular_pcts[i_cur], 4),
                'end_regular_pct': round(regular_pcts[i_nxt], 4),
                'delta_regular_pct': round(best_delta, 4),
            }
        return None

    # Focus on post-change epochs for recovery detection
    post_idx = [i for i, e in enumerate(epochs) if e >= event_epoch]
    recovery = find_recovery_interval(post_idx)

    # Convert epochs to request IDs (1-based: request_id = epoch + 1)
    recovery_request_interval = None
    if recovery:
        recovery_request_interval = [recovery['start_epoch'] + 1, recovery['end_epoch'] + 1]

    # Map to user-study display IDs with nearest-selected fallback
    display_interval = None
    if recovery and scenario_type:
        from use_cases.paratransit.config import ParatransitConfig
        selected = ParatransitConfig.USER_STUDY_SELECTED_INDICES.get(scenario_type)
        id_map = ParatransitConfig.get_request_id_mapping(scenario_type)
        if selected is not None and id_map is not None:
            nearest_start = _map_to_nearest_selected_floor(recovery['start_epoch'], selected)
            nearest_end = _map_to_nearest_selected_ceil(recovery['end_epoch'], selected)
            display_interval = [id_map[nearest_start], id_map[nearest_end]]

    return {
        'recovery_interval': recovery,
        'recovery_request_interval': recovery_request_interval,
        'display_interval': display_interval,
        'event_epoch': event_epoch,
        'smoothed_curve': smoothed_curve,
        'turning_points': [],  # backward compat
    }


# ============================================================================
# PARATRANSIT DERIVED METRICS COMPUTATION
# ============================================================================

def compute_paratransit_derived_metrics(
    formula: str,
    mcts_data: Dict,
    env,
    epoch: int,
    tree_root=None,
    comparison_data: Optional[Dict] = None,
    override_assigned_vehicle: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Compute paratransit-specific derived metrics.

    Supported metrics:
    - DERIVED: N_MIN                     -> hard-coded constant (3)
    - DERIVED: N_INTERVAL                -> hard-coded constant (2)
    - DERIVED: ASSIGNED_CAPACITY         -> hard-coded constant (3)
    - DERIVED: EVENT_MULTIPLIER          -> hard-coded constant (3.0)
    - DERIVED: ACCIDENT_MULTIPLIER       -> hard-coded constant (2.0)
    - DERIVED: EVENT_EPOCH               -> hard-coded constant (10)
    - DERIVED: ACCIDENT_EPOCH            -> hard-coded constant (10)
    - DERIVED: PENDING_REQUESTS_BY_VEHICLE -> pending count per vehicle {"V0": int, ...}
    - DERIVED: ETA_PICKUP_BY_VEHICLE     -> BNN-predicted pickup ETA per vehicle {"V0": float, ...}
    - DERIVED: ETA_DROPOFF_BY_VEHICLE    -> BNN-predicted dropoff ETA per vehicle {"V0": float, ...}
    - DERIVED: TRAFFIC_LEVEL             -> current traffic level at this epoch
    - DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE -> minutes to clear route per vehicle {"V0": float, ...}
    - DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE -> deadhead minutes per vehicle {"V0": float, ...}
    - DERIVED: DROPOFF_SLACK_BY_VEHICLE  -> dropoff slack per vehicle {"V0": float, ...}
    - DERIVED: SERVICE_PATH_DISRUPTION_BY_VEHICLE -> disruption per vehicle {"V0": str, ...}

    Args:
        formula: DERIVED formula string (e.g., "DERIVED: TRAFFIC_LEVEL")
        mcts_data: Full MCTS data with scenario_data, per_step_trees, etc.
        env: Environment adapter (ParatransitEnvAdapter or NSParatransitV0)
        epoch: Decision epoch to analyze
        tree_root: Optional MCTS tree root for this epoch
        comparison_data: Optional comparison data from mdp_comparisons[epoch]
        override_assigned_vehicle: Optional vehicle ID to use instead of the one from comparison_data
                                   (used for MDP_{t-n} to use old model's assignment)

    Returns:
        Dict with metric result, or None if metric cannot be computed
    """
    # Extract metric name from formula
    if not formula.startswith('DERIVED:'):
        return None
    metric_name = formula.replace('DERIVED:', '').strip()

    # ========================================================================
    # Hard-coded constants
    # ========================================================================
    if metric_name in PARATRANSIT_DERIVED_CONSTANTS:
        # EVENT_MULTIPLIER, EVENT_EPOCH, ACCIDENT_MULTIPLIER, ACCIDENT_EPOCH are
        # scenario-specific — only return them for the relevant scenario type.
        scenario_type = mcts_data.get('scenario_data', {}).get('type', '') if mcts_data else ''
        allowed_prefixes = SCENARIO_SPECIFIC_CONSTANTS.get(metric_name)
        if allowed_prefixes is not None:
            if not any(scenario_type.startswith(p) for p in allowed_prefixes):
                return None  # Not applicable to this scenario
        return {
            'value': PARATRANSIT_DERIVED_CONSTANTS[metric_name],
            'type': 'constant'
        }

    # ========================================================================
    # TURNING_INTERVAL: Find recovery interval after scenario change
    # ========================================================================
    if metric_name == 'TURNING_INTERVAL':
        scenario_type = mcts_data.get('scenario_data', {}).get('type')
        return _compute_turning_interval(mcts_data, epoch, scenario_type=scenario_type)

    if metric_name == 'STEP_CONFIDENCE':
        # Compute per-vehicle confidence based on visit share
        # Confidence for vehicle V_i = visits(V_i) / total_visits
        if tree_root is not None:
            children = getattr(tree_root, 'children', []) or []
            if children:
                total_visits = sum(getattr(c, 'visits', 0) for c in children)
                if total_visits > 0:
                    per_vehicle_confidence = {}
                    for child in children:
                        action = getattr(child, 'action', None)
                        visits = getattr(child, 'visits', 0)
                        if action is not None:
                            v_name = f"V{action}"
                            per_vehicle_confidence[v_name] = round(visits / total_visits, 4)
                    
                    return per_vehicle_confidence
        return None

    # ========================================================================
    # Dynamic metrics - need state/env data
    # ========================================================================
    scenario_data = mcts_data.get('scenario_data', {})
    env_data = mcts_data.get('env_data', {})

    # Get state from comparison_data (all epochs now have state_snapshot)
    state = None
    state_snapshot = None
    if comparison_data:
        state_snapshot = comparison_data.get('state_snapshot', {})
        state = state_snapshot.get('state')

    # Get assignment info
    mdp_comparisons = scenario_data.get('mdp_comparisons', {})
    epoch_comparison = mdp_comparisons.get(epoch, comparison_data or {})

    # Extract CI or event info
    ci_info = epoch_comparison.get('counter_intuitive_info', {})
    event_info = epoch_comparison.get('event_info', {})
    mdp_t_info = epoch_comparison.get('mdp_t', {})

    # Get assigned and closest vehicle IDs
    assigned_vehicle_id = None
    closest_vehicle_id = None

    if ci_info:
        assigned_vehicle_id = ci_info.get('assigned_vehicle')
        closest_vehicle_id = ci_info.get('closest_vehicle')
    elif event_info:
        assigned_vehicle_id = event_info.get('mdp_t_assignment')
        closest_vehicle_id = event_info.get('closest_vehicle')  # May be None for event scenarios
    elif mdp_t_info:
        assignment_info = mdp_t_info.get('assignment_info', {})
        assigned_vehicle_id = assignment_info.get('assigned_vehicle')
        closest_vehicle_id = assignment_info.get('closest_vehicle')

    # Override assigned_vehicle_id if specified (for MDP_{t-n} comparison)
    if override_assigned_vehicle is not None:
        assigned_vehicle_id = override_assigned_vehicle

    # ========================================================================
    # DERIVED METRICS CLASSIFICATION:
    # - MODEL-BASED (from BNN predictions): VEHICLE_OCCUPANCY, ETA_* metrics
    # - STATE-BASED (from real state): ASSIGNED_PENDING_REQUESTS, 
    #                                  CLOSEST_PENDING_REQUESTS, TRAFFIC_LEVEL
    # ========================================================================

    # Helper: Get BNN predictions for all vehicles from tree
    def _get_bnn_predictions_for_all_vehicles():
        """Get BNN-predicted states for all vehicles from tree's chance nodes."""
        if tree_root is None:
            return None
        
        children = getattr(tree_root, 'children', []) or []
        all_predictions = {}
        
        for child in children:
            action = getattr(child, 'action', None)
            if action is None:
                continue
            explain = getattr(child, 'explain', None)
            if explain is None or 'bnn_predicted_states' not in explain:
                continue
            bnn_states = explain['bnn_predicted_states']
            model_predictions = bnn_states.get('mdp_t', {})
            if model_predictions:
                all_predictions[action] = model_predictions
                break  # All chance nodes should have same predictions
        
        return all_predictions.get(list(all_predictions.keys())[0]) if all_predictions else None

    # ========================================================================
    # VEHICLE_OCCUPANCY: BNN-predicted occupancy for each vehicle
    # Returns model's prediction of vehicle workload after assignment
    # ========================================================================
    if metric_name == 'VEHICLE_OCCUPANCY':
        bnn_predictions = _get_bnn_predictions_for_all_vehicles()
        if bnn_predictions is None:
            return None  # No fallback - must have BNN predictions
        
        vehicle_occupancies = {}
        for v_key, v_data in bnn_predictions.items():
            occ = v_data.get('predicted_occupancy', 0)
            vehicle_occupancies[v_key] = occ
        return vehicle_occupancies

    # ========================================================================
    # PENDING_REQUESTS_BY_VEHICLE: Real route from state (STATE-BASED)
    # Count unique request_ids in each vehicle's route at decision time
    # ========================================================================
    if metric_name == 'PENDING_REQUESTS_BY_VEHICLE':
        if state is None:
            return None

        vehicles = state.get('vehicles', []) if isinstance(state, dict) else getattr(state, 'vehicles', [])
        if not vehicles:
            return None

        result = {}
        for v_idx, vehicle in enumerate(vehicles):
            route = vehicle.get('route', []) if isinstance(vehicle, dict) else getattr(vehicle, 'route', [])
            unique_request_ids = set(
                entry[0] for entry in route
                if isinstance(entry, (list, tuple)) and len(entry) >= 1
            )
            result[f"V{v_idx}"] = len(unique_request_ids)
        return result

    # ========================================================================
    # TRAFFIC_LEVEL: Real traffic_level from state (STATE-BASED)
    # Use the actual traffic_condition from state or env_data history
    # ========================================================================
    if metric_name == 'TRAFFIC_LEVEL':
        # Try state.traffic_level first (preferred, supports both object and dict)
        if state is not None:
            if isinstance(state, dict):
                traffic_level = state.get('traffic_level')
            else:
                traffic_level = getattr(state, 'traffic_level', None)
            if traffic_level is not None:
                return round(traffic_level, 3)
        
        # Fallback: env_data.traffic_level_history[epoch]
        if env_data:
            traffic_history = env_data.get('traffic_level_history', {})
            if epoch in traffic_history:
                return round(traffic_history[epoch], 3)
        
        return None

    # ========================================================================
    # ETA_PICKUP_BY_VEHICLE / ETA_DROPOFF_BY_VEHICLE: MODEL-BASED using BNN
    # Compute for all vehicles. Returns {"V0": float, "V1": float, ...}
    # ========================================================================
    if metric_name in ('ETA_PICKUP_BY_VEHICLE', 'ETA_DROPOFF_BY_VEHICLE'):
        return _compute_model_based_eta_all_vehicles(
            metric_name, state, epoch, tree_root, state_snapshot
        )

    # ========================================================================
    # Path metrics BY_VEHICLE (STATE-BASED, deterministic travel times)
    # Compute for all vehicles. Returns {"V0": float/str, "V1": float/str, ...}
    # ========================================================================
    _PATH_METRICS_BY_VEHICLE = {
        'CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE': 'CLEAR_CURRENT_ROUTE_TIME',
        'DEADHEAD_TO_PICKUP_BY_VEHICLE': 'DEADHEAD_TO_PICKUP',
        'DROPOFF_SLACK_BY_VEHICLE': 'DROPOFF_SLACK',
        'SERVICE_PATH_DISRUPTION_BY_VEHICLE': 'SERVICE_PATH_DISRUPTION',
    }
    if metric_name in _PATH_METRICS_BY_VEHICLE:
        base_metric = _PATH_METRICS_BY_VEHICLE[metric_name]
        return _compute_path_metric_all_vehicles(
            base_metric, state, env, epoch, state_snapshot, mcts_data
        )

    return None


def _compute_assignment_path_derived(
    metric_name: str,
    state,
    env,
    epoch: int,
    vehicle_id: Optional[int],
    state_snapshot: Optional[Dict],
    mcts_data: Optional[Dict],
) -> Optional[str]:
    """
    Compute assignment-path derived metrics using the environment's
    _compute_assignment_path_stats() helper.

    Called for both assigned and closest vehicle sides (the caller
    passes the appropriate vehicle_id).

    Metrics (base names — CLOSEST variants use the same logic):
        CLEAR_CURRENT_ROUTE_TIME: minutes to clear existing route (0.0 if idle)
        DEADHEAD_TO_PICKUP: minutes from route end / current loc to pickup
        DROPOFF_SLACK: deadline - eta_dropoff (negative = late)
        SERVICE_PATH_DISRUPTION: disruption type string (none/event/bridge/event+bridge)

    Returns:
        Formatted string like "12.34 (V0)" or "event+bridge (V0)", or None.
    """
    if vehicle_id is None or state is None:
        return None

    # We need an NSParatransitV0 instance to call _compute_assignment_path_stats.
    # Reuse the same ETA env cache (already loaded for ETA metrics).
    eta_env = _get_eta_env(state_snapshot, epoch)
    if eta_env is None:
        return None

    # Reconstruct a minimal ParatransitState from state dict/object
    try:
        ps = _reconstruct_paratransit_state(state, epoch)
    except Exception:
        return None

    if vehicle_id >= len(ps.vehicles):
        return None

    # Align epoch so event/bridge multipliers fire correctly
    saved_epoch = eta_env.current_decision_epoch
    eta_env.current_decision_epoch = epoch

    try:
        stats = eta_env._compute_assignment_path_stats(ps, vehicle_id)
    except Exception:
        return None
    finally:
        eta_env.current_decision_epoch = saved_epoch

    v_label = f"V{vehicle_id}"

    if metric_name == 'CLEAR_CURRENT_ROUTE_TIME':
        return f"{stats['clear_current_route_time']:.2f} ({v_label})"

    if metric_name == 'DEADHEAD_TO_PICKUP':
        return f"{stats['deadhead_to_pickup']:.2f} ({v_label})"

    if metric_name == 'DROPOFF_SLACK':
        return f"{stats['dropoff_slack']:.2f} ({v_label})"

    if metric_name == 'SERVICE_PATH_DISRUPTION':
        parts = []
        if stats['service_path_event_affected']:
            parts.append('event')
        if stats['service_path_bridge_affected']:
            parts.append('bridge')
        label = '+'.join(parts) if parts else 'none'
        return f"{label} ({v_label})"

    return None


def _reconstruct_paratransit_state(state, epoch: int):
    """
    Build a lightweight ParatransitState from a dict or object so that
    _compute_assignment_path_stats can operate on it.
    """
    from nsparatransit.nsparatransit_v0 import ParatransitState, VehicleState, PassengerRequest

    if isinstance(state, ParatransitState):
        return state

    # state is a dict from state_snapshot
    vehicles_raw = state.get('vehicles', [])
    vehicles = []
    for v in vehicles_raw:
        if isinstance(v, VehicleState):
            vehicles.append(v)
        elif isinstance(v, dict):
            vs = VehicleState(
                vehicle_id=v.get('vehicle_id', 0),
                current_location=v.get('current_location', 0),
                current_time=v.get('current_time', 0.0),
                current_occupancy=v.get('current_occupancy', 0),
                capacity=v.get('capacity', 3),
                route=v.get('route', []),
                next_time=v.get('next_time', v.get('current_time', 0.0)),
            )
            vehicles.append(vs)

    req_raw = state.get('current_request')
    if req_raw is None:
        raise ValueError("No current_request in state")
    if isinstance(req_raw, PassengerRequest):
        request = req_raw
    elif isinstance(req_raw, dict):
        request = PassengerRequest(
            request_id=req_raw.get('request_id', 0),
            pickup_node=req_raw.get('pickup_node', 0),
            dropoff_node=req_raw.get('dropoff_node', 0),
            request_time=req_raw.get('request_time', 0.0),
            earliest_pickup=req_raw.get('earliest_pickup', 0.0),
            latest_dropoff=req_raw.get('latest_dropoff', 0.0),
            original_id=req_raw.get('original_id', -1),
        )
    else:
        request = req_raw

    traffic_level = state.get('traffic_level', 0.3) if isinstance(state, dict) else getattr(state, 'traffic_level', 0.3)

    return ParatransitState(
        decision_epoch=epoch,
        current_request=request,
        vehicles=vehicles,
        traffic_level=traffic_level,
    )


def _compute_model_based_eta(
    metric_name: str,
    state,
    env,
    epoch: int,
    tree_root,
    comparison_data: Optional[Dict],
    assigned_vehicle_id: Optional[int],
    closest_vehicle_id: Optional[int],
    state_snapshot: Optional[Dict],
    override_assigned_vehicle: Optional[int]
) -> Optional[str]:
    """
    Compute MODEL-BASED ETA using BNN predictions from MCTS tree.

    ALL ETA metrics now use BNN-predicted vehicle times to ensure MDP_t vs MDP_{t-n}
    comparison reflects model drift (different models' understanding of travel times).

    The BNN predicts next-state observations including vehicle.current_time after
    taking an action. The predicted time implicitly captures the model's understanding
    of travel times under current traffic conditions.

    Model-based ETA formula:
    - ETA_PICKUP = BNN_predicted_vehicle_time (after assignment action)
    - ETA_DROPOFF = BNN_predicted_time + BNN_inferred_travel_time(pickup→dropoff)

    For DROPOFF, we estimate travel time based on BNN's understanding:
    travel_time = predicted_time - current_time (reflects model's travel time understanding)

    Args:
        metric_name: ETA_PICKUP_ASSIGNED, ETA_DROPOFF_ASSIGNED, ETA_PICKUP_CLOSEST,
                     or ETA_DROPOFF_CLOSEST
        state: Current ParatransitState
        env: Environment adapter
        epoch: Decision epoch
        tree_root: MCTS tree root for this epoch
        comparison_data: Comparison data with MDP_t vs MDP_{t-n} info
        assigned_vehicle_id: Assigned vehicle (from MDP_t or overridden for MDP_{t-n})
        closest_vehicle_id: Closest vehicle (for reference)
        state_snapshot: State snapshot with env_params
        override_assigned_vehicle: If set, use this vehicle (for MDP_{t-n} comparison)

    Returns:
        String like "36.75 (V0)" or None
    """
    if tree_root is None or state is None:
        return None

    # Determine which vehicle we're computing ETA for based on metric type
    is_closest = 'CLOSEST' in metric_name
    if is_closest:
        vehicle_id = closest_vehicle_id
    else:
        # ASSIGNED or MODEL metrics use assigned vehicle (with possible override)
        vehicle_id = override_assigned_vehicle if override_assigned_vehicle is not None else assigned_vehicle_id

    if vehicle_id is None:
        return None

    # Find the chance node for this vehicle's action
    children = getattr(tree_root, 'children', []) or []
    target_chance_node = None
    for child in children:
        action = getattr(child, 'action', None)
        if action == vehicle_id:
            target_chance_node = child
            break

    if target_chance_node is None:
        return None

    # Get BNN-predicted states from explain field
    explain = getattr(target_chance_node, 'explain', None)
    if explain is None or 'bnn_predicted_states' not in explain:
        return None

    bnn_states = explain['bnn_predicted_states']

    # ALWAYS use bnn_predicted_states['mdp_t'] - this is the prediction from the model
    # that was used to BUILD this tree:
    # - For MDP_t tree: BNN1 = current model (M̂k at time t)
    # - For MDP_{t-n} tree: BNN1 = old model (M̂k at time t-n, which is now outdated)
    # 
    # This correctly compares "current model understanding" vs "old model understanding"
    model_predictions = bnn_states.get('mdp_t', {})

    if not model_predictions:
        return None

    # Get the predicted time for the target vehicle
    v_key = f"V{vehicle_id}"
    if v_key not in model_predictions:
        return None

    predicted_time = model_predictions[v_key].get('predicted_time')
    if predicted_time is None:
        return None

    # For PICKUP metrics, the BNN-predicted time IS the ETA
    # (time when vehicle arrives at pickup after assignment)
    if 'PICKUP' in metric_name:
        return f"{round(predicted_time, 2)} (V{vehicle_id})"

    # For DROPOFF metrics, estimate travel time from BNN's understanding
    # BNN predicts vehicle time after assignment (implicitly includes travel to pickup)
    if 'DROPOFF' in metric_name:
        # Get current vehicle state from tree's root state
        # Handle both object (live) and dict (serialized) formats
        root_state = getattr(tree_root, 'state', None)
        if root_state is None:
            return None
        
        # Handle serialized dict format from adamcts_runner.py
        if isinstance(root_state, dict):
            vehicles = root_state.get('vehicles', [])
            if vehicle_id >= len(vehicles):
                return None
            current_vehicle = vehicles[vehicle_id]
            current_time = current_vehicle.get('current_time', 0.0)
            current_location = current_vehicle.get('current_location', 0)
            current_request = root_state.get('current_request')
            pickup_node = current_request.get('pickup_node') if current_request else None
            dropoff_node = current_request.get('dropoff_node') if current_request else None
        else:
            # Handle live object format
            vehicles = getattr(root_state, 'vehicles', [])
            if vehicle_id >= len(vehicles):
                return None
            current_vehicle = vehicles[vehicle_id]
            current_time = getattr(current_vehicle, 'current_time', 0.0)
            current_location = getattr(current_vehicle, 'current_location', 0)
            current_request = getattr(root_state, 'current_request', None)
            pickup_node = getattr(current_request, 'pickup_node', None) if current_request else None
            dropoff_node = getattr(current_request, 'dropoff_node', None) if current_request else None
        
        if current_request is None or pickup_node is None or dropoff_node is None:
            return None
        
        # BNN's implied travel time to pickup = predicted_time - current_time
        # This reflects the MODEL's understanding of travel time
        model_travel_to_pickup = predicted_time - current_time
        
        # Try to use base_travel_times ratio for better estimate
        # Ratio = base_travel(pickup→dropoff) / base_travel(current→pickup)
        eta_env = _get_eta_env(state_snapshot, epoch)
        if eta_env is not None and hasattr(eta_env, 'base_travel_times'):
            try:
                base_to_pickup = eta_env.base_travel_times[current_location, pickup_node]
                base_to_dropoff = eta_env.base_travel_times[pickup_node, dropoff_node]
                if base_to_pickup > 0:
                    # Scale model's pickup travel by base ratio
                    ratio = base_to_dropoff / base_to_pickup
                    model_travel_to_dropoff = model_travel_to_pickup * ratio
                else:
                    # Fallback: assume same travel time
                    model_travel_to_dropoff = model_travel_to_pickup
            except (IndexError, KeyError):
                # Fallback: assume same travel time
                model_travel_to_dropoff = model_travel_to_pickup
        else:
            # Fallback: assume same travel time (heuristic)
            model_travel_to_dropoff = model_travel_to_pickup
        
        eta_dropoff = predicted_time + model_travel_to_dropoff
        return f"{round(eta_dropoff, 2)} (V{vehicle_id})"

    return None


def _compute_model_based_eta_all_vehicles(
    metric_name: str,
    state,
    epoch: int,
    tree_root,
    state_snapshot: Optional[Dict],
) -> Optional[Dict[str, Any]]:
    """
    Compute MODEL-BASED ETA for all vehicles using BNN predictions from tree.

    Returns dict like {"V0": 12.34, "V1": None, ...} (float or None per vehicle).
    """
    if tree_root is None or state is None:
        return None

    is_dropoff = 'DROPOFF' in metric_name
    children = getattr(tree_root, 'children', []) or []
    if not children:
        return None

    # Get root state (needed for request_time and dropoff computation)
    root_state = getattr(tree_root, 'state', None)
    if root_state is None:
        return None

    # Extract request_time for relative ETA computation
    if isinstance(root_state, dict):
        cr = root_state.get('current_request')
        request_time = cr.get('request_time', 0.0) if cr else 0.0
    else:
        cr = getattr(root_state, 'current_request', None)
        request_time = getattr(cr, 'request_time', 0.0) if cr else 0.0

    # Pre-compute shared dropoff data if needed
    eta_env = None
    if is_dropoff:
        eta_env = _get_eta_env(state_snapshot, epoch)

    result = {}
    for child in children:
        action = getattr(child, 'action', None)
        if action is None:
            continue

        v_key = f"V{action}"

        # Get BNN-predicted states
        explain = getattr(child, 'explain', None)
        if explain is None or 'bnn_predicted_states' not in explain:
            result[v_key] = None
            continue

        model_predictions = explain['bnn_predicted_states'].get('mdp_t', {})
        if not model_predictions or v_key not in model_predictions:
            result[v_key] = None
            continue

        predicted_time = model_predictions[v_key].get('predicted_time')
        if predicted_time is None:
            result[v_key] = None
            continue

        if not is_dropoff:
            # PICKUP: BNN-predicted time relative to request_time
            result[v_key] = round(predicted_time - request_time, 2)
        else:
            # DROPOFF: estimate using BNN travel time ratio
            if isinstance(root_state, dict):
                vehicles = root_state.get('vehicles', [])
                if action >= len(vehicles):
                    result[v_key] = None
                    continue
                cv = vehicles[action]
                current_time = cv.get('current_time', 0.0)
                current_location = cv.get('current_location', 0)
                current_request = root_state.get('current_request')
                pickup_node = current_request.get('pickup_node') if current_request else None
                dropoff_node = current_request.get('dropoff_node') if current_request else None
            else:
                vehicles = getattr(root_state, 'vehicles', [])
                if action >= len(vehicles):
                    result[v_key] = None
                    continue
                cv = vehicles[action]
                current_time = getattr(cv, 'current_time', 0.0)
                current_location = getattr(cv, 'current_location', 0)
                current_request = getattr(root_state, 'current_request', None)
                pickup_node = getattr(current_request, 'pickup_node', None) if current_request else None
                dropoff_node = getattr(current_request, 'dropoff_node', None) if current_request else None

            if current_request is None or pickup_node is None or dropoff_node is None:
                result[v_key] = None
                continue

            model_travel_to_pickup = predicted_time - current_time
            model_travel_to_dropoff = model_travel_to_pickup  # fallback

            if eta_env is not None and hasattr(eta_env, 'base_travel_times'):
                try:
                    base_to_pickup = eta_env.base_travel_times[current_location, pickup_node]
                    base_to_dropoff = eta_env.base_travel_times[pickup_node, dropoff_node]
                    if base_to_pickup > 0:
                        ratio = base_to_dropoff / base_to_pickup
                        model_travel_to_dropoff = model_travel_to_pickup * ratio
                except (IndexError, KeyError):
                    pass

            result[v_key] = round(predicted_time + model_travel_to_dropoff - request_time, 2)

    return result if result else None


def _compute_path_metric_all_vehicles(
    base_metric: str,
    state,
    env,
    epoch: int,
    state_snapshot: Optional[Dict],
    mcts_data: Optional[Dict],
) -> Optional[Dict[str, Any]]:
    """
    Compute assignment-path metric for all vehicles.

    base_metric: one of CLEAR_CURRENT_ROUTE_TIME, DEADHEAD_TO_PICKUP,
                 DROPOFF_SLACK, SERVICE_PATH_DISRUPTION.

    Returns dict like {"V0": 12.34, "V1": 0.0, ...} (float or str per vehicle).
    """
    if state is None:
        return None

    vehicles = state.get('vehicles', []) if isinstance(state, dict) else getattr(state, 'vehicles', [])
    if not vehicles:
        return None

    eta_env = _get_eta_env(state_snapshot, epoch)
    if eta_env is None:
        return None

    try:
        ps = _reconstruct_paratransit_state(state, epoch)
    except Exception:
        return None

    saved_epoch = eta_env.current_decision_epoch
    eta_env.current_decision_epoch = epoch

    result = {}
    try:
        for v_idx in range(len(vehicles)):
            if v_idx >= len(ps.vehicles):
                continue
            v_key = f"V{v_idx}"
            try:
                stats = eta_env._compute_assignment_path_stats(ps, v_idx)
            except Exception:
                result[v_key] = None
                continue

            if base_metric == 'CLEAR_CURRENT_ROUTE_TIME':
                result[v_key] = round(stats['clear_current_route_time'], 2)
            elif base_metric == 'DEADHEAD_TO_PICKUP':
                result[v_key] = round(stats['deadhead_to_pickup'], 2)
            elif base_metric == 'DROPOFF_SLACK':
                result[v_key] = round(stats['dropoff_slack'], 2)
            elif base_metric == 'SERVICE_PATH_DISRUPTION':
                parts = []
                if stats['service_path_event_affected']:
                    parts.append('event')
                if stats['service_path_bridge_affected']:
                    parts.append('bridge')
                result[v_key] = '+'.join(parts) if parts else 'none'
    finally:
        eta_env.current_decision_epoch = saved_epoch

    return result if result else None


def _get_eta_env(state_snapshot: Optional[Dict], epoch: int):
    """
    Get or create a complete NSParatransitV0 environment for ETA computation.

    Uses state_snapshot['env_params'] to rebuild the environment with proper
    travel time computation methods. Caches environments to avoid reloading
    large CSV files.

    Args:
        state_snapshot: State snapshot containing env_params
        epoch: Current decision epoch (for event timing)

    Returns:
        NSParatransitV0 environment or None if cannot build
    """
    global _ETA_ENV_CACHE

    if state_snapshot is None:
        return None

    env_params = state_snapshot.get('env_params', {})
    if not env_params:
        return None

    # Build cache key from immutable env parameters
    # Sort event_nodes to avoid cache miss from different ordering
    cache_key = (
        env_params.get('n_vehicles'),
        env_params.get('n_requests'),
        env_params.get('data_path'),  # Travel time matrix path
        env_params.get('requests_csv_path'),
        tuple(env_params.get('fixed_request_ids') or []),
        env_params.get('seed'),
        env_params.get('traffic_condition'),
        env_params.get('max_time', 480),
        env_params.get('event_epoch'),
        tuple(sorted(env_params.get('event_nodes') or [])),
        env_params.get('event_multiplier'),
        env_params.get('bridge_accident_epoch'),
        env_params.get('bridge_accident_multiplier'),
    )

    # Check cache first
    if cache_key in _ETA_ENV_CACHE:
        env = _ETA_ENV_CACHE[cache_key]
    else:
        # Build new environment
        try:
            from nsparatransit.nsparatransit_v0 import NSParatransitV0

            env = NSParatransitV0(
                n_vehicles=env_params.get('n_vehicles', 5),
                n_requests=env_params.get('n_requests', 10),
                traffic_condition=env_params.get('traffic_condition', 0.3),
                seed=env_params.get('seed', 42),
                data_path=env_params.get('data_path'),  # Travel time matrix path
                use_real_data=True,
                fixed_request_ids=env_params.get('fixed_request_ids'),
                requests_csv_path=env_params.get('requests_csv_path'),
                max_time=env_params.get('max_time', 480),
            )

            # Configure event if present
            if env_params.get('event_epoch') is not None and env_params.get('event_nodes'):
                env.set_event_config(
                    event_epoch=env_params.get('event_epoch'),
                    event_nodes=env_params.get('event_nodes'),
                    event_multiplier=env_params.get('event_multiplier', 3.0),
                )

            # Configure bridge accident if present
            if env_params.get('bridge_accident_epoch') is not None:
                env.set_bridge_accident_config(
                    accident_epoch=env_params['bridge_accident_epoch'],
                    multiplier=env_params.get('bridge_accident_multiplier', 2.0),
                )

            # Cache the environment
            _ETA_ENV_CACHE[cache_key] = env

        except Exception as e:
            print(f"Warning: Failed to rebuild environment for ETA computation: {e}")
            return None

    # Set current epoch for proper event timing
    env.current_decision_epoch = epoch

    return env


# ============================================================================
# PCTL FORMULA PARSING AND EVALUATION
# ============================================================================

from enum import Enum
from dataclasses import dataclass
from typing import Union


class ExprType(Enum):
    """Expression node types for PCTL formula AST."""
    PROP = "prop"   # Atomic proposition
    NOT = "not"     # Negation
    OR = "or"       # Disjunction
    AND = "and"     # Conjunction


@dataclass
class ExprNode:
    """
    AST node for PCTL state formulas.

    Types:
    - PROP: Atomic proposition (value = proposition name)
    - NOT: Negation (children = [child])
    - OR: Disjunction (children = [left, right])
    - AND: Conjunction (children = [left, right])
    """
    type: ExprType
    value: Optional[str] = None
    children: Optional[List['ExprNode']] = None


def PROP(name: str) -> ExprNode:
    """Create an atomic proposition node."""
    return ExprNode(type=ExprType.PROP, value=name)


def NOT(child: ExprNode) -> ExprNode:
    """Create a negation node."""
    return ExprNode(type=ExprType.NOT, children=[child])


def OR(left: ExprNode, right: ExprNode) -> ExprNode:
    """Create a disjunction node."""
    return ExprNode(type=ExprType.OR, children=[left, right])


def AND(left: ExprNode, right: ExprNode) -> ExprNode:
    """Create a conjunction node."""
    return ExprNode(type=ExprType.AND, children=[left, right])


def _normalize_expr_string(s: str) -> str:
    """Normalize expression string by removing extra whitespace."""
    return ' '.join(s.split())


def parse_expr(expr_str: str) -> ExprNode:
    """
    Parse a state expression string into an ExprNode AST.

    Supports:
    - Atomic propositions: "goal", "hole", "safe"
    - Negation: "!hole", "~hole", "not hole"
    - Conjunction: "safe & !hole", "safe and !hole"
    - Disjunction: "goal | hole", "goal or hole"
    - Parentheses: "(goal | safe) & !hole"

    Operator precedence: NOT > AND > OR

    Args:
        expr_str: Expression string to parse

    Returns:
        ExprNode AST

    Raises:
        ValueError: If expression is invalid
    """
    expr_str = _normalize_expr_string(expr_str.strip())
    if not expr_str:
        raise ValueError("Empty expression")

    # Tokenize
    tokens = _tokenize_expr(expr_str)

    # Parse with recursive descent
    result, pos = _parse_or_expr(tokens, 0)

    if pos < len(tokens):
        raise ValueError(f"Unexpected token at position {pos}: {tokens[pos]}")

    return result


def _tokenize_expr(expr_str: str) -> List[str]:
    """Tokenize expression string into tokens."""
    tokens = []
    i = 0

    while i < len(expr_str):
        c = expr_str[i]

        if c.isspace():
            i += 1
            continue

        if c in '()&|!~':
            tokens.append(c)
            i += 1
            continue

        # Check for keywords
        remaining = expr_str[i:].lower()
        for keyword in ['and', 'or', 'not']:
            if remaining.startswith(keyword) and (
                len(remaining) == len(keyword) or
                not remaining[len(keyword)].isalnum()
            ):
                tokens.append(keyword)
                i += len(keyword)
                break
        else:
            # Read identifier (atomic proposition)
            j = i
            while j < len(expr_str) and (expr_str[j].isalnum() or expr_str[j] == '_'):
                j += 1
            if j > i:
                tokens.append(expr_str[i:j])
                i = j
            else:
                raise ValueError(f"Unexpected character at position {i}: {c}")

    return tokens


def _parse_or_expr(tokens: List[str], pos: int) -> Tuple[ExprNode, int]:
    """Parse OR expression (lowest precedence)."""
    left, pos = _parse_and_expr(tokens, pos)

    while pos < len(tokens) and tokens[pos] in ('|', 'or'):
        pos += 1
        right, pos = _parse_and_expr(tokens, pos)
        left = OR(left, right)

    return left, pos


def _parse_and_expr(tokens: List[str], pos: int) -> Tuple[ExprNode, int]:
    """Parse AND expression."""
    left, pos = _parse_not_expr(tokens, pos)

    while pos < len(tokens) and tokens[pos] in ('&', 'and'):
        pos += 1
        right, pos = _parse_not_expr(tokens, pos)
        left = AND(left, right)

    return left, pos


def _parse_not_expr(tokens: List[str], pos: int) -> Tuple[ExprNode, int]:
    """Parse NOT expression (highest precedence among operators)."""
    if pos < len(tokens) and tokens[pos] in ('!', '~', 'not'):
        pos += 1
        child, pos = _parse_not_expr(tokens, pos)
        return NOT(child), pos

    return _parse_primary(tokens, pos)


def _parse_primary(tokens: List[str], pos: int) -> Tuple[ExprNode, int]:
    """Parse primary expression (atomic prop or parenthesized expr)."""
    if pos >= len(tokens):
        raise ValueError("Unexpected end of expression")

    token = tokens[pos]

    if token == '(':
        pos += 1
        expr, pos = _parse_or_expr(tokens, pos)
        if pos >= len(tokens) or tokens[pos] != ')':
            raise ValueError("Missing closing parenthesis")
        return expr, pos + 1

    if token.isalnum() or '_' in token:
        return PROP(token), pos + 1

    raise ValueError(f"Unexpected token: {token}")


# ============================================================================
# PCTL PATH FORMULA PARSING
# ============================================================================

@dataclass
class PropertyFormula:
    """
    Represents a parsed PCTL property formula.

    Supports:
    - P=? [F expr]: Probability query for eventually expr
    - P>=p [F expr]: Probability threshold (at least p)
    - P<=p [F expr]: Probability threshold (at most p)
    - P=? [G expr]: Probability of globally expr
    - P=? [X expr]: Probability of next expr
    - P=? [expr1 U expr2]: Probability of expr1 until expr2
    - P=? [F<=n expr]: Bounded eventually (within n steps)
    - R{"steps"}=? [F expr]: Expected steps to reach expr (conditional on reaching)

    Attributes:
        kind: 'P' for probability, 'R' for reward
        cmp: None for query (=?), '>=' for lower bound, '<=' for upper bound
        threshold: Threshold value for cmp (e.g., 0.9 for P>=0.9)
        reward_struct: Reward structure name for R formulas (e.g., "steps")
        pattern: Path pattern ('F', 'F_bounded', 'G', 'X', 'U')
        expr: Parsed expression (ExprNode for F/G/X, tuple of ExprNode for U)
        bound: Step bound for bounded eventually (None for unbounded)
    """
    kind: str  # 'P' or 'R'
    cmp: Optional[str]  # None, '>=', '<='
    threshold: Optional[float]
    reward_struct: Optional[str]  # For R formulas
    pattern: str  # 'F', 'F_bounded', 'G', 'X', 'U'
    expr: Any  # ExprNode or tuple of ExprNode
    bound: Optional[int] = None  # For bounded operators


def _find_toplevel_until(s: str) -> int:
    """
    Find the position of 'U' at the top level (not inside parentheses).

    Returns:
        Position of 'U' if found at top level, -1 otherwise
    """
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == 'U' and depth == 0:
            # Check it's not part of a longer identifier
            before_ok = i == 0 or not s[i-1].isalnum()
            after_ok = i == len(s) - 1 or not s[i+1].isalnum()
            if before_ok and after_ok:
                return i
        i += 1
    return -1


def _parse_path_formula(path_content: str) -> Tuple[str, Any, Optional[int]]:
    """
    Parse path formula content (inside brackets).

    Args:
        path_content: Content inside brackets, e.g., "F goal", "F(goal)", "G !hole"

    Returns:
        Tuple of (pattern, expr, bound) where:
        - pattern: 'F', 'F_bounded', 'G', 'X', 'U'
        - expr: ExprNode for F/G/X, tuple (expr1, expr2) for U
        - bound: Step bound for bounded operators (None otherwise)
    """
    path_content = path_content.strip()

    # Check for Until operator first (has two operands)
    u_pos = _find_toplevel_until(path_content)
    if u_pos != -1:
        expr1_str = path_content[:u_pos].strip()
        expr2_str = path_content[u_pos+1:].strip()
        return 'U', (parse_expr(expr1_str), parse_expr(expr2_str)), None

    # Check for bounded eventually: F<=n or F<n
    import re
    bounded_match = re.match(r'^F\s*<=?\s*(\d+)\s+(.+)$', path_content)
    if bounded_match:
        bound = int(bounded_match.group(1))
        expr_str = bounded_match.group(2)
        return 'F_bounded', parse_expr(expr_str), bound

    # Check for temporal operators with parentheses or spaces
    for op in ['F', 'G', 'X']:
        # Pattern: "F(expr)" or "F expr"
        if path_content.startswith(op):
            rest = path_content[len(op):].strip()
            # Handle parenthesized form: F(expr)
            if rest.startswith('(') and rest.endswith(')'):
                expr_str = rest[1:-1].strip()
            else:
                expr_str = rest
            return op, parse_expr(expr_str), None

    raise ValueError(f"Unknown path formula: {path_content}")


def parse_property_formula(formula: str) -> PropertyFormula:
    """
    Parse extended PCTL property formula.

    Supports:
    - P=? [F expr]: Probability query
    - P>=p [F expr]: Probability at least p
    - P<=p [F expr]: Probability at most p
    - R{"struct"}=? [F expr]: Expected reward to reach expr

    Args:
        formula: Formula string

    Returns:
        PropertyFormula object

    Raises:
        ValueError: If formula is invalid
    """
    formula = formula.strip()
    import re

    # Try R{"struct"}=? [path] format
    r_match = re.match(r'^R\s*\{\s*"(\w+)"\s*\}\s*=\s*\?\s*\[(.+)\]$', formula)
    if r_match:
        reward_struct = r_match.group(1)
        path_content = r_match.group(2).strip()
        pattern, expr, bound = _parse_path_formula(path_content)
        return PropertyFormula(
            kind='R',
            cmp=None,
            threshold=None,
            reward_struct=reward_struct,
            pattern=pattern,
            expr=expr,
            bound=bound
        )

    # Try P>=p [path] or P<=p [path] format
    threshold_match = re.match(r'^P\s*(>=|<=)\s*([\d.]+)\s*\[(.+)\]$', formula)
    if threshold_match:
        cmp = threshold_match.group(1)
        threshold = float(threshold_match.group(2))
        path_content = threshold_match.group(3).strip()
        pattern, expr, bound = _parse_path_formula(path_content)
        return PropertyFormula(
            kind='P',
            cmp=cmp,
            threshold=threshold,
            reward_struct=None,
            pattern=pattern,
            expr=expr,
            bound=bound
        )

    # Try P=? [path] format
    query_match = re.match(r'^P\s*=\s*\?\s*\[(.+)\]$', formula)
    if query_match:
        path_content = query_match.group(1).strip()
        pattern, expr, bound = _parse_path_formula(path_content)
        return PropertyFormula(
            kind='P',
            cmp=None,
            threshold=None,
            reward_struct=None,
            pattern=pattern,
            expr=expr,
            bound=bound
        )

    raise ValueError(f"Cannot parse property formula: {formula}")


# ============================================================================
# TRACE-BASED PCTL EVALUATION
# ============================================================================

def _eval_expr_at_step(expr: ExprNode, trace: List[Dict], step: int) -> bool:
    """
    Evaluate expression at a specific step in trace.

    Args:
        expr: ExprNode to evaluate
        trace: List of state dictionaries with atomic propositions
        step: Step index

    Returns:
        Boolean result
    """
    if step >= len(trace):
        return False

    state = trace[step]

    if expr.type == ExprType.PROP:
        return state.get(expr.value, False)
    elif expr.type == ExprType.NOT:
        return not _eval_expr_at_step(expr.children[0], trace, step)
    elif expr.type == ExprType.AND:
        return (_eval_expr_at_step(expr.children[0], trace, step) and
                _eval_expr_at_step(expr.children[1], trace, step))
    elif expr.type == ExprType.OR:
        return (_eval_expr_at_step(expr.children[0], trace, step) or
                _eval_expr_at_step(expr.children[1], trace, step))

    return False

def _eval_F_on_trace(expr: ExprNode, trace: List[Dict], bound: Optional[int] = None) -> bool:
    """
    Evaluate F (eventually) on trace.
    
    NOTE: Trace is transition-based (each element is from evaluate_atomic_props(s,a,s')).
    Step 0 = current action's immediate result (s' from current transition).
    
    F<=k means "within k transitions" starting from current.
    - F<=0: only check step 0 (current action's result)
    - F<=1: check step 0 and step 1
    - F<=k: check steps 0..k

    Args:
        expr: Target expression
        trace: List of state dictionaries (transition-based)
        bound: Optional step bound (F<=k means check steps 0 to k)

    Returns:
        True if expr is satisfied at some point in trace within bound
    """
    # F<=k checks steps 0..k (bound+1 elements total)
    # This is correct: F<=0 checks step 0 only, F<=1 checks steps 0 and 1, etc.
    max_step = len(trace) if bound is None else min(bound + 1, len(trace))
    return any(_eval_expr_at_step(expr, trace, i) for i in range(max_step))


def _eval_G_on_trace(expr: ExprNode, trace: List[Dict]) -> bool:
    """
    Evaluate G (globally) on trace.
    
    NOTE: Trace is transition-based (each element is from evaluate_atomic_props(s,a,s')).
    Step 0 = current action's immediate result.

    Args:
        expr: Target expression
        trace: List of state dictionaries (transition-based)

    Returns:
        True if expr is satisfied at all points in trace
    """
    return all(_eval_expr_at_step(expr, trace, i) for i in range(len(trace)))


def _eval_X_on_trace(expr: ExprNode, trace: List[Dict]) -> bool:
    """
    Evaluate X (next) on trace.
    
    NOTE: Trace is transition-based. Step 0 = current action's immediate result.
    X (next) checks if expr holds at step 0 - the direct consequence of the current action.
    
    In transition-based semantics:
    - We're at state s, taking action a
    - trace[0] = props from evaluate_atomic_props(s, a, s') = result of current action
    - X checks "does the next state (s') satisfy expr?" → trace[0]

    Args:
        expr: Target expression
        trace: List of state dictionaries (transition-based)

    Returns:
        True if expr is satisfied at step 0 (current action's immediate result)
    """
    # X checks the immediate result of current action (step 0)
    return _eval_expr_at_step(expr, trace, 0) if len(trace) > 0 else False


def _eval_U_on_trace(expr1: ExprNode, expr2: ExprNode, trace: List[Dict]) -> bool:
    """
    Evaluate U (until) on trace.

    Args:
        expr1: First expression (must hold until expr2)
        expr2: Second expression (eventually holds)
        trace: List of state dictionaries

    Returns:
        True if expr1 holds until expr2 becomes true
    """
    for i in range(len(trace)):
        if _eval_expr_at_step(expr2, trace, i):
            return True
        if not _eval_expr_at_step(expr1, trace, i):
            return False
    return False


def _first_step_satisfying(expr: ExprNode, trace: List[Dict], bound: Optional[int] = None) -> Optional[int]:
    """
    Find first step where expression is satisfied.

    Args:
        expr: Target expression
        trace: List of state dictionaries
        bound: Optional step bound

    Returns:
        Step index or None if never satisfied
    """
    max_step = len(trace) if bound is None else min(bound + 1, len(trace))
    for i in range(max_step):
        if _eval_expr_at_step(expr, trace, i):
            return i
    return None


# ============================================================================
# REWARD FORMULA EVALUATION
# ============================================================================

@dataclass
class RewardWithProbability:
    """
    Result of reward formula evaluation with probability context.

    For R{"steps"}=? [F goal], this captures:
    - probability: P(reaching goal) = n_satisfied / n_total
    - conditional_steps: E[steps | reached goal] = average steps among satisfied traces
    - n_satisfied: Number of traces that reached goal
    - n_total: Total number of traces

    IMPORTANT: conditional_steps is a CONDITIONAL expectation. If probability is low,
    the conditional_steps may be misleadingly good because it only counts successful traces.
    For example:
    - Action A: 10% success, 5 steps when successful → conditional_steps=5
    - Action B: 90% success, 10 steps when successful → conditional_steps=10
    Action A looks better by conditional_steps but B is clearly better overall.

    For meaningful comparison, use probability-weighted expectation:
    E[steps] = probability * conditional_steps
    """
    probability: float
    conditional_steps: float
    n_satisfied: int
    n_total: int


def eval_reward_with_probability_on_traces(formula: str, traces: List[List[Dict]]) -> RewardWithProbability:
    """
    Evaluate reward formula on traces with full probability context.

    Args:
        formula: Reward formula string (e.g., 'R{"steps"}=? [F goal]')
        traces: List of traces

    Returns:
        RewardWithProbability with probability, conditional_steps, and counts
    """
    if not traces:
        return RewardWithProbability(0.0, 0.0, 0, 0)

    try:
        prop = parse_property_formula(formula)
    except ValueError:
        return RewardWithProbability(0.0, 0.0, 0, len(traces))

    if prop.kind != 'R' or prop.reward_struct != 'steps':
        return RewardWithProbability(0.0, 0.0, 0, len(traces))

    steps_list = []
    for trace in traces:
        step = _first_step_satisfying(prop.expr, trace, prop.bound)
        if step is not None:
            steps_list.append(step)

    n_satisfied = len(steps_list)
    n_total = len(traces)
    probability = n_satisfied / n_total if n_total > 0 else 0.0
    conditional_steps = sum(steps_list) / n_satisfied if n_satisfied > 0 else 0.0

    return RewardWithProbability(probability, conditional_steps, n_satisfied, n_total)


def eval_property_formula_on_traces(formula: str, traces: List[List[Dict]]) -> Union[float, bool, Dict, str]:
    """
    Evaluate extended property formula on traces.

    Args:
        formula: Property formula string
        traces: List of traces

    Returns:
        - For P=?: float probability
        - For P>=p/P<=p: bool whether threshold is met
        - For R{"steps"}=?: Dict with probability, conditional_steps, n_satisfied, n_total
        - For errors: str error message
    """
    if not traces:
        return 0.0

    try:
        prop = parse_property_formula(formula)
    except ValueError as e:
        return f"Parse error: {e}"

    # Reward formula
    if prop.kind == 'R':
        result = eval_reward_with_probability_on_traces(formula, traces)
        return {
            'probability': result.probability,
            'conditional_steps': result.conditional_steps,
            'n_satisfied': result.n_satisfied,
            'n_total': result.n_total
        }

    # Probability formula
    satisfied = 0
    for trace in traces:
        if prop.pattern == 'F':
            if _eval_F_on_trace(prop.expr, trace):
                satisfied += 1
        elif prop.pattern == 'F_bounded':
            if _eval_F_on_trace(prop.expr, trace, prop.bound):
                satisfied += 1
        elif prop.pattern == 'G':
            if _eval_G_on_trace(prop.expr, trace):
                satisfied += 1
        elif prop.pattern == 'X':
            if _eval_X_on_trace(prop.expr, trace):
                satisfied += 1
        elif prop.pattern == 'U':
            if _eval_U_on_trace(prop.expr[0], prop.expr[1], trace):
                satisfied += 1

    probability = satisfied / len(traces)

    # Apply threshold if present
    if prop.cmp == '>=':
        return probability >= prop.threshold
    elif prop.cmp == '<=':
        return probability <= prop.threshold
    else:
        return probability


# ============================================================================
# AP WHITELIST VALIDATION
# ============================================================================

def validate_atomic_propositions(formula: str, use_case: str = 'paratransit') -> Tuple[bool, List[str]]:
    """
    Validate that all atomic propositions in formula are in the whitelist.

    Args:
        formula: PCTL formula string
        use_case: Use case name ('paratransit')

    Returns:
        Tuple of (is_valid, list of invalid propositions)
    """
    # Get whitelist for use case
    if use_case == 'paratransit':
        from use_cases.paratransit.config import ParatransitConfig
        whitelist = ParatransitConfig.AP_WHITELIST
    else:
        return True, []  # No validation for unknown use cases

    # Extract atomic propositions from formula
    try:
        prop = parse_property_formula(formula)
    except ValueError:
        return True, []  # Can't parse, skip validation

    # Collect all proposition names from expression
    def collect_props(expr: ExprNode) -> List[str]:
        if expr.type == ExprType.PROP:
            return [expr.value]
        elif expr.children:
            result = []
            for child in expr.children:
                result.extend(collect_props(child))
            return result
        return []

    if isinstance(prop.expr, tuple):
        # Until formula
        props = collect_props(prop.expr[0]) + collect_props(prop.expr[1])
    else:
        props = collect_props(prop.expr)

    # Validate against whitelist
    invalid = [p for p in props if p not in whitelist]
    return len(invalid) == 0, invalid


# ============================================================================
# STARTUP PRE-COMPUTATION
# ============================================================================

def precompute_all_formulas(mcts_data: Dict, all_formulas: List[str],
                            selected_epochs: Optional[List[int]] = None) -> None:
    """
    Pre-compute all PCTL formulas for selected epochs in mcts_data at startup.

    Iterates over each epoch's MDP_t and MDP_{t-n} tree, evaluates every
    formula, and stores results in _FORMULA_ON_TREE_CACHE.  Also populates
    _MDP_REBUILD_CACHE so pkl files are loaded into memory once.

    Called from user_study/app.py synchronously before the server starts.

    Args:
        mcts_data: Full MCTS data dict (per_step_trees already converted to nodes)
        all_formulas: List of unique formula strings to evaluate (DERIVED: skipped)
        selected_epochs: If provided, only pre-compute these epochs (e.g. user study subset).
                         If None, pre-compute all epochs.
    """
    global _ACTIVE_CONFIG, _FORMULA_ON_TREE_CACHE, _MDP_REBUILD_CACHE

    per_step_trees = mcts_data.get('per_step_trees', {})
    scenario_data = mcts_data.get('scenario_data', {})
    mdp_comparisons = scenario_data.get('mdp_comparisons', {})
    model_version_history = scenario_data.get('model_version_history', {})

    def action_name(a: int) -> str:
        if _ACTIVE_CONFIG and hasattr(_ACTIVE_CONFIG, 'action_to_name'):
            return _ACTIVE_CONFIG.action_to_name(a)
        return f"V{a}"

    # Only PCTL formulas (not DERIVED:) need tree-based evaluation
    pctl_formulas = [f for f in all_formulas if not f.startswith('DERIVED:')]

    # Use runtime int-keyed lookup if available, else parse keys
    rt = mcts_data.get('_runtime', {})
    run_id = rt.get('run_id')
    trees_by_epoch = rt.get('trees_by_epoch')
    if trees_by_epoch:
        all_epochs = sorted(trees_by_epoch.keys())
    else:
        all_epochs = sorted({int(k.split('|')[1]) for k in per_step_trees if '|' in k})

    # Filter to selected epochs if specified
    if selected_epochs is not None:
        selected_set = set(selected_epochs)
        epochs = [e for e in all_epochs if e in selected_set]
    else:
        epochs = all_epochs
    total = len(epochs)

    for i, epoch in enumerate(epochs, 1):
        if trees_by_epoch:
            tree_t = trees_by_epoch.get(epoch)
        else:
            tree_t = per_step_trees.get(f"ep_1|{epoch}")
        if tree_t is None:
            continue

        # Stable cache key prefix for MDP_t
        t_prefix = (run_id, epoch, 'mdp_t') if run_id else None

        # Evaluate all formulas on MDP_t tree
        for formula in pctl_formulas:
            _evaluate_formula_on_tree_children(formula, tree_t, action_name, cache_key_prefix=t_prefix)

        # Load MDP_{t-n} tree from pkl (triggers _rebuild_mdp_for_comparison which
        # checks pkl cache, so this is just a fast disk read + tree build)
        comparison_data = mdp_comparisons.get(epoch)
        if comparison_data:
            state_snapshot = comparison_data.get('state_snapshot', {})

            # Use pre-computed prev_version_id (falls back to scan for legacy pkl)
            prev_version_id = state_snapshot.get('prev_version_id')
            if prev_version_id is None:
                model_version_at_t = state_snapshot.get('model_version_at_t', 0)
                for v_id in range(model_version_at_t - 1, -1, -1):
                    if v_id in model_version_history:
                        prev_version_id = v_id
                        break
                if prev_version_id is None and model_version_at_t == 0 and 0 in model_version_history:
                    prev_version_id = 0

            if prev_version_id is not None:
                mem_key = (run_id, epoch, prev_version_id)

                # Load into _MDP_REBUILD_CACHE if not already there
                if mem_key not in _MDP_REBUILD_CACHE:
                    tree_dict = _rebuild_mdp_for_comparison(
                        state_snapshot, model_version_history, prev_version_id, mcts_data
                    )
                    if tree_dict is not None and _ACTIVE_CONFIG and hasattr(_ACTIVE_CONFIG, 'build_tree_from_data'):
                        tree_t_minus_n = _ACTIVE_CONFIG.build_tree_from_data(tree_dict)
                        if tree_t_minus_n is not None:
                            _MDP_REBUILD_CACHE[mem_key] = tree_t_minus_n

                tree_t_minus_n = _MDP_REBUILD_CACHE.get(mem_key)
                if tree_t_minus_n is not None:
                    tn_role = f'mdp_t_minus_n_{prev_version_id}'
                    tn_prefix = (run_id, epoch, tn_role) if run_id else None
                    for formula in pctl_formulas:
                        _evaluate_formula_on_tree_children(formula, tree_t_minus_n, action_name, cache_key_prefix=tn_prefix)

        print(f"  [warmup] epoch {epoch} pre-computed ({i}/{total})", flush=True)
