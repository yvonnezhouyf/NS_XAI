"""
NS-XAI Orchestrator: Main coordinator for Non-Stationary eXplainable AI system.
Flow: User Query -> Classification -> PCTL Formulas -> PCTL Results -> Explanation

This orchestrator is designed for Paratransit domain with MDP comparison for non-stationarity.
For Frozen Lake support, see orchestrator_frozenlake.py.
"""

import traceback
import uuid
from typing import Any, Dict, List, Optional

from core.query_classification import classify_query_with_openai
from core.explanation_generation import generate_explanation_with_openai
from core.formula_evaluator import (
    evaluate_mdp_comparison,
    set_evidence_cache,
    set_active_config as set_formula_evaluator_config,
)
from core.pctl_checking import set_active_config
from logiex_baseline.evaluator import build_logiex_prompt


def prepare_runtime_artifacts(config, mcts_data: Dict) -> None:
    """Idempotent one-time preparation of runtime artifacts for fast query serving.

    Builds trees_by_epoch (int-keyed), epochs list, latest_epoch, converts
    per_step_trees dicts to node objects, and stores everything in
    mcts_data["_runtime"]. Safe to call multiple times - no-ops after first.

    Args:
        config: Use case configuration (ParatransitConfig)
        mcts_data: Full MCTS data dict (modified in place)
    """
    rt = mcts_data.get('_runtime')
    if rt is not None and rt.get('runtime_prepared'):
        return

    rt = mcts_data.setdefault('_runtime', {})

    # Convert per_step_trees from dict to AdaptedMCTSNode objects (once)
    if 'per_step_trees' in mcts_data:
        build_tree = config.build_tree_from_data
        for key, tree_data in mcts_data['per_step_trees'].items():
            if isinstance(tree_data, dict):
                mcts_data['per_step_trees'][key] = build_tree(tree_data)

    # Build int-keyed trees_by_epoch and sorted epochs list
    trees_by_epoch = {}
    per_step_trees = mcts_data.get('per_step_trees', {})
    for k, v in per_step_trees.items():
        if '|' in k:
            epoch = int(k.split('|')[1])
            trees_by_epoch[epoch] = v

    epochs = sorted(trees_by_epoch.keys())

    rt['trees_by_epoch'] = trees_by_epoch
    rt['epochs'] = epochs
    rt['latest_epoch'] = epochs[-1] if epochs else 0
    rt['run_id'] = str(uuid.uuid4())
    rt['comparison_results'] = {}
    rt['rebuilt_trees'] = {}
    rt.setdefault('ns_explanation_history', {})
    rt['runtime_prepared'] = True


# ============================================================================
# RESULT FILTERING UTILITIES
# ============================================================================

def should_keep_pctl_result(result_value: Any, formula: str = '') -> bool:
    """
    Filter out PCTL results where there's no meaningful difference.

    For PCTL formulas (P=?, R=?):
    - Check if there's variation WITHIN each MDP's actions
    - Check if there's difference BETWEEN MDP_t and MDP_{t-n} for same action

    For DERIVED metrics:
    - Check if MDP_t value differs from MDP_{t-n} value
    """
    if not isinstance(result_value, dict):
        return True

    # Check for MDP comparison format
    if 'mdp_t' in result_value:
        mdp_t = result_value.get('mdp_t')
        mdp_t_minus_n = result_value.get('mdp_t_minus_n')

        # DERIVED metrics: always keep (they provide query-relevant context,
        # not only differential evidence)
        if formula.startswith('DERIVED:'):
            return True

        # PCTL formulas: check variation across actions
        if not isinstance(mdp_t, dict):
            return True

        t_values = _extract_pctl_values_from_actions(mdp_t)

        # If we have MDP_{t-n}, check for cross-MDP differences
        if mdp_t_minus_n and isinstance(mdp_t_minus_n, dict):
            t_n_values = _extract_pctl_values_from_actions(mdp_t_minus_n)

            # Keep if there's variation within either MDP
            if _has_significant_variation(t_values) or _has_significant_variation(t_n_values):
                return True

            # Keep if there's difference between the two MDPs
            if t_values and t_n_values:
                for action in set(mdp_t.keys()) & set(mdp_t_minus_n.keys()):
                    v_t = _get_pctl_value(mdp_t.get(action))
                    v_tn = _get_pctl_value(mdp_t_minus_n.get(action))
                    if v_t is not None and v_tn is not None:
                        if round(v_t, 4) != round(v_tn, 4):
                            return True
            return False

        # Only MDP_t available - check for variation within it
        return _has_significant_variation(t_values) if t_values else True

    # Legacy single MDP format
    numeric_values = _extract_pctl_values_from_actions(result_value)
    if not numeric_values:
        return True
    return _has_significant_variation(numeric_values)


def _extract_pctl_values_from_actions(action_dict: Dict) -> List[float]:
    """Extract numeric PCTL values from action dictionary."""
    values = []
    if not isinstance(action_dict, dict):
        return values

    for value in action_dict.values():
        v = _get_pctl_value(value)
        if v is not None:
            values.append(v)
    return values


def _get_pctl_value(value: Any) -> Optional[float]:
    """Extract numeric value from PCTL result (handles nested dict format)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        if 'pctl_value' in value:
            pctl_val = value['pctl_value']
            if isinstance(pctl_val, (int, float)):
                return float(pctl_val)
            elif isinstance(pctl_val, dict) and 'probability' in pctl_val:
                return float(pctl_val.get('probability', 0))
    return None


def _has_significant_variation(values: List[float], threshold: float = 0.0001) -> bool:
    """Check if values have significant variation."""
    if len(values) < 2:
        return False
    return (max(values) - min(values)) > threshold


def _check_for_eval_errors(action_results: Dict) -> Optional[str]:
    """Check for evaluation errors and evidence warnings in action results."""
    if not isinstance(action_results, dict):
        return None

    errors = []
    for action, result in action_results.items():
        if isinstance(result, dict):
            status = result.get('status', '')
            if status in ('parse_error', 'error'):
                error_msg = result.get('error', status)
                errors.append(f"{action}: {error_msg}")

    if errors:
        return "; ".join(errors[:5])
    return None


# ============================================================================
# TREE FORMATTING
# ============================================================================

def _format_tree_markdown(tree_node) -> str:
    """Format MCTS tree as indented markdown for frontend display."""
    def _attr(node, attr, default=None):
        if hasattr(node, attr):
            return getattr(node, attr, default)
        if isinstance(node, dict):
            return node.get(attr, default)
        return default

    root_visits = _attr(tree_node, 'visits', 0)
    children = _attr(tree_node, 'children', []) or []

    lines = [f"Root ({root_visits} visits)"]

    for i, child in enumerate(children):
        action = _attr(child, 'action')
        visits = _attr(child, 'visits', 0)
        q_val = _attr(child, 'q_value', 0.0)
        name = f"V{action}" if action is not None else "?"
        pct = (visits / root_visits * 100) if root_visits > 0 else 0

        is_last = (i == len(children) - 1)
        branch = "└─" if is_last else "├─"
        cont = "   " if is_last else "│  "

        lines.append(f"{branch} {name}: {visits} visits ({pct:.1f}%), Q={q_val:.3f}")

        # Show grandchildren summary (one level deeper)
        grandchildren = _attr(child, 'children', []) or []
        if grandchildren:
            gc_visits = sum(_attr(gc, 'visits', 0) for gc in grandchildren)
            gc_values = [_attr(gc, 'value', 0.0) for gc in grandchildren]
            avg_v = sum(gc_values) / len(gc_values) if gc_values else 0.0
            lines.append(f"{cont}  └─ {len(grandchildren)} outcome(s), "
                         f"{gc_visits} visits, avg value={avg_v:.1f}")

    # Wrap in fenced code block so markdown preserves indentation
    return "```\n" + "\n".join(lines) + "\n```"


# ============================================================================
# MAIN ORCHESTRATOR CLASS
# ============================================================================

class NSXAIOrchestrator:
    """Main orchestrator for the NS-XAI explanation system (Paratransit focused)."""

    def __init__(self, config):
        """
        Initialize orchestrator with use case configuration.

        Args:
            config: Use case configuration object (ParatransitConfig)
        """
        self.config = config
        self.pctl_templates = config.PCTL_TEMPLATES

    def process_query(self, query: str, mcts_root, env, mcts_data: Optional[Dict] = None,
                      epoch: Optional[int] = None) -> Dict[str, Any]:
        """
        Main processing pipeline: Query -> Classification -> PCTL -> Results

        Args:
            query: User's natural language query
            mcts_root: Root of MCTS tree (may be None for paratransit)
            env: Environment adapter
            mcts_data: Full MCTS data with per_step_trees, scenario_data, etc.
            epoch: If provided, use this epoch directly (skips query text parsing)

        Returns:
            Dict with query, classification, formulas, and PCTL results
        """
        # Classification returns (type_id, level) tuple
        type_id, level = classify_query_with_openai(
            query,
            prompt_id=self.config.QUERY_CLASSIFICATION_PROMPT_ID
        )

        pctl_formulas = self.pctl_templates.get(type_id, self.pctl_templates.get(-1, []))

        # Evaluate formulas using MDP comparison
        pctl_results = self._evaluate_paratransit_formulas(
            pctl_formulas, mcts_data, env, query, level, epoch=epoch
        )

        return {
            "query": query,
            "classification_label": type_id,
            "classification_level": level,
            "pctl_formulas": pctl_formulas,
            "pctl_results": pctl_results
        }

    def _evaluate_paratransit_formulas(
        self,
        formulas: List[str],
        mcts_data: Dict,
        env,
        query: str,
        level: Optional[str] = None,
        epoch: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate PCTL formulas for Paratransit using MDP comparison.

        Paratransit is non-stationary by design - always uses MDP_t vs MDP_{t-n} comparison
        to demonstrate how model adaptation affects decisions.

        Args:
            formulas: PCTL formulas to evaluate
            mcts_data: Full MCTS data with per_step_trees, scenario_data, etc.
            env: Environment adapter
            query: User query string
            level: Evaluation level from LLM (currently unused, always MDP comparison)
            epoch: If provided, use this epoch directly (skips query text parsing)

        Returns:
            List of evaluation results with MDP comparison data
        """
        if not mcts_data:
            return [{'note': 'No MCTS data available'}]

        scenario_data = mcts_data.get('scenario_data', {})
        rt = mcts_data.get('_runtime', {})

        # Use explicit epoch if provided, otherwise extract from query text
        if epoch is None:
            if hasattr(self.config, 'extract_epoch_from_query'):
                scenario_type = scenario_data.get('type')
                epoch = self.config.extract_epoch_from_query(query, scenario_type)
            if epoch is None:
                epoch = rt.get('latest_epoch', 0)

        # Fast path: use precomputed cache if available for this epoch
        precomputed = rt.get('formula_results_by_epoch', {}).get(epoch)
        if precomputed is not None:
            scenario_type = scenario_data.get('type', '')
            results = []
            for f in formulas:
                item = precomputed.get(f)
                if item is not None:
                    results.append(item)
                else:
                    # Check if this formula was intentionally excluded for this scenario
                    # (e.g. EVENT_MULTIPLIER not cached for counter_intuitive case)
                    if f.startswith('DERIVED:'):
                        metric = f.replace('DERIVED:', '').strip()
                        from core.formula_evaluator import SCENARIO_SPECIFIC_CONSTANTS
                        allowed_prefixes = SCENARIO_SPECIFIC_CONSTANTS.get(metric)
                        if allowed_prefixes and not any(scenario_type.startswith(p) for p in allowed_prefixes):
                            continue  # Skip — not applicable to this scenario
                    # Formula should be in cache but isn't — fall back
                    results = None
                    break
            if results is not None:
                return results

        # Always use MDP comparison for paratransit (non-stationarity)
        # Always allow rebuild to enable complete comparison for any queried epoch
        return evaluate_mdp_comparison(
            formulas, mcts_data, env, query, epoch, allow_rebuild=True
        )

    def _action_to_name(self, action) -> str:
        """Convert action number to action name using config."""
        if hasattr(self.config, 'action_to_name'):
            return self.config.action_to_name(action)
        if isinstance(action, int):
            return f"V{action}"
        return str(action)

    def _get_query_specific_node(self, query: str, mcts_root):
        """Navigate to the MCTS node that corresponds to the query's epoch."""
        from core.formula_evaluator import _MCTS_DATA

        if hasattr(self.config, 'extract_epoch_from_query'):
            epoch = self.config.extract_epoch_from_query(query)
            if epoch is not None and _MCTS_DATA:
                rt = _MCTS_DATA.get('_runtime', {})
                trees_by_epoch = rt.get('trees_by_epoch', {})
                if epoch in trees_by_epoch:
                    return trees_by_epoch[epoch]
        return mcts_root

    def _extract_visit_info_from_mcts(self, mcts_root) -> str:
        """Extract visit counts directly from MCTS root's children."""
        if not mcts_root or not hasattr(mcts_root, 'children') or not mcts_root.children:
            return ""

        visits_by_action_id = {}
        for child in mcts_root.children:
            if hasattr(child, 'action') and hasattr(child, 'visits'):
                action_id = child.action
                visits_by_action_id[action_id] = visits_by_action_id.get(action_id, 0) + child.visits

        if not visits_by_action_id:
            return ""

        total_visits = sum(visits_by_action_id.values())
        if total_visits == 0:
            return ""

        lines = []
        for action_id in sorted(visits_by_action_id.keys()):
            action_name = self._action_to_name(action_id)
            visits = visits_by_action_id[action_id]
            percentage = (visits / total_visits * 100) if total_visits > 0 else 0
            lines.append(f"  {action_name}: {visits} visits ({percentage:.1f}%)")

        return "\n".join(lines)

    def _format_value(self, value: Any, formula: str) -> str:
        """Unified value formatter for PCTL results."""
        if value is None:
            return "N/A"

        if isinstance(value, dict) and 'pctl_value' in value:
            pctl_val = value['pctl_value']
            if pctl_val is None:
                return "N/A"
            if isinstance(pctl_val, float):
                formatted = f"{pctl_val:.4e}" if abs(pctl_val) < 0.001 and pctl_val != 0 else f"{pctl_val:.4f}"
            else:
                formatted = str(pctl_val)
            return formatted

        if isinstance(value, float):
            return f"{value:.4e}" if abs(value) < 0.001 and value != 0 else f"{value:.4f}"

        return str(value)

    def format_for_explanation(self, result: Dict[str, Any], stationary: bool = False,
                               id_map: Optional[Dict[int, int]] = None) -> str:
        """
        Format PCTL results for explanation generation.

        Args:
            result: PCTL evaluation results.
            stationary: If True, only include MDP_t results (no MDP_{t-n} comparison).
                        Used for user study baseline (stationary explanation).
            id_map: Optional original_index -> display_id mapping for user study renumbering.
        """
        if not result['pctl_results']:
            return "No differentiating metrics found."

        pctl_metrics = []
        notes = []

        for pctl_result in result['pctl_results']:
            if 'note' in pctl_result:
                notes.append(pctl_result['note'])
            elif pctl_result.get('formula') == 'NOTE':
                notes.append(str(pctl_result.get('result', '')))
            else:
                result_value = pctl_result.get('result', {})
                formula = pctl_result.get('formula', '')
                has_errors = False

                # Check for errors in MDP comparison format
                if isinstance(result_value, dict) and 'mdp_t' in result_value:
                    mdp_t_errors = _check_for_eval_errors(result_value.get('mdp_t', {}))
                    if mdp_t_errors:
                        has_errors = True
                    if result_value.get('mdp_t_error'):
                        has_errors = True
                    if not stationary:
                        mdp_t_n_errors = _check_for_eval_errors(result_value.get('mdp_t_minus_n', {}))
                        if mdp_t_n_errors:
                            has_errors = True
                        if result_value.get('mdp_t_minus_n_error'):
                            has_errors = True

                # Filtering: stationary checks only MDP_t variation
                # DERIVED metrics are always kept (query-relevant context)
                if stationary:
                    if formula.startswith('DERIVED:'):
                        pctl_metrics.append(pctl_result)
                    elif isinstance(result_value, dict) and 'mdp_t' in result_value:
                        mdp_t = result_value.get('mdp_t')
                        if isinstance(mdp_t, dict):
                            t_values = _extract_pctl_values_from_actions(mdp_t)
                            if _has_significant_variation(t_values) or has_errors:
                                pctl_metrics.append(pctl_result)
                        else:
                            pctl_metrics.append(pctl_result)
                    elif should_keep_pctl_result(result_value, formula) or has_errors:
                        pctl_metrics.append(pctl_result)
                else:
                    if should_keep_pctl_result(result_value, formula) or has_errors:
                        pctl_metrics.append(pctl_result)

        if not pctl_metrics:
            if notes:
                return "⚠ " + "\n⚠ ".join(notes)
            return "No differentiating metrics found."

        lines = []
        for note in notes:
            lines.append(f"⚠ {note}")

        for pctl_result in pctl_metrics:
            result_value = pctl_result['result']
            formula = pctl_result['formula']

            if isinstance(result_value, dict) and 'mdp_t' in result_value:
                lines.append(f"  {formula}:")

                # Format MDP_t results
                mdp_t = result_value.get('mdp_t', {})
                mdp_t_label = "Current model" if stationary else "MDP_t (current model)"
                if isinstance(mdp_t, dict) and mdp_t:
                    formatted_t = [f"{k}={self._format_value(v, formula)}"
                                   for k, v in mdp_t.items() if v is not None]
                    lines.append(f"    {mdp_t_label}: {{{', '.join(formatted_t)}}}")
                else:
                    lines.append(f"    {mdp_t_label}: {self._format_value(mdp_t, formula)}")

                # Format MDP_{t-n} results (non-stationary only)
                if not stationary:
                    mdp_t_minus_n = result_value.get('mdp_t_minus_n', {})
                    prev_epoch = result_value.get('prev_epoch')
                    prev_version = result_value.get('prev_version_id')
                    version_label = f"version {prev_version}" if prev_version is not None else "previous"
                    label_suffix = f" ({version_label})"

                    if isinstance(mdp_t_minus_n, dict) and mdp_t_minus_n:
                        formatted_t_n = [f"{k}={self._format_value(v, formula)}"
                                         for k, v in mdp_t_minus_n.items() if v is not None]
                        lines.append(f"    MDP_{{t-n}}{label_suffix}: {{{', '.join(formatted_t_n)}}}")
                    else:
                        lines.append(f"    MDP_{{t-n}}{label_suffix}: {self._format_value(mdp_t_minus_n, formula)}")

            elif isinstance(result_value, dict):
                formatted = [f"{label}={self._format_value(val, formula)}"
                            for label, val in result_value.items()]
                result_str = "{" + ", ".join(formatted) + "}"
                lines.append(f"  {formula}: {result_str}")
            else:
                result_str = self._format_value(result_value, formula)
                lines.append(f"  {formula}: {result_str}")

        return "\n".join(lines)

    # ================================================================
    # PCTL FAMILY COMPRESSION (non-stationary prompt only)
    # ================================================================

    # --- Support-aware prefilter polarity sets ---
    _HIGHER_BETTER = {'service_complete', 'G (!pickup_delay & !dropoff_delay)'}
    _LOWER_BETTER = {
        'pickup_delay', 'dropoff_delay', 'pickup_delay_ge_', 'dropoff_delay_ge_',
        'capacity_violation', 'vehicle_busy_before_assignment',
        'clear_current_route_ge_', 'deadhead_to_pickup_ge_',
        'dropoff_slack_le_', 'service_path_event_affected', 'service_path_bridge_affected',
    }
    _NO_SUPPORT_CHECK = {'carpool_active', 'capacity_full', 'any_vehicle_idle', 'all_vehicles_busy'}

    # Family mapping: formula substring -> family name
    _PCTL_FAMILY_MAP = {
        'service_complete': 'service_success',
        'pickup_delay': 'pickup_delay',
        'dropoff_delay': 'dropoff_delay',
        'capacity_violation': 'capacity_carpool',
        'capacity_full': 'capacity_carpool',
        'carpool_active': 'capacity_carpool',
        'all_vehicles_busy': 'capacity_carpool',
        'any_vehicle_idle': 'capacity_carpool',
        'vehicle_busy_before_assignment': 'route_burden',
        'clear_current_route': 'route_burden',
        'deadhead_to_pickup': 'route_burden',
        'dropoff_slack': 'slack_disruption',
        'service_path_event_affected': 'slack_disruption',
        'service_path_bridge_affected': 'slack_disruption',
    }

    # Families guaranteed to keep at least one representative
    _BALANCED_ANCHORS = ('service_success', 'pickup_delay', 'dropoff_delay')

    @classmethod
    def _classify_pctl_family(cls, formula: str) -> Optional[str]:
        """Map a PCTL formula to its family name, or None for DERIVED."""
        if formula.startswith('DERIVED:'):
            return None
        for substr, family in cls._PCTL_FAMILY_MAP.items():
            if substr in formula:
                return family
        return None

    @classmethod
    def _classify_polarity(cls, formula: str) -> Optional[str]:
        """Classify formula polarity for support checking.

        Returns 'higher_better', 'lower_better', or None (skip support check).
        Check order matters: _HIGHER_BETTER before _LOWER_BETTER so that
        'G (!pickup_delay & !dropoff_delay)' matches as higher_better before
        bare 'pickup_delay' matches as lower_better.
        """
        if formula.startswith('DERIVED:'):
            return None
        for substr in cls._NO_SUPPORT_CHECK:
            if substr in formula:
                return None
        for substr in cls._HIGHER_BETTER:
            if substr in formula:
                return 'higher_better'
        for substr in cls._LOWER_BETTER:
            if substr in formula:
                return 'lower_better'
        return None

    @staticmethod
    def _resolve_support_mode(assigned_id: int,
                              vehicle_targets: Optional[List[int]]) -> tuple:
        """Determine support comparison mode based on query targets.

        Returns (mode, comparator_id):
          ('unique_best', None)  – assigned must beat all others
          ('pairwise', target)   – assigned must beat specific target
          ('skip', None)         – multi-target non-assigned, skip gate
        """
        if not vehicle_targets or vehicle_targets == [assigned_id]:
            return ('unique_best', None)
        if len(vehicle_targets) == 1 and vehicle_targets[0] != assigned_id:
            return ('pairwise', vehicle_targets[0])
        return ('skip', None)

    def _check_support(self, mdp_t: Dict, focal_key: str, polarity: str,
                       mode: str, comparator_key: Optional[str]) -> bool:
        """Check if focal vehicle's MDP_t value supports its assignment.

        Returns True to keep the formula, False to drop it.
        """
        if mode == 'skip' or polarity is None:
            return True

        focal_val = _get_pctl_value(mdp_t.get(focal_key))
        if focal_val is None:
            return True

        if mode == 'unique_best':
            for key, raw_val in mdp_t.items():
                if key == focal_key:
                    continue
                other_val = _get_pctl_value(raw_val)
                if other_val is None:
                    return True
                if polarity == 'higher_better' and focal_val <= other_val:
                    return False
                if polarity == 'lower_better' and focal_val >= other_val:
                    return False
            return True

        if mode == 'pairwise':
            comp_val = _get_pctl_value(mdp_t.get(comparator_key))
            if comp_val is None:
                return True
            if polarity == 'higher_better':
                return focal_val > comp_val
            if polarity == 'lower_better':
                return focal_val < comp_val

        return True

    def _score_pctl_item(self, item: Dict, assigned_key: str, closest_key: str) -> float:
        """Score a PCTL result item by four-term informativeness.

        score = within_mdp_t_spread + within_mdp_tn_spread
              + cross_mdp_shift + assigned_vs_closest_gap
        """
        result = item.get('result', {})
        mdp_t = result.get('mdp_t', {})
        mdp_tn = result.get('mdp_t_minus_n')

        def _spread(action_dict):
            vals = [_get_pctl_value(v) for v in action_dict.values() if _get_pctl_value(v) is not None]
            return (max(vals) - min(vals)) if len(vals) >= 2 else 0.0

        def _assigned_closest_gap(action_dict):
            va = _get_pctl_value(action_dict.get(assigned_key))
            vc = _get_pctl_value(action_dict.get(closest_key))
            if va is not None and vc is not None:
                return abs(va - vc)
            return 0.0

        def _cross_shift(a_dict, b_dict):
            if not b_dict or not isinstance(b_dict, dict):
                return 0.0
            total = 0.0
            n = 0
            for k in set(a_dict.keys()) & set(b_dict.keys()):
                va = _get_pctl_value(a_dict.get(k))
                vb = _get_pctl_value(b_dict.get(k))
                if va is not None and vb is not None:
                    total += abs(va - vb)
                    n += 1
            return total / n if n else 0.0

        t_spread = _spread(mdp_t) if isinstance(mdp_t, dict) else 0.0
        tn_spread = _spread(mdp_tn) if isinstance(mdp_tn, dict) else 0.0
        cross = _cross_shift(mdp_t, mdp_tn) if isinstance(mdp_t, dict) else 0.0
        gap = _assigned_closest_gap(mdp_t) if isinstance(mdp_t, dict) else 0.0
        if isinstance(mdp_tn, dict):
            gap += _assigned_closest_gap(mdp_tn)

        return t_spread + tn_spread + cross + gap

    def _compress_pctl_results(self, pctl_results: List[Dict],
                               mcts_data: Dict, epoch: int,
                               vehicle_targets: List[int] = None) -> List[Dict]:
        """Apply support-aware prefilter to PCTL results.

        Drops formulas whose MDP_t values do not support the assigned
        vehicle being the best choice (by the formula's polarity).

        Family-based Top-1 compression is preserved below but disabled.
        """
        # --- Phase 1: Support-aware prefilter ---
        assigned_key, closest_key = self._resolve_vehicle_keys(mcts_data, epoch)
        assigned_id = int(assigned_key[1:])
        mode, comparator_id = self._resolve_support_mode(assigned_id, vehicle_targets)
        comparator_key = f"V{comparator_id}" if comparator_id is not None else None

        filtered = []
        for item in pctl_results:
            if 'note' in item and 'formula' not in item:
                filtered.append(item)
                continue

            formula = item.get('formula', '')

            if formula.startswith('DERIVED:'):
                filtered.append(item)
                continue

            polarity = self._classify_polarity(formula)
            if polarity is None:
                filtered.append(item)
                continue

            mdp_t = item.get('result', {}).get('mdp_t', {})
            if not isinstance(mdp_t, dict):
                filtered.append(item)
                continue

            if self._check_support(mdp_t, assigned_key, polarity, mode, comparator_key):
                filtered.append(item)

        # --- Phase 2: Family-based Top-1 compression ---
        families: Dict[str, List[Dict]] = {}
        derived_items: List[Dict] = []
        note_items: List[Dict] = []

        for item in filtered:
            if 'note' in item and 'formula' not in item:
                note_items.append(item)
                continue
            formula = item.get('formula', '')
            family = self._classify_pctl_family(formula)
            if family is None:
                derived_items.append(item)
            else:
                families.setdefault(family, []).append(item)

        # Score and pick Top-1 per family
        best_per_family: Dict[str, Dict] = {}
        scores_per_family: Dict[str, float] = {}
        for family, items in families.items():
            scored = [(self._score_pctl_item(it, assigned_key, closest_key), it)
                      for it in items]
            scored.sort(key=lambda x: -x[0])
            best_per_family[family] = scored[0][1]
            scores_per_family[family] = scored[0][0]

        # Balanced anchors: always include these families if they exist
        selected = {}
        for anchor in self._BALANCED_ANCHORS:
            if anchor in best_per_family:
                selected[anchor] = best_per_family[anchor]

        # route_burden vs slack_disruption: pick the one with higher score
        rb_score = scores_per_family.get('route_burden', -1)
        sd_score = scores_per_family.get('slack_disruption', -1)
        if rb_score >= sd_score and 'route_burden' in best_per_family:
            selected['route_burden'] = best_per_family['route_burden']
        elif 'slack_disruption' in best_per_family:
            selected['slack_disruption'] = best_per_family['slack_disruption']

        # Also include capacity_carpool if it exists and not yet selected
        if 'capacity_carpool' in best_per_family and 'capacity_carpool' not in selected:
            selected['capacity_carpool'] = best_per_family['capacity_carpool']

        # Assemble: notes + selected PCTL (in stable family order) + derived
        family_order = ['service_success', 'pickup_delay', 'dropoff_delay',
                        'capacity_carpool', 'route_burden', 'slack_disruption']
        compressed = list(note_items)
        for fam in family_order:
            if fam in selected:
                compressed.append(selected[fam])
        compressed.extend(derived_items)
        return compressed

    @staticmethod
    def _resolve_vehicle_keys(mcts_data: Dict, epoch: int):
        """Return (assigned_key, closest_key) like ('V4', 'V0')."""
        scenario_data = mcts_data.get('scenario_data', {})
        mdp_comp = scenario_data.get('mdp_comparisons', {}).get(epoch, {})
        ci = mdp_comp.get('counter_intuitive_info', {})
        ei = mdp_comp.get('event_info', {})
        ad_list = mcts_data.get('env_data', {}).get('assignment_details', [])
        ad = ad_list[epoch] if isinstance(ad_list, list) and epoch < len(ad_list) else (
            ad_list.get(epoch, {}) if isinstance(ad_list, dict) else {}
        )

        assigned = ci.get('assigned_vehicle') or ei.get('mdp_t_assignment') or ad.get('assigned_vehicle')
        closest = ci.get('closest_vehicle') or ad.get('closest_vehicle')

        a_key = f"V{assigned}" if assigned is not None else "V0"
        c_key = f"V{closest}" if closest is not None else "V1"
        return a_key, c_key

    # ================================================================
    # DERIVED METRICS GROUPED SUMMARY (non-stationary prompt only)
    # ================================================================

    _DERIVED_GROUPS = {
        'System Estimated Travel-Time': [
            'DERIVED: ETA_PICKUP_BY_VEHICLE', 'DERIVED: ETA_DROPOFF_BY_VEHICLE',
        ],
        'Workload summary': [
            'DERIVED: PENDING_REQUESTS_BY_VEHICLE',
            'DERIVED: VEHICLE_OCCUPANCY',
        ],
        'Route burden summary': [
            'DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE',
            'DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE',
            'DERIVED: DROPOFF_SLACK_BY_VEHICLE',
            'DERIVED: SERVICE_PATH_DISRUPTION_BY_VEHICLE',
        ],
        'Global context summary': [
            'DERIVED: TRAFFIC_LEVEL', 'DERIVED: STEP_CONFIDENCE',
            'DERIVED: TURNING_INTERVAL',
        ],
    }

    _DERIVED_UNITS = {
        'ETA_PICKUP_BY_VEHICLE': 'minutes',
        'ETA_DROPOFF_BY_VEHICLE': 'minutes',
        'CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE': 'minutes',
        'DEADHEAD_TO_PICKUP_BY_VEHICLE': 'minutes',
        'DROPOFF_SLACK_BY_VEHICLE': 'minutes',
    }

    def _build_derived_summary(self, derived_items: List[Dict],
                               id_map: Optional[Dict[int, int]] = None) -> str:
        """Format DERIVED results into grouped summary blocks.

        Derived metrics are always kept (they provide query-relevant context,
        not only differential evidence).
        """
        # Build formula -> item lookup
        by_formula = {it['formula']: it for it in derived_items if 'formula' in it}
        if not by_formula:
            return ""

        blocks = []
        for group_name, formulas in self._DERIVED_GROUPS.items():
            group_items = [(f, by_formula[f]) for f in formulas if f in by_formula]
            if not group_items:
                continue

            # Build compact lines
            lines = [f"  [{group_name}]"]
            for formula, item in group_items:
                r = item.get('result', {})
                short = formula.replace('DERIVED: ', '')
                unit = self._DERIVED_UNITS.get(short)
                if unit:
                    short = f"{short} ({unit})"
                v_t = r.get('mdp_t')
                v_tn = r.get('mdp_t_minus_n')

                # Dedicated TURNING_INTERVAL formatter
                if short == 'TURNING_INTERVAL' and isinstance(v_t, dict):
                    display_iv = v_t.get('display_interval')
                    req_iv = v_t.get('recovery_request_interval')
                    if display_iv:
                        fmt_t = f"request {display_iv[0]}-{display_iv[1]}"
                    elif req_iv:
                        fmt_t = f"request {req_iv[0]}-{req_iv[1]}"
                    else:
                        fmt_t = "N/A"
                    lines.append(f"    {short}: {fmt_t}")
                    continue

                # Format value, preserving categorical strings
                fmt_t = self._format_derived_val(v_t)
                fmt_tn = self._format_derived_val(v_tn)

                if v_tn is not None:
                    lines.append(f"    {short}: t={fmt_t} | t-n={fmt_tn}")
                else:
                    lines.append(f"    {short}: {fmt_t}")
            blocks.append("\n".join(lines))

        return "\n".join(blocks)

    @staticmethod
    def _format_derived_val(val) -> str:
        if val is None:
            return "N/A"
        if isinstance(val, dict):
            # Some derived metrics return dicts like {'value': x, 'type': 'constant'}
            if 'value' in val:
                v = val['value']
                if isinstance(v, float):
                    return f"{v:.2f}"
                return str(v)
            # BY_VEHICLE dicts: {"V0": 12.34, "V1": None, ...}
            # Format each value with proper numeric formatting
            def _fmt_entry(v):
                if v is None:
                    return "N/A"
                if isinstance(v, float):
                    return f"{v:.2f}"
                return str(v)
            return ", ".join(f"{k}={_fmt_entry(v)}" for k, v in val.items())
        if isinstance(val, float):
            return f"{val:.2f}"
        return str(val)

    # ================================================================
    # ALGORITHM STATE BLOCK (non-stationary prompt only)
    # ================================================================

    _TRIGGER_EPOCH = 10
    _RECOVERY_END_EPOCH = 19

    @classmethod
    def _build_algorithm_state_block(cls, mcts_data: Dict, epoch: int) -> str:
        """Three-line Algorithm State block for the non-stationary prompt.

        Environment change + phase from hardcoded Case 0 boundaries;
        confidence from dpas_history[epoch]['regular_pct'].
        """
        if epoch < cls._TRIGGER_EPOCH:
            env_changed = "no"
            phase = "pre-change stable"
        elif epoch <= cls._RECOVERY_END_EPOCH:
            env_changed = "yes"
            phase = "post-change adapting"
        else:
            env_changed = "yes"
            phase = "post-change stable"

        scenario_data = mcts_data.get('scenario_data', {})
        dpas_history = (scenario_data.get('dpas_history')
                        or mcts_data.get('env_data', {}).get('dpas_history', {}))
        entry = dpas_history.get(epoch, {}) if isinstance(dpas_history, dict) else {}
        regular_pct = entry.get('regular_pct')
        if regular_pct is None:
            confidence = "high"
        else:
            confidence = "high" if regular_pct >= 0.5 else "low"

        return (
            "Algorithm State:\n"
            f"  Environment changed: {env_changed}\n"
            f"  Algorithm phase: {phase}\n"
            f"  Planner confidence: {confidence}"
        )

    # ================================================================
    # SAME-REQUEST EXPLANATION HISTORY (non-stationary only)
    # ================================================================

    _HISTORY_MAX_TURNS = 2
    _HISTORY_TRUNC_CHARS = 600

    @staticmethod
    def _get_ns_history_store(mcts_data: Dict) -> Dict[int, List[Dict[str, str]]]:
        rt = mcts_data.setdefault('_runtime', {})
        return rt.setdefault('ns_explanation_history', {})

    @classmethod
    def _build_history_block(cls, mcts_data: Dict, epoch: int) -> str:
        """Format prior NS turns for this epoch as a prompt block.

        Returns "" if there is no history for this epoch.
        """
        store = cls._get_ns_history_store(mcts_data)
        turns = store.get(epoch, [])
        if not turns:
            return ""
        lines = ["Already shown to the user in this request:"]
        for t in turns:
            q = t.get('query', '').strip()
            text = (t.get('text') or '').strip()
            if len(text) > cls._HISTORY_TRUNC_CHARS:
                text = text[:cls._HISTORY_TRUNC_CHARS].rstrip() + "…"
            lines.append(f"  Previous query: {q}")
            lines.append(f"  Previous explanation (condensed):")
            for ln in text.splitlines():
                lines.append(f"    {ln}")
            lines.append("")
        return "\n".join(lines).rstrip()

    @classmethod
    def _record_ns_explanation(cls, mcts_data: Dict, epoch: int,
                                query: str, text: str) -> None:
        """Append a (query, text) turn to the NS history for this epoch.

        Dedups by normalized query (case-insensitive, stripped), keeps last N.
        Skips error strings.
        """
        if not text or not isinstance(text, str):
            return
        lowered = text.lower()
        if lowered.startswith('unable to generate explanation') or \
           lowered.startswith('error generating explanation'):
            return
        store = cls._get_ns_history_store(mcts_data)
        turns = store.setdefault(epoch, [])
        norm_q = (query or '').strip().lower()
        turns[:] = [t for t in turns if (t.get('query') or '').strip().lower() != norm_q]
        turns.append({'query': query, 'text': text})
        if len(turns) > cls._HISTORY_MAX_TURNS:
            del turns[:-cls._HISTORY_MAX_TURNS]

    # ================================================================
    # PROMPT BUILDING
    # ================================================================

    @staticmethod
    def _filter_results_by_vehicles(pctl_results: List[Dict], vehicle_keys: List[str]) -> List[Dict]:
        """Filter PCTL and DERIVED result dicts to only include specified vehicles.

        For PCTL results: filters mdp_t / mdp_t_minus_n action dicts.
        For DERIVED BY_VEHICLE results: filters the per-vehicle dict values.

        Args:
            pctl_results: List of result dicts from process_query.
            vehicle_keys: List of vehicle key strings like ["V3", "V4"].
        """
        if not vehicle_keys:
            return pctl_results

        vk_set = set(vehicle_keys)
        filtered = []
        for item in pctl_results:
            if 'note' in item and 'formula' not in item:
                filtered.append(item)
                continue

            result_value = item.get('result', {})
            if not isinstance(result_value, dict) or 'mdp_t' not in result_value:
                filtered.append(item)
                continue

            new_result = dict(result_value)

            # Filter mdp_t
            mdp_t = result_value.get('mdp_t')
            if isinstance(mdp_t, dict):
                # Check if this is a vehicle-keyed dict (keys like "V0", "V1")
                if any(k.startswith('V') and k[1:].isdigit() for k in mdp_t):
                    new_result['mdp_t'] = {k: v for k, v in mdp_t.items() if k in vk_set}

            # Filter mdp_t_minus_n
            mdp_tn = result_value.get('mdp_t_minus_n')
            if isinstance(mdp_tn, dict):
                if any(k.startswith('V') and k[1:].isdigit() for k in mdp_tn):
                    new_result['mdp_t_minus_n'] = {k: v for k, v in mdp_tn.items() if k in vk_set}

            filtered.append({**item, 'result': new_result})

        return filtered

    def _build_explanation_prompt(self, query: str, result: Dict[str, Any], env_context: str,
                                   tree_node=None, stationary: bool = False,
                                   id_map: Optional[Dict[int, int]] = None,
                                   mcts_data: Optional[Dict] = None,
                                   epoch: Optional[int] = None,
                                   vehicle_targets: Optional[List[int]] = None) -> str:
        """Build LLM input string for explanation generation.

        Args:
            tree_node: Pre-resolved MCTS tree node for the query's epoch.
            stationary: If True, format for stationary explanation (MDP_t only, no comparison).
            id_map: Optional original_index -> display_id mapping for user study renumbering.
            mcts_data: Needed for non-stationary PCTL compression (vehicle key resolution).
            epoch: Needed for non-stationary PCTL compression.
            vehicle_targets: Resolved vehicle IDs to filter evidence by (empty = all vehicles).
        """
        if not stationary and mcts_data is not None and epoch is not None:
            # Non-stationary path: apply family compression + derived summary
            pctl_results = result.get('pctl_results', [])

            # Apply vehicle targeting filter if targets are specified
            support_targets = vehicle_targets
            if vehicle_targets:
                # For single-target queries, expand so both sides of the
                # comparison are present in evidence and support filtering.
                filter_ids = list(vehicle_targets)
                skip_filter = False
                if len(filter_ids) == 1:
                    assigned_key, closest_key = self._resolve_vehicle_keys(mcts_data, epoch)
                    assigned_id = int(assigned_key[1:])
                    closest_id = int(closest_key[1:])
                    if filter_ids[0] == assigned_id:
                        # "why V4?" where V4 is assigned — show all vehicles
                        # so the user sees the full comparative evidence.
                        skip_filter = True
                    else:
                        # "why not car3?" — add assigned for display
                        filter_ids.insert(0, assigned_id)
                if not skip_filter:
                    vehicle_keys = [f"V{v}" for v in filter_ids]
                    pctl_results = self._filter_results_by_vehicles(pctl_results, vehicle_keys)

            compressed = self._compress_pctl_results(pctl_results, mcts_data, epoch, support_targets)

            # Split into PCTL and DERIVED
            pctl_only = [it for it in compressed if not it.get('formula', '').startswith('DERIVED:')]
            derived_only = [it for it in compressed if it.get('formula', '').startswith('DERIVED:')]

            # Format compressed PCTL via existing formatter
            compressed_result = dict(result)
            compressed_result['pctl_results'] = pctl_only
            formatted_pctl = self.format_for_explanation(compressed_result, stationary=False, id_map=id_map)

            # Format derived as grouped summary
            derived_summary = self._build_derived_summary(derived_only, id_map=id_map)

            algo_state = self._build_algorithm_state_block(mcts_data, epoch)
            history_block = self._build_history_block(mcts_data, epoch)

            llm_input = f"Query: {query}\n\n"
            if history_block:
                llm_input += f"{history_block}\n\n"
            llm_input += f"{algo_state}\n\n"
            llm_input += f"PCTL Analysis Results:\n{formatted_pctl}\n\n"
            if derived_summary:
                llm_input += f"Derived Metrics Summary:\n{derived_summary}\n\n"
            return llm_input
        else:
            # Stationary path or fallback: use existing formatting unchanged
            formatted_results = self.format_for_explanation(result, stationary=stationary, id_map=id_map)
            visit_info = self._extract_visit_info_from_mcts(tree_node) if tree_node else ""
            query_type = result.get('classification_label', 'unknown')

            llm_input = (
                f"Query: {query}\n"
                f"Query Type: {query_type}\n\n"
            )
            if visit_info:
                llm_input += f"Action Exploration Statistics:\n{visit_info}\n\n"
            llm_input += f"PCTL Analysis Results:\n{formatted_results}\n\n"
            return llm_input

    def _run_query_pipeline(self, query: str, mcts_data: Dict,
                            epoch: Optional[int] = None) -> Dict:
        """Shared query pipeline: setup + classification + PCTL evaluation + context.

        Returns dict with: result, epoch, tree_node, mcts_root, env, env_context,
                           id_map, resolved_targets, raw_refs.
        """
        # Ensure runtime artifacts are prepared (idempotent)
        prepare_runtime_artifacts(self.config, mcts_data)
        rt = mcts_data['_runtime']

        # Set configs
        set_active_config(self.config)
        set_formula_evaluator_config(self.config)
        set_evidence_cache({}, {}, mcts_data)

        # Cache env adapter (loads heavy travel matrix on first call)
        if 'cached_env' in rt:
            mcts_root, env = rt['cached_mcts_root'], rt['cached_env']
        else:
            mcts_root, env = self.config.load_ada_mcts_tree(mcts_data)
            rt['cached_mcts_root'] = mcts_root
            rt['cached_env'] = env

        # Renumber mapping for user study
        _scenario_type = mcts_data.get('scenario_data', {}).get('type')
        id_map = None
        if hasattr(self.config, 'get_request_id_mapping'):
            id_map = self.config.get_request_id_mapping(_scenario_type)

        # 1. Resolve epoch FIRST (needed for role resolution)
        if epoch is None:
            if hasattr(self.config, 'extract_epoch_from_query'):
                epoch = self.config.extract_epoch_from_query(query, _scenario_type)
            if epoch is None:
                epoch = rt['latest_epoch']

        # 2. Parse vehicle references (epoch-independent)
        raw_refs = []
        resolved_targets = []
        if hasattr(self.config, 'extract_vehicles_from_query'):
            n_vehicles = self.config.get_n_actions(mcts_data) if hasattr(self.config, 'get_n_actions') else 5
            raw_refs = self.config.extract_vehicles_from_query(query, n_vehicles)

        # 3. Resolve roles using epoch-specific assignment info
        if raw_refs:
            assigned_key, closest_key = self._resolve_vehicle_keys(mcts_data, epoch)
            assigned_id = int(assigned_key[1:])  # "V4" -> 4
            closest_id = int(closest_key[1:])    # "V0" -> 0
            if hasattr(self.config, 'resolve_vehicle_references'):
                resolved_targets = self.config.resolve_vehicle_references(
                    raw_refs, assigned_id, closest_id
                )

        # Process query (classification + PCTL evaluation)
        result = self.process_query(query, mcts_root, env, mcts_data, epoch=epoch)

        # Environment context — cache per epoch (formatted string is deterministic for a given epoch)
        env_ctx_cache = rt.setdefault('env_context_by_epoch', {})
        if epoch in env_ctx_cache:
            env_context = env_ctx_cache[epoch]
        else:
            if hasattr(self.config.format_environment_context, '__code__') and \
               'query' in self.config.format_environment_context.__code__.co_varnames:
                env_context = self.config.format_environment_context(mcts_data, query)
            else:
                env_context = self.config.format_environment_context(mcts_data)
            env_ctx_cache[epoch] = env_context

        # Get tree node for epoch (use int-keyed lookup)
        tree_node = rt['trees_by_epoch'].get(epoch)

        return {
            'result': result,
            'epoch': epoch,
            'tree_node': tree_node,
            'mcts_root': mcts_root,
            'env': env,
            'env_context': env_context,
            'id_map': id_map,
            'resolved_targets': resolved_targets,
            'raw_refs': raw_refs,
        }

    def _build_tree_str(self, tree_node, epoch: int) -> str:
        """Build tree structure string for display."""
        if not tree_node:
            return f"No tree found for epoch {epoch}"
        return _format_tree_markdown(tree_node)

    def explain_query_with_ada_mcts(self, query: str, mcts_data: Dict, skip_explanation: bool = False) -> str:
        """Process query using ADA-MCTS data and generate explanation.

        Args:
            skip_explanation: If True, skip OpenAI explanation generation and just output PCTL results.
        """
        try:
            pipe = self._run_query_pipeline(query, mcts_data)
            result, epoch, tree_node = pipe['result'], pipe['epoch'], pipe['tree_node']
            mcts_root, env_context, id_map = pipe['mcts_root'], pipe['env_context'], pipe['id_map']
            resolved_targets = pipe.get('resolved_targets', [])
            tree_str = self._build_tree_str(tree_node, epoch)

            # Build LLM inputs
            stationary_llm_input = build_logiex_prompt(
                query, result, env_context, mcts_data, id_map=id_map, epoch=epoch,
                vehicle_targets=resolved_targets)
            ns_llm_input = self._build_explanation_prompt(
                query, result, env_context, tree_node, id_map=id_map,
                mcts_data=mcts_data, epoch=epoch, vehicle_targets=resolved_targets)

            query_type = result.get('classification_label', 'unknown')

            # 1. Tree Structure
            print("\n" + "=" * 60)
            print("EXPLANATION TYPE 1: TREE STRUCTURE")
            print("=" * 60)
            print(tree_str)

            # 2. Stationary Explanation input
            print("\n" + "=" * 60)
            print("EXPLANATION TYPE 2: STATIONARY")
            print("=" * 60)
            print(f"Query Type: {query_type}")
            print("Stationary LLM Input:")
            print(stationary_llm_input)

            # 3. Non-Stationary Explanation (MDP_t vs MDP_{t-n} comparison)
            print("\n" + "=" * 60)
            print("EXPLANATION TYPE 3: NON-STATIONARY (MDP comparison)")
            print("=" * 60)
            print(f"Query Type: {query_type}")
            print("Non-Stationary LLM Input:")
            print(ns_llm_input)

            if skip_explanation:
                print("[Explanations skipped - skip_explanation=True]")
                return ns_llm_input

            # Parallel LLM calls (same as structured path)
            stationary_prompt_id = getattr(self.config, 'STATIONARY_EXPLANATION_PROMPT_ID', None)
            has_stationary = stationary_prompt_id and stationary_prompt_id != "TODO_REPLACE_WITH_ACTUAL_PROMPT_ID"

            from core.explanation_generation import generate_explanation_async
            import asyncio

            async def _run_parallel():
                tasks = []
                if has_stationary:
                    tasks.append(generate_explanation_async(stationary_llm_input, stationary_prompt_id))
                else:
                    async def _noop():
                        return "[Stationary explanation not configured]"
                    tasks.append(_noop())
                tasks.append(generate_explanation_async(ns_llm_input, self.config.EXPLANATION_GENERATION_PROMPT_ID))
                return await asyncio.gather(*tasks)

            llm_results = asyncio.run(_run_parallel())

            print("\n" + "=" * 60)
            print("GENERATED STATIONARY EXPLANATION")
            print("=" * 60)
            print(llm_results[0])

            print("\n" + "=" * 60)
            print("GENERATED NON-STATIONARY EXPLANATION")
            print("=" * 60)
            print(llm_results[1])

            self._record_ns_explanation(mcts_data, epoch, query, llm_results[1])
            return llm_results[1]

        except Exception as e:
            traceback.print_exc()
            return f"Unable to generate explanation: {str(e)}"

    def explain_query_structured(self, query: str, mcts_data: Dict) -> Dict:
        """Process query and return all 3 explanation types as a structured dict."""
        try:
            import time as _time
            _t0 = _time.time()

            pipe = self._run_query_pipeline(query, mcts_data)
            result, epoch, tree_node = pipe['result'], pipe['epoch'], pipe['tree_node']
            mcts_root, env_context, id_map = pipe['mcts_root'], pipe['env_context'], pipe['id_map']
            resolved_targets = pipe.get('resolved_targets', [])
            print(f"  [pipeline] {_time.time()-_t0:.1f}s", flush=True)

            tree_str = self._build_tree_str(tree_node, epoch)

            # Parallel LLM calls
            stationary_llm_input = build_logiex_prompt(
                query, result, env_context, mcts_data, id_map=id_map, epoch=epoch,
                vehicle_targets=resolved_targets)
            ns_llm_input = self._build_explanation_prompt(
                query, result, env_context, tree_node, id_map=id_map,
                mcts_data=mcts_data, epoch=epoch, vehicle_targets=resolved_targets)

            stationary_prompt_id = getattr(self.config, 'STATIONARY_EXPLANATION_PROMPT_ID', None)
            has_stationary = stationary_prompt_id and stationary_prompt_id != "TODO_REPLACE_WITH_ACTUAL_PROMPT_ID"

            from core.explanation_generation import generate_explanation_async
            import asyncio

            async def _run_parallel():
                tasks = []
                if has_stationary:
                    tasks.append(generate_explanation_async(stationary_llm_input, stationary_prompt_id))
                else:
                    async def _noop():
                        return "[Stationary explanation not configured]"
                    tasks.append(_noop())
                tasks.append(generate_explanation_async(ns_llm_input, self.config.EXPLANATION_GENERATION_PROMPT_ID))

                _t_llm = _time.time()
                results = await asyncio.gather(*tasks)
                print(f"  [parallel LLM calls] {_time.time()-_t_llm:.1f}s", flush=True)
                return results

            llm_results = asyncio.run(_run_parallel())

            self._record_ns_explanation(mcts_data, epoch, query, llm_results[1])
            return {
                "tree_structure": tree_str,
                "stationary": llm_results[0],
                "non_stationary": llm_results[1],
            }

        except Exception as e:
            traceback.print_exc()
            error_msg = f"Unable to generate explanation: {str(e)}"
            return {
                "tree_structure": error_msg,
                "stationary": error_msg,
                "non_stationary": error_msg,
            }

    def explain_query_streaming(self, query: str, mcts_data: Dict,
                                epoch: Optional[int] = None):
        """Generator that yields SSE events with true token streaming.

        SSE event types:
            status, tree_structure,
            stationary_delta, stationary_done,
            non_stationary_delta, non_stationary_done,
            error, done
        """
        import json as _json
        import time as _time
        import queue
        import threading
        from core.explanation_generation import generate_explanation_streaming

        try:
            yield _json.dumps({"type": "status", "data": "Analyzing query..."})

            _t0 = _time.time()
            pipe = self._run_query_pipeline(query, mcts_data, epoch=epoch)
            result, epoch, tree_node = pipe['result'], pipe['epoch'], pipe['tree_node']
            mcts_root, env_context, id_map = pipe['mcts_root'], pipe['env_context'], pipe['id_map']
            resolved_targets = pipe.get('resolved_targets', [])
            print(f"  [pipeline] {_time.time()-_t0:.1f}s", flush=True)

            # 1. Tree Structure (instant)
            tree_str = self._build_tree_str(tree_node, epoch)
            yield _json.dumps({"type": "tree_structure", "data": tree_str})

            # 2 & 3. Parallel streaming LLM calls
            yield _json.dumps({"type": "status", "data": "Generating explanations..."})

            stationary_llm_input = build_logiex_prompt(
                query, result, env_context, mcts_data, id_map=id_map, epoch=epoch,
                vehicle_targets=resolved_targets)
            ns_llm_input = self._build_explanation_prompt(
                query, result, env_context, tree_node, id_map=id_map,
                mcts_data=mcts_data, epoch=epoch, vehicle_targets=resolved_targets)

            stationary_prompt_id = getattr(self.config, 'STATIONARY_EXPLANATION_PROMPT_ID', None)
            has_stationary = stationary_prompt_id and stationary_prompt_id != "TODO_REPLACE_WITH_ACTUAL_PROMPT_ID"

            # Shared queue: items are (exp_type, event_kind, data)
            # event_kind is "delta", "done", or "error"
            event_queue = queue.Queue()
            _t_llm = _time.time()

            def _stream_llm(exp_type, llm_input, prompt_id):
                try:
                    for event_kind, data in generate_explanation_streaming(llm_input, prompt_id):
                        event_queue.put((exp_type, event_kind, data))
                    print(f"  [{exp_type}] {_time.time()-_t_llm:.1f}s", flush=True)
                except Exception as e:
                    event_queue.put((exp_type, "error", str(e)))

            threads = []
            if has_stationary:
                t = threading.Thread(target=_stream_llm,
                                     args=('stationary', stationary_llm_input, stationary_prompt_id))
                t.start()
                threads.append(t)
            else:
                event_queue.put(('stationary', 'done', '[Stationary explanation not configured]'))

            t = threading.Thread(target=_stream_llm,
                                 args=('non_stationary', ns_llm_input, self.config.EXPLANATION_GENERATION_PROMPT_ID))
            t.start()
            threads.append(t)

            # Drain events until both streams complete
            done_count = 0
            expected_done = 2  # stationary + non_stationary
            while done_count < expected_done:
                try:
                    exp_type, event_kind, data = event_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                if event_kind == "delta":
                    yield _json.dumps({"type": f"{exp_type}_delta", "data": data})
                elif event_kind == "done":
                    if exp_type == 'non_stationary':
                        self._record_ns_explanation(mcts_data, epoch, query, data)
                    yield _json.dumps({"type": f"{exp_type}_done", "data": data})
                    done_count += 1
                elif event_kind == "error":
                    yield _json.dumps({"type": "error", "data": f"{exp_type}: {data}"})
                    done_count += 1

            for t in threads:
                t.join()

            print(f"  [all LLM calls] {_time.time()-_t_llm:.1f}s", flush=True)
            yield _json.dumps({"type": "done", "data": ""})

        except Exception as e:
            traceback.print_exc()
            error_msg = f"Unable to generate explanation: {str(e)}"
            yield _json.dumps({"type": "error", "data": error_msg})
