"""
LogiEx baseline evaluator for explanation 2 (stationary).

Main entry point: build_logiex_prompt() is called by orchestrator.py
to replace the PCTL-based stationary explanation with LogiEx-style
"formula: value" evidence.

Follows the original LogiEx pipeline:
  transit.py:process_llm_answer() → parse formulas → score → format
  backend_main.py:run() → logic + query + results → LLM
"""
from typing import Dict, List, Optional, Tuple

from .transit_logics import parse
from .scorer import SCORING_MAP, quantitativescore
from .templates import LOGIEX_TEMPLATES


_STATIONARY_DISABLED_PREFIXES = ("pcta(", "pctd(", "vioa(", "Phi2(")


def _is_disabled_stationary_formula(text: str) -> bool:
    """Return True for LogiEx formulas intentionally excluded from stationary prompts."""
    return text.strip().startswith(_STATIONARY_DISABLED_PREFIXES)


def _filter_stationary_prompt_content(logic_str: str, evidence_str: str) -> Tuple[str, str]:
    """Remove disabled formulas from Logic and Logic Checking Results before prompt assembly."""
    logic_parts = [
        part.strip()
        for part in logic_str.split(";")
        if part.strip() and not _is_disabled_stationary_formula(part)
    ]
    evidence_lines = [
        line
        for line in evidence_str.splitlines()
        if line.strip() and not _is_disabled_stationary_formula(line.split(":", 1)[0])
    ]
    return "; ".join(logic_parts), "\n".join(evidence_lines)


def _get_assignment_info(mcts_data: Dict, epoch: int) -> Tuple[int, int]:
    """Extract assigned_vehicle and closest_vehicle for the given epoch."""
    # Try assignment_details first (always available)
    ad_list = mcts_data.get('env_data', {}).get('assignment_details', [])
    for ad in ad_list:
        if ad.get('decision_epoch') == epoch or ad.get('request_id') == epoch:
            return ad.get('assigned_vehicle', 0), ad.get('closest_vehicle', 0)

    # Fall back to mdp_comparisons
    comp = mcts_data.get('scenario_data', {}).get('mdp_comparisons', {}).get(epoch, {})
    mdp_t = comp.get('mdp_t', {})
    ai = mdp_t.get('assignment_info', {})
    if ai:
        return ai.get('assigned_vehicle', 0), ai.get('closest_vehicle', 0)

    # Last resort: use the tree root's best child
    tree = mcts_data.get('per_step_trees', {}).get(f'ep_1|{epoch}')
    if tree:
        children = tree.get('children', [])
        if children:
            best = max(children, key=lambda c: c.get('visits', 0))
            assigned = best.get('action', 0)
            # closest = second most visited (rough heuristic)
            others = [c for c in children if c.get('action') != assigned]
            closest = others[0].get('action', 0) if others else assigned
            return assigned, closest

    return 0, 1


def evaluate_logiex_evidence(
    query_type: int,
    tree_root_dict: dict,
    state: dict,
    assigned_vehicle: int,
    closest_vehicle: int,
    epoch: int,
    request_label: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Evaluate LogiEx formulas on MCTS tree and return evidence.

    Follows original LogiEx transit.py:process_llm_answer() flow:
    1. Get templates for query type, fill placeholders
    2. Parse each formula via transit_logics.parse()
    3. Score via quantitativescore/qualitativescore
    4. Format as "formula: value" lines

    Args:
        query_type: Classification label (1-28 or -1)
        tree_root_dict: per_step_trees[f'ep_1|{epoch}'] dict
        state: state_snapshot['state'] dict
        assigned_vehicle: Vehicle ID chosen by planner
        closest_vehicle: Closest vehicle ID
        epoch: Decision epoch used to fetch the active state/tree
        request_label: User-facing request id to display inside formulas.
            Defaults to epoch when no renumbering is active.

    Returns:
        (logic_str, evidence_str):
          logic_str: "r(4); r(0); N(0,4); ..."
          evidence_str: "r(4): -0.401\nr(0): -4.523\n..."
    """
    # 1. Get templates and fill placeholders
    templates = LOGIEX_TEMPLATES.get(query_type, LOGIEX_TEMPLATES.get(-1, []))
    request_ref = epoch if request_label is None else request_label
    placeholders = {
        'v1': str(assigned_vehicle),
        'v2': str(closest_vehicle),
        'r': str(request_ref),
    }

    filled_formulas = []
    for tmpl in templates:
        try:
            filled = tmpl.format(**placeholders)
            filled_formulas.append(filled)
        except (KeyError, IndexError):
            filled_formulas.append(tmpl)

    # 2. Parse and score each formula
    data = (tree_root_dict, state)
    logic_parts = []
    evidence_lines = []

    for formula_str in filled_formulas:
        try:
            parsed = parse(formula_str)
            score_func = SCORING_MAP.get(type(parsed), quantitativescore)
            result = score_func(parsed, data)
            logic_parts.append(formula_str)
            evidence_lines.append(f"{formula_str}: {result}")
        except Exception as e:
            logic_parts.append(formula_str)
            evidence_lines.append(f"{formula_str}: Error ({e})")

    logic_str = "; ".join(logic_parts)
    evidence_str = "\n".join(evidence_lines)
    return logic_str, evidence_str


def build_logiex_prompt(
    query: str,
    result: Dict,
    env_context: str,
    mcts_data: Dict,
    id_map: Optional[Dict] = None,
    epoch: Optional[int] = None,
    vehicle_targets: Optional[List[int]] = None,
) -> str:
    """
    Build the full LLM input for explanation 2 using LogiEx evidence.

    Called from orchestrator.py in place of _build_explanation_prompt(..., stationary=True).

    Args:
        query: User query string
        result: Dict from process_query() with classification_label, etc.
        env_context: Legacy formatted environment context string; not included in the LLM input.
        mcts_data: Full mcts_data dict
        id_map: Optional renumbering map from real epoch -> user-facing request id.
        epoch: Decision epoch (if None, uses latest_epoch from _runtime)
        vehicle_targets: Ordered list of resolved vehicle IDs from query.
            Used for target-early binding of {v1}/{v2} placeholders
            according to per-template-family rules.

    Returns:
        Formatted prompt string for LLM.
    """
    query_type = result.get('classification_label', -1)

    # Use provided epoch, or fall back to latest
    if epoch is None:
        rt = mcts_data.get('_runtime', {})
        epoch = rt.get('latest_epoch', 0)

    # Get tree root dict for this epoch
    tree_root_dict = mcts_data.get('per_step_trees', {}).get(f'ep_1|{epoch}')
    if tree_root_dict is None:
        return f"Query: {query}\n\nNo tree data for epoch {epoch}."

    # Get state from state_snapshot
    comp = mcts_data.get('scenario_data', {}).get('mdp_comparisons', {}).get(epoch, {})
    state_snapshot = comp.get('state_snapshot', {})
    state = state_snapshot.get('state', {})

    # If no state in comparison, try from tree root
    if not state and tree_root_dict:
        state = tree_root_dict.get('state', {})

    # Get assigned/closest vehicle (default binding)
    assigned_vehicle, closest_vehicle = _get_assignment_info(mcts_data, epoch)

    # Target-early: resolve v1/v2 binding based on vehicle targets
    # and per-template-family semantics
    v1, v2 = assigned_vehicle, closest_vehicle

    if vehicle_targets:
        templates = LOGIEX_TEMPLATES.get(query_type, LOGIEX_TEMPLATES.get(-1, []))
        uses_v1 = any('{v1}' in t for t in templates)
        uses_v2 = any('{v2}' in t for t in templates)

        if uses_v1 and uses_v2:  # Pairwise family
            if len(vehicle_targets) >= 2:
                v1, v2 = vehicle_targets[0], vehicle_targets[1]
            elif vehicle_targets[0] != assigned_vehicle:
                # Single non-assigned target in comparison context
                v1, v2 = assigned_vehicle, vehicle_targets[0]
            # else target IS assigned → keep default (assigned vs closest)
        elif uses_v1 and not uses_v2:  # Single-v1 family
            v1 = vehicle_targets[0]
        elif uses_v2 and not uses_v1:  # Single-v2 family
            v2 = vehicle_targets[0]
        # Global family (no v1/v2): no change

    # Run LogiEx evidence evaluation with resolved bindings
    request_label = id_map.get(epoch, epoch) if id_map else epoch
    logic_str, evidence_str = evaluate_logiex_evidence(
        query_type, tree_root_dict, state,
        v1, v2, epoch, request_label=request_label
    )
    logic_str, evidence_str = _filter_stationary_prompt_content(logic_str, evidence_str)

    # Format like original LogiEx: backend_main.py lines 60-67
    sections = [
        f"Query: {query}\nQuery Type: {query_type}",
    ]
    if logic_str:
        sections.append(f"Logic: {logic_str}")
    if evidence_str:
        sections.append(f"Logic Checking Results:\n{evidence_str}")
    llm_input = "\n\n".join(sections) + "\n"

    return llm_input
