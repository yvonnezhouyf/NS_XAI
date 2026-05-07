"""
Simplified PCTL checker - domain-aware via config.check_atomic_proposition()

This module provides two evaluation modes:
1. Tree-based: Evaluate PCTL on MCTS tree nodes (legacy, uses environment transition probabilities)
2. Successor-based: Evaluate PCTL using MCTS-observed successor distributions from evidence cache

IMPORTANT: Successor-based mode (used for planned path explanations) uses probabilities
derived from MCTS visit counts, NOT environment transition probabilities. This ensures
PCTL values reflect what MCTS observed during planning.

Domain-aware atomic proposition checking:
- Frozen Lake: Uses env.is_goal(), env.is_hole(), node attributes
- Paratransit: Uses config.check_atomic_proposition() with _cached_props
"""

import re
import sys
from typing import List, Dict, Any, Optional

# Global config reference - set by orchestrator before PCTL evaluation
_ACTIVE_CONFIG = None

def set_active_config(config):
    """Set the active domain config for atomic proposition checking."""
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = config

# Configuration for non-terminal leaf nodes
NON_TERMINAL_LEAF_ASSUMPTIONS = {
    'F': 0.0,  # For F "goal", a SAFE leaf has not reached the goal
    'G': 1.0,  # For G "not hole", a SAFE leaf has so far satisfied the property  
    'U': 0.0,  # For "prop" U "goal", a SAFE leaf has not reached the goal
    'X': 0.0,  # For X "prop", a SAFE leaf has no next state
}

MAX_ITERATIONS = 5000
CONVERGENCE_THRESHOLD = 1e-6

def _normalize_q_value(q_value: float, min_q: float, q_range: float, inverse: bool = False) -> float:
    """Normalize Q-value to [0,1] probability. If inverse=True, use 1.0 - normalized."""
    if q_range > 0:
        normalized = (q_value - min_q) / q_range
    else:
        normalized = max(0.0, (q_value + 1.0) / 2.0)
    
    result = 1.0 - normalized if inverse else normalized
    return max(0.0, min(1.0, result))

def _get_transition_prob(node, child, env):
    """
    Get transition probability from node to child using MCTS visit-based statistics.
    
    CRITICAL FIX: MCTS tree structure has children that represent (state, action) outcome nodes.
    The same child state can be reached via different actions, so:
    - sum(child.visits) can exceed node.visits (children are counted across all actions)
    - We need to normalize by the action's total visits, not the node's visits
    
    Algorithm:
    1. Group all children by action
    2. Find which action group the target child belongs to
    3. Calculate probability = child.visits / action_total_visits
    
    This ensures probabilities for outcomes of a given action sum to 1.0.
    """
    if not hasattr(node, 'children') or not node.children:
        return 0.0
    
    if not hasattr(child, 'visits') or not hasattr(child, 'action'):
        return 0.0
    
    child_action = child.action
    
    # Group children by action and calculate total visits per action
    action_total_visits = 0
    for sibling in node.children:
        if hasattr(sibling, 'action') and sibling.action == child_action:
            action_total_visits += getattr(sibling, 'visits', 0)
    
    # Calculate probability as child.visits / action_total_visits
    if action_total_visits > 0:
        return child.visits / action_total_visits
    
    return 0.0

def _get_all_nodes_prob(root):
    """Traverse the tree and return a list of all nodes for probability checking."""
    nodes = []
    q = [root]
    visited = {root}
    while q:
        node = q.pop(0)
        nodes.append(node)
        if node.children:
            for child in node.children:
                if child not in visited:
                    q.append(child)
                    visited.add(child)
    return nodes

def _satisfies(node, env, formula_str):
    """
    Check if an atomic proposition is satisfied by the given node in the environment.
    Supports: atomic props, negation (!), conjunction (&), disjunction (|)
    """
    cleaned = formula_str.replace('"', '').strip()
    
    # Handle conjunction: prop1 & prop2
    if ' & ' in cleaned:
        parts = cleaned.split(' & ')
        return all(_satisfies(node, env, part.strip()) for part in parts)
    
    # Handle disjunction: prop1 | prop2
    if ' | ' in cleaned:
        parts = cleaned.split(' | ')
        return any(_satisfies(node, env, part.strip()) for part in parts)
    
    # Remove parentheses for single prop
    cleaned = cleaned.replace('(', '').replace(')', '').strip()
    
    # Handle negation
    if cleaned.startswith('!'):
        prop = cleaned[1:]
        negated = True
    elif cleaned.startswith('not '):
        prop = cleaned[4:]
        negated = True
    else:
        prop = cleaned
        negated = False

    result = False

    # =========================================================================
    # Strategy 1: Use config's check_atomic_proposition if available
    # This handles domain-specific propositions (paratransit, etc.)
    # =========================================================================
    global _ACTIVE_CONFIG
    if _ACTIVE_CONFIG is not None and hasattr(_ACTIVE_CONFIG, 'check_atomic_proposition'):
        result = _ACTIVE_CONFIG.check_atomic_proposition(node, env, prop)
        return not result if negated else result

    # =========================================================================
    # Strategy 2: Frozen Lake fallback (env has is_goal/is_hole methods)
    # =========================================================================
    if prop == 'goal' and hasattr(env, 'is_goal'):
        result = env.is_goal(node.state)
    elif prop == 'hole' and hasattr(env, 'is_hole'):
        result = env.is_hole(node.state)
    elif prop == 'nearH':
        result = getattr(node, 'nearH', False)

    # Path-based atomic propositions (used in queries 3, 9-14)
    elif prop == 'on_direct_path':
        result = getattr(node, 'on_direct_path', False)
    elif prop == 'on_opt_path_pre':
        result = getattr(node, 'on_opt_path_pre', False)
    elif prop == 'on_opt_path_post':
        result = getattr(node, 'on_opt_path_post', False)

    # HuGS environment change atomic propositions (used in queries 11-14)
    elif prop == 'new_hole':
        result = getattr(node, 'new_hole', False)
    elif prop == 'removed_hole':
        result = getattr(node, 'removed_hole', False)

    # Handle unsupported propositions - only show warning once
    else:
        if not hasattr(_satisfies, '_warned_props'):
            _satisfies._warned_props = set()
        if prop not in _satisfies._warned_props:
            print(f"WARNING: Unsupported atomic proposition '{prop}' in formula '{formula_str}'", file=sys.stderr)
            _satisfies._warned_props.add(prop)
        result = False

    return not result if negated else result

def _compute_fixed_point_U(formula1_str, formula2_str, env, all_nodes):
    """
    Calculates P(formula1 U formula2) for all nodes using fixed-point iteration.
    
    Special handling for 'goal' and 'hole' in formula2: Use normalized Q-values for leaf nodes.
    """
    prob_k = {id(n): 0.0 for n in all_nodes}
    
    # Check if formula2 is about reaching goal or hole
    is_goal_until = 'goal' in formula2_str.lower()
    is_hole_until = 'hole' in formula2_str.lower() and not is_goal_until
    
    # Use fixed absolute normalization range [-1, +1] for Q-values
    # This avoids outliers distorting the probability mapping
    if is_goal_until or is_hole_until:
        min_q = -1.0
        q_range = 2.0

    for _ in range(MAX_ITERATIONS):
        prob_k_plus_1 = {}
        max_diff = 0.0
        for node in all_nodes:
            if _satisfies(node, env, formula2_str):
                p_node = 1.0
            elif not _satisfies(node, env, formula1_str):
                p_node = 0.0
            elif not node.children:  # It's a non-terminal leaf that satisfies formula1
                if is_goal_until:
                    # Q-value → goal probability
                    q_value = getattr(node, 'q_value', 0.0)
                    p_node = _normalize_q_value(q_value, min_q, q_range)
                elif is_hole_until:
                    # Q-value → hole probability (inverse)
                    q_value = getattr(node, 'q_value', 0.0)
                    p_node = _normalize_q_value(q_value, min_q, q_range, inverse=True)
                else:
                    p_node = NON_TERMINAL_LEAF_ASSUMPTIONS['U']
            else:
                p_node = 0.0
                for child in node.children:
                    transition_prob = _get_transition_prob(node, child, env)
                    p_node += transition_prob * prob_k.get(id(child), 0.0)
            
            # Clip probability to valid range [0, 1]
            p_node = max(0.0, min(1.0, p_node))
            prob_k_plus_1[id(node)] = p_node
            max_diff = max(max_diff, abs(p_node - prob_k.get(id(node), 0.0)))

        prob_k = prob_k_plus_1
        if max_diff < CONVERGENCE_THRESHOLD:
            break
    
    # Ensure all final probabilities are in valid range
    return {node_id: max(0.0, min(1.0, prob)) for node_id, prob in prob_k.items()}

def _compute_fixed_point_G(formula_str, env, all_nodes):
    """Calculates P(G formula) for all nodes using greatest fixed-point iteration."""
    prob_k = {id(n): 1.0 for n in all_nodes} # Initialize with 1.0

    for _ in range(MAX_ITERATIONS):
        prob_k_plus_1 = {}
        max_diff = 0.0
        for node in all_nodes:
            if not _satisfies(node, env, formula_str):
                p_node = 0.0
            elif not node.children: # It's a leaf
                p_node = NON_TERMINAL_LEAF_ASSUMPTIONS['G']
            else:
                p_node = 0.0
                if node.visits > 0:
                    for child in node.children:
                        transition_prob = _get_transition_prob(node, child, env)
                        p_node += transition_prob * prob_k.get(id(child), 0.0)
            
            # Clip probability to valid range [0, 1]
            p_node = max(0.0, min(1.0, p_node))
            prob_k_plus_1[id(node)] = p_node
            max_diff = max(max_diff, abs(p_node - prob_k.get(id(node), 0.0)))

        prob_k = prob_k_plus_1
        if max_diff < CONVERGENCE_THRESHOLD:
            break
    
    # Ensure all final probabilities are in valid range
    return {node_id: max(0.0, min(1.0, prob)) for node_id, prob in prob_k.items()}

def _compute_fixed_point_F(formula_str, env, all_nodes):
    """
    Calculates P(F formula) for all nodes using least fixed-point iteration.
    
    Special handling for 'goal' and 'hole': Use normalized Q-values for leaf nodes since
    MCTS trees don't include terminal states (episodes terminate at goal/hole).
    In Frozen Lake: reward=+1 for goal, -1 for hole, 0 otherwise.
    Q-values represent expected cumulative reward.
    """
    prob_k = {id(n): 0.0 for n in all_nodes}
    
    # Check if formula is about reaching goal or hole
    is_goal_formula = 'goal' in formula_str.lower()
    is_hole_formula = 'hole' in formula_str.lower() and not is_goal_formula
    
    # Use fixed absolute normalization range [-1, +1] for Q-values
    # This avoids outliers distorting the probability mapping
    if is_goal_formula or is_hole_formula:
        min_q = -1.0
        q_range = 2.0

    for _ in range(MAX_ITERATIONS):
        prob_k_plus_1 = {}
        max_diff = 0.0
        for node in all_nodes:
            if _satisfies(node, env, formula_str):
                p_node = 1.0
            elif not node.children:  # It's a leaf
                if is_goal_formula:
                    # Q-value → probability of reaching goal
                    q_value = getattr(node, 'q_value', 0.0)
                    p_node = _normalize_q_value(q_value, min_q, q_range)
                elif is_hole_formula:
                    # Q-value → probability of reaching hole (inverse)
                    q_value = getattr(node, 'q_value', 0.0)
                    p_node = _normalize_q_value(q_value, min_q, q_range, inverse=True)
                else:
                    # For non-goal/hole formulas: use standard assumption
                    p_node = NON_TERMINAL_LEAF_ASSUMPTIONS['F']
            else:
                p_node = 0.0
                if node.visits > 0:
                    for child in node.children:
                        transition_prob = _get_transition_prob(node, child, env)
                        p_node += transition_prob * prob_k.get(id(child), 0.0)
            
            # Clip probability to valid range [0, 1]
            p_node = max(0.0, min(1.0, p_node))
            prob_k_plus_1[id(node)] = p_node
            max_diff = max(max_diff, abs(p_node - prob_k.get(id(node), 0.0)))

        prob_k = prob_k_plus_1
        if max_diff < CONVERGENCE_THRESHOLD:
            break
    
    # Ensure all final probabilities are in valid range
    return {node_id: max(0.0, min(1.0, prob)) for node_id, prob in prob_k.items()}

def _compute_fixed_point_X(formula_str, env, all_nodes):
    """Calculates P(X formula) for all nodes."""
    prob = {}
    for node in all_nodes:
        if not node.children: # It's a leaf
            prob[id(node)] = NON_TERMINAL_LEAF_ASSUMPTIONS['X']
        else:
            p_node = 0.0
            if node.visits > 0:
                for child in node.children:
                    if _satisfies(child, env, formula_str):
                        transition_prob = _get_transition_prob(node, child, env)
                        p_node += transition_prob * 1.0
            # Clip probability to valid range [0, 1]
            p_node = max(0.0, min(1.0, p_node))
            prob[id(node)] = p_node
    
    # Ensure all final probabilities are in valid range
    return {node_id: max(0.0, min(1.0, p)) for node_id, p in prob.items()}

def _compute_F_bounded(formula_str, env, all_nodes, bound):
    """Calculates P(F<=bound formula) for all nodes."""
    if bound == 0:
        # F<=0 means immediately satisfy the formula
        prob = {}
        for node in all_nodes:
            prob[id(node)] = 1.0 if _satisfies(node, env, formula_str) else 0.0
        return prob
    
    # For F<=k with k > 0, use dynamic programming
    prob_prev = {id(n): 1.0 if _satisfies(n, env, formula_str) else 0.0 for n in all_nodes}
    
    for step in range(1, bound + 1):
        prob_curr = {}
        for node in all_nodes:
            if _satisfies(node, env, formula_str):
                prob_curr[id(node)] = 1.0
            elif not node.children:
                prob_curr[id(node)] = NON_TERMINAL_LEAF_ASSUMPTIONS['F']
            else:
                p_node = 0.0
                if node.visits > 0:
                    for child in node.children:
                        transition_prob = _get_transition_prob(node, child, env)
                        p_node += transition_prob * prob_prev.get(id(child), 0.0)
                # Clip probability to valid range [0, 1]
                p_node = max(0.0, min(1.0, p_node))
                prob_curr[id(node)] = p_node
        prob_prev = prob_curr
    
    # Ensure all final probabilities are in valid range
    return {node_id: max(0.0, min(1.0, prob)) for node_id, prob in prob_prev.items()}

def _compute_reward_F(formula_str, env, all_nodes):
    """
    Calculates expected reward/steps until reaching a state satisfying formula.
    
    Special handling for 'goal' and 'hole': Use normalized Q-values for leaf probability,
    and estimate steps based on success probability.
    """
    prob_k = {id(n): 0.0 for n in all_nodes}
    reward_k = {id(n): 0.0 for n in all_nodes}
    
    # Check if formula is about goal or hole
    is_goal_formula = 'goal' in formula_str.lower()
    is_hole_formula = 'hole' in formula_str.lower() and not is_goal_formula
    
    # Collect Q-values for normalization if needed
    if is_goal_formula or is_hole_formula:
        all_q_values = []
        for node in all_nodes:
            if not node.children and hasattr(node, 'q_value'):
                all_q_values.append(node.q_value)
        
        if all_q_values:
            min_q = min(all_q_values)
            max_q = max(all_q_values)
            q_range = max_q - min_q if max_q > min_q else 1.0
        else:
            min_q, max_q, q_range = -1.0, 1.0, 2.0
    
    for iteration in range(MAX_ITERATIONS):
        prob_k_plus_1 = {}
        reward_k_plus_1 = {}
        max_diff = 0.0
        
        for node in all_nodes:
            if _satisfies(node, env, formula_str):
                p_node = 1.0
                r_node = 0.0
            elif not node.children:  # Leaf node
                if is_goal_formula:
                    # Q-value → goal probability
                    q_value = getattr(node, 'q_value', 0.0)
                    p_node = _normalize_q_value(q_value, min_q, q_range)
                    
                    # Estimate remaining steps from leaf to goal
                    normalized_q = (q_value - min_q) / q_range if q_range > 0 else 0.5
                    r_node = 10.0 - (normalized_q * 5.0)
                elif is_hole_formula:
                    # Q-value → hole probability (inverse)
                    q_value = getattr(node, 'q_value', 0.0)
                    p_node = _normalize_q_value(q_value, min_q, q_range, inverse=True)
                    
                    # Estimate remaining steps to hole
                    normalized_q = (q_value - min_q) / q_range if q_range > 0 else 0.5
                    r_node = 10.0 - ((1.0 - normalized_q) * 5.0)
                else:
                    p_node = NON_TERMINAL_LEAF_ASSUMPTIONS['F']
                    r_node = 0.0
            else:
                p_node = 0.0
                r_node = 0.0
                if node.visits > 0:
                    for child in node.children:
                        transition_prob = _get_transition_prob(node, child, env)
                        child_prob = prob_k.get(id(child), 0.0)
                        child_reward = reward_k.get(id(child), 0.0)
                        
                        p_node += transition_prob * child_prob
                        # Expected steps: weighted by probability of reaching goal through this child
                        if child_prob > 0:
                            r_node += transition_prob * child_prob * (1 + child_reward)
                
                # Normalize by total success probability to get conditional expectation
                if p_node > 0:
                    r_node = r_node / p_node
            
            # Clip probability to valid range [0, 1]
            p_node = max(0.0, min(1.0, p_node))
            prob_k_plus_1[id(node)] = p_node
            reward_k_plus_1[id(node)] = r_node
            max_diff = max(max_diff, abs(r_node - reward_k.get(id(node), 0.0)))
        
        prob_k = prob_k_plus_1
        reward_k = reward_k_plus_1
        if max_diff < CONVERGENCE_THRESHOLD:
            break
    
    return reward_k

def _compute_reward_U(formula1_str, formula2_str, env, all_nodes):
    """
    Calculates expected reward/steps until reaching formula2 while maintaining formula1.
    
    Special handling for 'goal' and 'hole' in formula2: Use normalized Q-values for leaf probability.
    """
    prob_k = {id(n): 0.0 for n in all_nodes}
    reward_k = {id(n): 0.0 for n in all_nodes}
    
    # Check if formula2 is about goal or hole
    is_goal_until = 'goal' in formula2_str.lower()
    is_hole_until = 'hole' in formula2_str.lower() and not is_goal_until
    
    # Collect Q-values for normalization if needed
    if is_goal_until or is_hole_until:
        all_q_values = []
        for node in all_nodes:
            if not node.children and _satisfies(node, env, formula1_str) and hasattr(node, 'q_value'):
                all_q_values.append(node.q_value)
        
        if all_q_values:
            min_q = min(all_q_values)
            max_q = max(all_q_values)
            q_range = max_q - min_q if max_q > min_q else 1.0
        else:
            min_q, max_q, q_range = -1.0, 1.0, 2.0
    
    for iteration in range(MAX_ITERATIONS):
        prob_k_plus_1 = {}
        reward_k_plus_1 = {}
        max_diff = 0.0
        
        for node in all_nodes:
            if _satisfies(node, env, formula2_str):
                p_node = 1.0
                r_node = 0.0
            elif not _satisfies(node, env, formula1_str):
                p_node = 0.0
                r_node = 0.0
            elif not node.children:  # Leaf satisfying formula1
                if is_goal_until:
                    # Q-value → goal probability
                    q_value = getattr(node, 'q_value', 0.0)
                    p_node = _normalize_q_value(q_value, min_q, q_range)
                    
                    # Estimate remaining steps to goal
                    normalized_q = (q_value - min_q) / q_range if q_range > 0 else 0.5
                    r_node = 10.0 - (normalized_q * 5.0)
                elif is_hole_until:
                    # Q-value → hole probability (inverse)
                    q_value = getattr(node, 'q_value', 0.0)
                    p_node = _normalize_q_value(q_value, min_q, q_range, inverse=True)
                    
                    # Estimate remaining steps to hole
                    normalized_q = (q_value - min_q) / q_range if q_range > 0 else 0.5
                    r_node = 10.0 - ((1.0 - normalized_q) * 5.0)
                else:
                    p_node = NON_TERMINAL_LEAF_ASSUMPTIONS['U']
                    r_node = 0.0
            else:
                p_node = 0.0
                r_node = 0.0
                if node.visits > 0:
                    for child in node.children:
                        transition_prob = _get_transition_prob(node, child, env)
                        child_prob = prob_k.get(id(child), 0.0)
                        child_reward = reward_k.get(id(child), 0.0)
                        
                        p_node += transition_prob * child_prob
                        # Expected steps: weighted by probability of reaching goal through this child
                        if child_prob > 0:
                            r_node += transition_prob * child_prob * (1 + child_reward)
                
                # Normalize by total success probability to get conditional expectation
                if p_node > 0:
                    r_node = r_node / p_node
            
            # Clip probability to valid range [0, 1]
            p_node = max(0.0, min(1.0, p_node))
            prob_k_plus_1[id(node)] = p_node
            reward_k_plus_1[id(node)] = r_node
            max_diff = max(max_diff, abs(r_node - reward_k.get(id(node), 0.0)))
        
        prob_k = prob_k_plus_1
        reward_k = reward_k_plus_1
        if max_diff < CONVERGENCE_THRESHOLD:
            break
    
    return reward_k


def check_property(root_node, env, property_str):
    """
    Main function to check a PCTL/CTL property on an MCTS tree.
    """
    if not property_str:
        return None

    # Handle quantitative PCTL queries (P=?, P>=threshold, P<=threshold)
    pctl_match = re.match(r'P(>=|<=|=\?)\s*([0-9]*\.?[0-9]*)\s*\[(.*)\]', property_str.strip())
    
    if pctl_match:
        operator = pctl_match.group(1)
        threshold_str = pctl_match.group(2)
        formula = pctl_match.group(3).strip()
        
        all_nodes = _get_all_nodes_prob(root_node)
        
        # Parse temporal operators
        if formula.startswith('F<='):
            # Bounded Eventually: F<=k prop
            bound_match = re.match(r'F<=(\d+)\s+(.+)', formula)
            if bound_match:
                bound = int(bound_match.group(1))
                prop = bound_match.group(2).strip()
                prob_map = _compute_F_bounded(prop, env, all_nodes, bound)
            else:
                return None
        elif formula.startswith('F '):
            # Eventually: F prop
            prop = formula[2:].strip()
            prob_map = _compute_fixed_point_F(prop, env, all_nodes)
        elif formula.startswith('G '):
            # Globally: G prop
            prop = formula[2:].strip()
            prob_map = _compute_fixed_point_G(prop, env, all_nodes)
        elif formula.startswith('X '):
            # Next: X prop
            prop = formula[2:].strip()
            prob_map = _compute_fixed_point_X(prop, env, all_nodes)
        elif ' U ' in formula:
            # Until: prop1 U prop2
            parts = formula.split(' U ', 1)
            if len(parts) == 2:
                prop1 = parts[0].strip()
                prop2 = parts[1].strip()
                prob_map = _compute_fixed_point_U(prop1, prop2, env, all_nodes)
            else:
                return None
        else:
            # Simple proposition
            prob_map = {id(n): 1.0 if _satisfies(n, env, formula) else 0.0 for n in all_nodes}
        
        probability = prob_map.get(id(root_node), 0.0)
        
        if operator == '=?':
            return probability
        elif operator == '>=':
            threshold = float(threshold_str) if threshold_str else 0.0
            return probability >= threshold
        elif operator == '<=':
            threshold = float(threshold_str) if threshold_str else 1.0
            return probability <= threshold
    
    # Handle R (reward/cost) queries
    reward_match = re.match(r'R(\{[^}]*\})?(>=|<=|=\?)\s*([0-9]*\.?[0-9]*)\s*\[(.*)\]', property_str.strip())
    if reward_match:
        reward_type_str = reward_match.group(1)
        operator = reward_match.group(2)
        threshold_str = reward_match.group(3)
        formula = reward_match.group(4).strip()
        
        all_nodes = _get_all_nodes_prob(root_node)
        
        # Parse temporal operators
        if formula.startswith('F '):
            prop = formula[2:].strip()
            reward_map = _compute_reward_F(prop, env, all_nodes)
        elif ' U ' in formula:
            parts = formula.split(' U ', 1)
            if len(parts) == 2:
                prop1 = parts[0].strip()
                prop2 = parts[1].strip()
                reward_map = _compute_reward_U(prop1, prop2, env, all_nodes)
            else:
                return "UNSUPPORTED_REWARD_QUERY"
        else:
            return "UNSUPPORTED_REWARD_QUERY"
        
        expected_reward = reward_map.get(id(root_node), 0.0)
        
        if operator == '=?':
            return expected_reward
        elif operator == '>=':
            threshold = float(threshold_str) if threshold_str else 0.0
            return expected_reward >= threshold
        elif operator == '<=':
            threshold = float(threshold_str) if threshold_str else float('inf')
            return expected_reward <= threshold
    
    return None
