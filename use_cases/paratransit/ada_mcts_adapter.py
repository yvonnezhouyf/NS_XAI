"""
Adapter module to bridge ADA-MCTS (Python 3.8) with NS-XAI explanation system (Python 3.9).
Uses conda environment isolation for version compatibility.

Paratransit-specific implementation.
"""

import os
import subprocess
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configuration for conda environments
ADAMCTS_ENV = "adamcts38"
XAI_ENV = "xai39"


class ParatransitEnvAdapter:
    """Environment adapter for paratransit PCTL checking."""
    
    def __init__(self, env_data: Dict):
        self.n_vehicles = env_data.get('n_vehicles', 5)
        self.n_requests = env_data.get('n_requests', 10)
        self.total_reward = env_data.get('total_reward', 0.0)
        self.violations = env_data.get('violations', [])
        self.assignment_details = env_data.get('assignment_details', [])
        
        # Store raw env_data for access to all fields
        self._env_data = env_data
    
    def get_assignment_for_request(self, request_id: int) -> Optional[Dict]:
        """Get assignment details for a specific request."""
        for detail in self.assignment_details:
            if detail.get('request_id') == request_id:
                return detail
        return None
    
    def get_vehicle_assignments(self, vehicle_id: int) -> List[Dict]:
        """Get all assignments for a specific vehicle."""
        return [d for d in self.assignment_details 
                if d.get('assigned_vehicle') == vehicle_id]


def check_conda_environment(env_name: str) -> bool:
    """Check if the specified conda environment exists."""
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return env_name in result.stdout
        return False
        
    except Exception:
        return False


# ============================================================================
# Tree Building and Loading Functions (for PCTL evaluation)
# ============================================================================

class AdaptedMCTSNode:
    """
    Adapted MCTS node for paratransit PCTL evaluation.

    Wraps serialized dict data into a node object with consistent interface
    for the PCTL checker.
    """

    def __init__(self, data: Dict):
        """
        Initialize node from serialized dict data.

        Args:
            data: Dict with keys like 'visits', 'value', 'q_value', 'type',
                  'action', 'state', 'children', 'atomic_props', 'explain', etc.
        """
        self.visits = data.get('visits', 0)
        self.value = data.get('value', 0.0)
        self.v = data.get('value', 0.0)  # Alias for compatibility
        self.q_value = data.get('q_value', 0.0)
        self.q = data.get('q_value', 0.0)  # Alias for compatibility
        self.type = data.get('type', 'unknown')
        self.action = data.get('action')
        self.state = data.get('state')
        self.explain = data.get('explain', {})

        # Cached atomic propositions
        self._cached_props = data.get('atomic_props', {})

        # Trace-based PCTL evaluation: rollout traces
        # Each trace is a list of {ap_name: bool} dicts (one per step in rollout)
        #
        # ONLY direct traces are stored (prefix-correct, valid for ALL PCTL operators)
        # Subtree traces are no longer used (semantically incorrect, O(N²) bloat)
        self.rollout_traces = data.get('rollout_traces', [])

        # ALIAS: traces = rollout_traces for compatibility with formula_evaluator
        # eval_property_formula() and trace-based evaluation functions read node.traces
        self.traces = self.rollout_traces

        # Transition AP from parent to this node (for trace prefix reconstruction)
        self.transition_ap = data.get('transition_ap', None)

        # Build children recursively
        self.children = []
        for child_data in data.get('children', []):
            if isinstance(child_data, dict):
                self.children.append(AdaptedMCTSNode(child_data))
            elif hasattr(child_data, 'visits'):  # Already a node object
                self.children.append(child_data)

    def get_atomic_prop(self, prop_name: str) -> bool:
        """Get atomic proposition value from cached props."""
        return self._cached_props.get(prop_name, False)

    def __repr__(self):
        return f"AdaptedMCTSNode(state={self.state}, visits={self.visits}, action={self.action})"


def build_tree_from_data(tree_data: Dict) -> AdaptedMCTSNode:
    """
    Build MCTS tree node from serialized data.

    Args:
        tree_data: Serialized tree data (dict format from serialize_node_for_json)

    Returns:
        AdaptedMCTSNode object for paratransit PCTL evaluation
    """
    if tree_data is None:
        return None
    if isinstance(tree_data, AdaptedMCTSNode):
        return tree_data  # Already converted
    if not isinstance(tree_data, dict):
        return None

    return AdaptedMCTSNode(tree_data)


def load_ada_mcts_tree(mcts_data: Dict) -> Tuple[Optional[AdaptedMCTSNode], ParatransitEnvAdapter]:
    """
    Load ADA-MCTS tree for Paratransit domain.

    Creates an AdaptedMCTSNode from the serialized tree data and
    a ParatransitEnvAdapter for PCTL evaluation.

    Args:
        mcts_data: Dictionary containing MCTS execution data with:
                   - 'per_step_trees': Dict of serialized trees per epoch
                   - 'env_data': Environment configuration data

    Returns:
        Tuple of (mcts_root, env_adapter)
        - mcts_root: AdaptedMCTSNode for the latest epoch, or None
        - env_adapter: ParatransitEnvAdapter for atomic proposition checking
    """
    env_data = mcts_data.get('env_data', {})
    env_adapter = ParatransitEnvAdapter(env_data)

    # Get the latest tree from per_step_trees
    per_step_trees = mcts_data.get('per_step_trees', {})

    if not per_step_trees:
        return None, env_adapter

    # Find the latest epoch
    epochs = []
    for key in per_step_trees.keys():
        try:
            # Keys are like "ep_1|0", "ep_1|1", etc.
            epoch = int(key.split('|')[1])
            epochs.append(epoch)
        except (ValueError, IndexError):
            pass

    if not epochs:
        return None, env_adapter

    latest_epoch = max(epochs)
    latest_key = f"ep_1|{latest_epoch}"

    tree_data = per_step_trees.get(latest_key)
    if tree_data is None:
        return None, env_adapter

    # Build tree from data
    mcts_root = build_tree_from_data(tree_data)

    return mcts_root, env_adapter


# Test function
if __name__ == "__main__":
    print("Testing ADA-MCTS conda adapter for Paratransit...")
    
    try:
        # Check environments
        print(f"Checking conda environment '{ADAMCTS_ENV}'...")
        if not check_conda_environment(ADAMCTS_ENV):
            print(f"❌ Environment '{ADAMCTS_ENV}' not found")
        else:
            print(f"✅ Environment '{ADAMCTS_ENV}' found")
        
        print(f"Checking conda environment '{XAI_ENV}'...")
        if not check_conda_environment(XAI_ENV):
            print(f"❌ Environment '{XAI_ENV}' not found")
        else:
            print(f"✅ Environment '{XAI_ENV}' found")
        
        print("✅ Environment check completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
