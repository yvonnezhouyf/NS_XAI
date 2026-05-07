"""
LogiEx scoring functions adapted for NS-XAI paratransit MCTS tree data.

Follows the same singledispatch pattern as the original LogiEx
logic_parameterizer.py, but reads from AdaptedMCTSNode / tree dicts
+ state snapshots instead of LogiEx's JSON tree format.

Data argument convention:
    data = (tree_root, state)
    - tree_root: dict from per_step_trees[f'ep_1|{epoch}'] (or AdaptedMCTSNode)
    - state: dict from mdp_comparisons[epoch]['state_snapshot']['state']
             (keys: decision_epoch, traffic_level, current_request, vehicles)
"""
from functools import singledispatch

from .transit_logics import (
    PickUpTime, DropOffTime, TreeVisit, Capacity, Occupancy,
    Reward, DecompReward1, DecompReward2, ETA,
    StopsPickUp, StopsDropOff,
    DegreeVioDelay, DegreeVioAdv, ChanceVioDelay, ChanceVioAdv,
    CapacityVio, CapacityVioQuant,
    CompVioTime, CompPctTime, CompReward, CompNumStops,
    CarAssign, AvailableCar,
    Congestion, Exclude, Reassign, MultiPass, AdditionalSearch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def minutes_to_hhmm(minutes):
    """Convert float minutes to HH:MM string."""
    if minutes is None:
        return "N/A"
    total_min = int(round(minutes))
    h = total_min // 60
    m = total_min % 60
    return f"{h:02}:{m:02}"


def _get_child_for_vehicle(tree_root, vehicle_id):
    """Find the child node whose action matches the given vehicle id."""
    children = tree_root.get('children', []) if isinstance(tree_root, dict) else getattr(tree_root, 'children', [])
    for child in children:
        action = child.get('action') if isinstance(child, dict) else getattr(child, 'action', None)
        if action == int(vehicle_id):
            return child
    return None


def _get_node_attr(node, key, default=None):
    """Get attribute from either dict or object."""
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _get_all_children(tree_root):
    """Get children list from tree root."""
    if isinstance(tree_root, dict):
        return tree_root.get('children', [])
    return getattr(tree_root, 'children', [])


def calculate_true_percentage(values):
    """Calculate the percentage of True values in a list."""
    true_count = sum(1 for element in values if element is True)
    total_elements = len(values)
    if total_elements == 0:
        return 0
    return (true_count / total_elements) * 100


def _enumerate_delay_from_traces(child, prop_name='pickup_delay'):
    """
    Enumerate rollout traces from a child node to check delay status.

    Returns list of booleans indicating whether prop_name was True
    in each rollout trace (at any step).
    """
    traces = _get_node_attr(child, 'rollout_traces', [])
    results = []
    for trace in traces:
        # Each trace is a list of {prop_name: bool} dicts (one per step)
        found = False
        for step in trace:
            if isinstance(step, dict) and step.get(prop_name, False):
                found = True
                break
        results.append(found)
    return results


def _enumerate_delay_degree_from_traces(child, time_prop='pickup_delay',
                                         degree_props=None):
    """
    Estimate average delay degree from rollout traces.

    Since we don't have exact fulfillment times in traces (only boolean APs),
    we estimate delay degree by checking threshold-based APs:
    pickup_delay_ge_5, pickup_delay_ge_15, pickup_delay_ge_30, pickup_delay_ge_60

    Returns average estimated delay in minutes (HH:MM string).
    """
    if degree_props is None:
        if 'pickup' in time_prop:
            degree_props = [
                ('pickup_delay_ge_60', 60),
                ('pickup_delay_ge_30', 30),
                ('pickup_delay_ge_15', 15),
                ('pickup_delay_ge_5', 5),
            ]
        else:
            degree_props = [
                ('dropoff_delay_ge_60', 60),
                ('dropoff_delay_ge_30', 30),
                ('dropoff_delay_ge_15', 15),
                ('dropoff_delay_ge_5', 5),
            ]

    traces = _get_node_attr(child, 'rollout_traces', [])
    total_delay = 0
    delay_count = 0

    for trace in traces:
        # For each trace, find the max delay threshold that is True
        max_delay = 0
        for step in trace:
            if not isinstance(step, dict):
                continue
            for prop, mins in degree_props:
                if step.get(prop, False) and mins > max_delay:
                    max_delay = mins
        if max_delay > 0:
            total_delay += max_delay
            delay_count += 1

    if delay_count == 0:
        return "00:00"

    avg_delay = total_delay / delay_count
    h = int(avg_delay) // 60
    m = int(avg_delay) % 60
    return f"{h:02}:{m:02}"


# ---------------------------------------------------------------------------
# Quantitative scoring (adapted from original logic_parameterizer.py)
# ---------------------------------------------------------------------------

@singledispatch
def quantitativescore(transit_logics, data, scenario_num=0):
    return "Not applicable"


@quantitativescore.register(PickUpTime)
def _(transit_logics, data, scenario_num=0):
    tree_root, state = data
    earliest_pickup = state.get('current_request', {}).get('earliest_pickup')
    return minutes_to_hhmm(earliest_pickup)


@quantitativescore.register(DropOffTime)
def _(transit_logics, data, scenario_num=0):
    tree_root, state = data
    latest_dropoff = state.get('current_request', {}).get('latest_dropoff')
    return minutes_to_hhmm(latest_dropoff)


@quantitativescore.register(TreeVisit)
def _(transit_logics, data, scenario_num=0):
    tree_root, state = data
    vehicle_id = int(transit_logics.vehicle)
    child = _get_child_for_vehicle(tree_root, vehicle_id)
    if child is not None:
        return _get_node_attr(child, 'visits', 0)
    return 0


@quantitativescore.register(Capacity)
def _(transit_logics, data, scenario_num=0):
    tree_root, state = data
    vehicle_id = int(transit_logics.vehicle)
    vehicles = state.get('vehicles', [])
    for v in vehicles:
        if v.get('vehicle_id') == vehicle_id:
            return v.get('capacity', 3)
    return 3


@quantitativescore.register(Occupancy)
def _(transit_logics, data, scenario_num=0):
    tree_root, state = data
    vehicle_id = int(transit_logics.vehicle)
    vehicles = state.get('vehicles', [])
    for v in vehicles:
        if v.get('vehicle_id') == vehicle_id:
            return v.get('current_occupancy', 0)
    return 0


@quantitativescore.register(Reward)
def _(transit_logics, data, scenario_num=0):
    tree_root, state = data
    vehicle_id = int(transit_logics.node)
    child = _get_child_for_vehicle(tree_root, vehicle_id)
    if child is not None:
        return round(_get_node_attr(child, 'q_value', 0.0), 4)
    return "N/A"


@quantitativescore.register(DecompReward1)
def _(transit_logics, data, scenario_num=0):
    tree_root, state = data
    vehicle_id = int(transit_logics.node)
    child = _get_child_for_vehicle(tree_root, vehicle_id)
    if child is not None:
        explain = _get_node_attr(child, 'explain', {})
        decomp = explain.get('decomposed_R') if explain else None
        if decomp and len(decomp) > 0:
            return round(decomp[0], 4)
        # Fall back to q_value
        return round(_get_node_attr(child, 'q_value', 0.0), 4)
    return "N/A"


@quantitativescore.register(DecompReward2)
def _(transit_logics, data, scenario_num=0):
    tree_root, state = data
    vehicle_id = int(transit_logics.node)
    child = _get_child_for_vehicle(tree_root, vehicle_id)
    if child is not None:
        explain = _get_node_attr(child, 'explain', {})
        decomp = explain.get('decomposed_R') if explain else None
        if decomp and len(decomp) > 1:
            return round(decomp[1], 4)
        return round(_get_node_attr(child, 'q_value', 0.0), 4)
    return "N/A"


@quantitativescore.register(ETA)
def _(transit_logics, data, scenario_num=0):
    tree_root, state = data
    vehicle_id = int(transit_logics.request)
    child = _get_child_for_vehicle(tree_root, vehicle_id)
    if child is not None:
        explain = _get_node_attr(child, 'explain', {})
        if explain:
            bnn = explain.get('bnn_predicted_states', {})
            mdp_t = bnn.get('mdp_t', {})
            v_key = f"V{vehicle_id}"
            if v_key in mdp_t:
                predicted_time = mdp_t[v_key].get('predicted_time')
                if predicted_time is not None:
                    return minutes_to_hhmm(predicted_time)
    return "N/A"


@quantitativescore.register(StopsPickUp)
def _(transit_logics, data, scenario_num=0):
    tree_root, state = data
    vehicle_id = int(transit_logics.vehicle)
    vehicles = state.get('vehicles', [])
    for v in vehicles:
        if v.get('vehicle_id') == vehicle_id:
            route = v.get('route', [])
            return len(route)
    return 0


@quantitativescore.register(StopsDropOff)
def _(transit_logics, data, scenario_num=0):
    tree_root, state = data
    vehicle_id = int(transit_logics.vehicle)
    vehicles = state.get('vehicles', [])
    for v in vehicles:
        if v.get('vehicle_id') == vehicle_id:
            route = v.get('route', [])
            # Count stops after the pickup for this request
            # In the route, each entry is (request_id, location)
            # Stops after pickup = total route length (all pending)
            return len(route)
    return 0


@quantitativescore.register(DegreeVioDelay)
def _(transit_logics, data, scenario_num=0):
    tree_root, state = data
    # Determine which vehicle to check from the ETA component
    vehicle_id = int(transit_logics.est_arrival.request)
    child = _get_child_for_vehicle(tree_root, vehicle_id)
    if child is None:
        return "00:00"
    # Determine pickup or dropoff based on the time component
    if isinstance(transit_logics.time, PickUpTime):
        prop = 'pickup_delay'
    else:
        prop = 'dropoff_delay'
    return _enumerate_delay_degree_from_traces(child, prop)


@quantitativescore.register(DegreeVioAdv)
def _(transit_logics, data, scenario_num=0):
    # Advance (early) violation -- not directly available in our traces.
    # Return "00:00" as we don't track early arrivals in atomic props.
    return "00:00"


@quantitativescore.register(ChanceVioDelay)
def _(transit_logics, data, scenario_num=0):
    tree_root, state = data
    vehicle_id = int(transit_logics.est_arrival.request)
    child = _get_child_for_vehicle(tree_root, vehicle_id)
    if child is None:
        return "0%"
    if isinstance(transit_logics.time, PickUpTime):
        prop = 'pickup_delay'
    else:
        prop = 'dropoff_delay'
    delay_results = _enumerate_delay_from_traces(child, prop)
    pct = calculate_true_percentage(delay_results)
    return f"{pct:.1f}%"


@quantitativescore.register(ChanceVioAdv)
def _(transit_logics, data, scenario_num=0):
    # Advance (early) violation chance -- not tracked in our atomic props.
    return "0%"


@quantitativescore.register(CapacityVioQuant)
def _(transit_logics, data, scenario_num=0):
    cap = quantitativescore(transit_logics.cap, data, scenario_num)
    occ = quantitativescore(transit_logics.occ, data, scenario_num)
    try:
        return int(cap) - int(occ)
    except (ValueError, TypeError):
        return "N/A"


@quantitativescore.register(CompVioTime)
def _(transit_logics, data, scenario_num=0):
    left = quantitativescore(transit_logics.left, data, scenario_num)
    right = quantitativescore(transit_logics.right, data, scenario_num)
    return [left, right]


@quantitativescore.register(CompPctTime)
def _(transit_logics, data, scenario_num=0):
    left = quantitativescore(transit_logics.left, data, scenario_num)
    right = quantitativescore(transit_logics.right, data, scenario_num)
    return [left, right]


@quantitativescore.register(CompReward)
def _(transit_logics, data, scenario_num=0):
    left = quantitativescore(transit_logics.left, data, scenario_num)
    right = quantitativescore(transit_logics.right, data, scenario_num)
    try:
        return round(float(left) - float(right), 4)
    except (ValueError, TypeError):
        return "N/A"


@quantitativescore.register(CompNumStops)
def _(transit_logics, data, scenario_num=0):
    left = quantitativescore(transit_logics.left, data, scenario_num)
    right = quantitativescore(transit_logics.right, data, scenario_num)
    try:
        return int(left) - int(right)
    except (ValueError, TypeError):
        return "N/A"


@quantitativescore.register(CarAssign)
def _(transit_logics, data, scenario_num=0):
    tree_root, state = data
    children = _get_all_children(tree_root)
    max_r = -float('inf')
    assigned = -1
    for child in children:
        action = _get_node_attr(child, 'action')
        q = _get_node_attr(child, 'q_value', -float('inf'))
        if action is not None and q > max_r:
            max_r = q
            assigned = action
    return assigned


@quantitativescore.register(AvailableCar)
def _(transit_logics, data, scenario_num=0):
    tree_root, state = data
    children = _get_all_children(tree_root)
    count = 0
    for child in children:
        action = _get_node_attr(child, 'action')
        if action is not None:
            vehicle_id = int(action)
            vehicles = state.get('vehicles', [])
            for v in vehicles:
                if v.get('vehicle_id') == vehicle_id:
                    if v.get('current_occupancy', 0) < v.get('capacity', 3):
                        count += 1
                    break
    return count


# Not applicable in our system (require re-running MCTS)
@quantitativescore.register(Congestion)
def _(transit_logics, data, scenario_num=0):
    return "N/A (not supported in baseline)"

@quantitativescore.register(Exclude)
def _(transit_logics, data, scenario_num=0):
    return "N/A (not supported in baseline)"

@quantitativescore.register(Reassign)
def _(transit_logics, data, scenario_num=0):
    return "N/A (not supported in baseline)"

@quantitativescore.register(MultiPass)
def _(transit_logics, data, scenario_num=0):
    return "N/A (not supported in baseline)"

@quantitativescore.register(AdditionalSearch)
def _(transit_logics, data, scenario_num=0):
    return "N/A (not supported in baseline)"


# ---------------------------------------------------------------------------
# Qualitative scoring
# ---------------------------------------------------------------------------

@singledispatch
def qualitativescore(transit_logics, data, scenario_num=0):
    return "Not applicable"


@qualitativescore.register(CapacityVio)
def _(transit_logics, data, scenario_num=0):
    occ = quantitativescore(transit_logics.occ, data, scenario_num)
    cap = quantitativescore(transit_logics.cap, data, scenario_num)
    try:
        return int(occ) >= int(cap)
    except (ValueError, TypeError):
        return False


@qualitativescore.register(CompReward)
def _(transit_logics, data, scenario_num=0):
    left = quantitativescore(transit_logics.left, data, scenario_num)
    right = quantitativescore(transit_logics.right, data, scenario_num)
    try:
        return float(left) < float(right)
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Scoring dispatcher (same as original transit.py:process_llm_answer)
# ---------------------------------------------------------------------------

SCORING_MAP = {
    PickUpTime: quantitativescore,
    DropOffTime: quantitativescore,
    TreeVisit: quantitativescore,
    Capacity: quantitativescore,
    Congestion: quantitativescore,
    MultiPass: quantitativescore,
    Exclude: quantitativescore,
    Reassign: quantitativescore,
    Occupancy: quantitativescore,
    StopsPickUp: quantitativescore,
    StopsDropOff: quantitativescore,
    Reward: quantitativescore,
    DecompReward1: quantitativescore,
    DecompReward2: quantitativescore,
    ETA: quantitativescore,
    DegreeVioDelay: quantitativescore,
    DegreeVioAdv: quantitativescore,
    ChanceVioDelay: quantitativescore,
    ChanceVioAdv: quantitativescore,
    CapacityVio: qualitativescore,
    CapacityVioQuant: quantitativescore,
    CompVioTime: quantitativescore,
    CompPctTime: quantitativescore,
    CompReward: quantitativescore,
    CompNumStops: quantitativescore,
    AdditionalSearch: quantitativescore,
    CarAssign: quantitativescore,
    AvailableCar: quantitativescore,
}
