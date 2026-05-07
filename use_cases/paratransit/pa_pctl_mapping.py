# PCTL Templates mapping to classification labels (Paratransit)
# Categories follow Terminology and Query Categories.docx:
# - reward-maximizing model (pre-change / post-adaptation)
# - risk-averse adaptive model
# - model comparisons across phases
# - HuGS counterfactuals
# - operational / background knowledge

# ============================================================================
# Atomic propositions (defined in nsparatransit_v0.py evaluate_atomic_props):
# ============================================================================
# AP computation is SINGLE SOURCE OF TRUTH in nsparatransit_v0.py.
# adamcts_runner.py delegates to env.evaluate_atomic_props() for consistency.
#
# Core completion/violation:
#   service_complete: True iff all requests are fulfilled without any constraint violation
#   capacity_violation: True iff any vehicle exceeds capacity
#
# Timing performance:
#   pickup_delay: True iff pickup time > earliest_pickup
#   dropoff_delay: True iff dropoff time > latest_dropoff
#   pickup_delay_ge_5/15/30/60: True iff pickup delay >= threshold minutes
#   dropoff_delay_ge_5/15/30/60: True iff dropoff delay >= threshold minutes
#
# Vehicle state:
#   carpool_active: True iff any vehicle carries 2+ passengers
#   capacity_full: True iff any vehicle is exactly at capacity
#   any_vehicle_idle: True iff any vehicle has no pending route
#   all_vehicles_busy: True iff all vehicles have pending routes
#
# Assignment path (risk-like, lower P=? [X ...] is generally better):
#   vehicle_busy_before_assignment: action vehicle has non-empty route
#   clear_current_route_ge_15/30: clearing existing route takes >= 15/30 min
#   deadhead_to_pickup_ge_15/30: deadhead from route end to pickup >= 15/30 min
#   dropoff_slack_le_15/30: dropoff slack (deadline - eta) <= 15/30 min
#   service_path_event_affected: service path leg touches event nodes
#   service_path_bridge_affected: service path has S→N bridge crossing
#
# ============================================================================
# Derived metrics:
# ============================================================================
# DERIVED: TURNING_INTERVAL, STEP_CONFIDENCE
#
# --- Hard-coded constants (computed in formula_evaluator.py) ---
# DERIVED: N_MIN = 3                      # Minimum exploration steps before adaptation
# DERIVED: N_INTERVAL = 2                 # Online update frequency constant
# DERIVED: ASSIGNED_CAPACITY = 3          # Per-vehicle capacity (NSParatransitV0 default)
# DERIVED: EVENT_MULTIPLIER = 3.0         # Congestion multiplier for event scenarios
# DERIVED: ACCIDENT_MULTIPLIER = 2.0      # Travel time multiplier for accident scenarios
# DERIVED: EVENT_EPOCH = 10               # Epoch at which event (congestion) occurs
# DERIVED: ACCIDENT_EPOCH = 10            # Epoch at which accident occurs
#
# --- Generic per-vehicle metrics (computed for all vehicles) ---
#
# DERIVED: PENDING_REQUESTS_BY_VEHICLE = pending request count per vehicle {V0: int, ...}
# DERIVED: TRAFFIC_LEVEL = real traffic level at the decision epoch (global, not per-vehicle)
#
# DERIVED: ETA_PICKUP_BY_VEHICLE = BNN-predicted pickup time per vehicle {V0: float, ...}
# DERIVED: ETA_DROPOFF_BY_VEHICLE = BNN-predicted dropoff time per vehicle {V0: float, ...}
# DERIVED: VEHICLE_OCCUPANCY = BNN-predicted occupancy per vehicle {V0: int, ...}
#
# DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE = minutes to clear each vehicle's route {V0: float, ...}
# DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE = minutes deadhead to pickup per vehicle {V0: float, ...}
# DERIVED: DROPOFF_SLACK_BY_VEHICLE = deadline - eta_dropoff per vehicle {V0: float, ...}
# DERIVED: SERVICE_PATH_DISRUPTION_BY_VEHICLE = disruption exposure per vehicle {V0: str, ...}

PCTL_CTL_TEMPLATES = {
    1: [  # Two-vehicle comparison: why one was chosen over the other, why the
          # closest wasn't picked, or how two vehicles compare on delay risk/reward.
          # Covers any query that names or implies two vehicles and asks for a
          # rationale or comparison. (Former types 6 and 9 merged in — same
          # PCTL+metric set serves all three.)
        "P=? [F service_complete]",
        "P=? [F<=1 service_complete]",
        "P=? [F<=2 pickup_delay]",
        "P=? [F<=2 dropoff_delay]",
        "P=? [F pickup_delay]",
        "P=? [F dropoff_delay]",
        "P=? [F capacity_violation]",
        "P=? [X pickup_delay_ge_30]",
        "P=? [X pickup_delay_ge_60]",
        "P=? [X dropoff_delay_ge_30]",
        "P=? [X dropoff_delay_ge_60]",
        "P=? [F<=2 pickup_delay_ge_30]",
        "P=? [F<=2 pickup_delay_ge_60]",
        "P=? [F<=2 dropoff_delay_ge_30]",
        "P=? [F<=2 dropoff_delay_ge_60]",
        "P=? [X vehicle_busy_before_assignment]",
        "P=? [X clear_current_route_ge_15]",
        "P=? [X clear_current_route_ge_30]",
        "P=? [X deadhead_to_pickup_ge_15]",
        "P=? [X deadhead_to_pickup_ge_30]",
        "P=? [X dropoff_slack_le_15]",
        "P=? [X dropoff_slack_le_30]",
        "P=? [X service_path_event_affected]",
        "P=? [X service_path_bridge_affected]",
        "DERIVED: PENDING_REQUESTS_BY_VEHICLE",
        "DERIVED: ETA_PICKUP_BY_VEHICLE",
        "DERIVED: ETA_DROPOFF_BY_VEHICLE",
        "DERIVED: TRAFFIC_LEVEL",
        "DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE",
        "DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE",
        "DERIVED: DROPOFF_SLACK_BY_VEHICLE",
        "DERIVED: SERVICE_PATH_DISRUPTION_BY_VEHICLE",
    ],

    2: [  # At epoch {}, which vehicle offers the most reliable service?
        "P=? [G (!pickup_delay & !dropoff_delay)]",
        "P=? [F pickup_delay]",
        "P=? [F dropoff_delay]",
        "P=? [F service_complete]",
        "P=? [X pickup_delay_ge_30]",
        "P=? [X pickup_delay_ge_60]",
        "P=? [X dropoff_delay_ge_30]",
        "P=? [X dropoff_delay_ge_60]",
        "P=? [F<=2 pickup_delay_ge_30]",
        "P=? [F<=2 pickup_delay_ge_60]",
        "P=? [F<=2 dropoff_delay_ge_30]",
        "P=? [F<=2 dropoff_delay_ge_60]",
        "P=? [X vehicle_busy_before_assignment]",
        "P=? [X clear_current_route_ge_15]",
        "P=? [X clear_current_route_ge_30]",
        "P=? [X deadhead_to_pickup_ge_15]",
        "P=? [X deadhead_to_pickup_ge_30]",
        "P=? [X dropoff_slack_le_15]",
        "P=? [X dropoff_slack_le_30]",
        "P=? [X service_path_event_affected]",
        "P=? [X service_path_bridge_affected]",
        "DERIVED: PENDING_REQUESTS_BY_VEHICLE",
        "DERIVED: TRAFFIC_LEVEL",
        "DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE",
        "DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE",
        "DERIVED: DROPOFF_SLACK_BY_VEHICLE",
        "DERIVED: SERVICE_PATH_DISRUPTION_BY_VEHICLE",
    ],

    3: [  # At epoch {}, why avoid/start carpooling for request {}?
        "P=? [F carpool_active]",
        "P=? [F capacity_full]",
        "P=? [F capacity_violation]",
        "P=? [F service_complete]",
        "P=? [F<=1 service_complete]"
    ],

    4: [  # At what epoch does the dispatcher stop being overly pessimistic?
        "DERIVED: TURNING_INTERVAL"
    ],

    5: [  # At epoch {}, how confident is the dispatcher about assigning Vehicle {}?
        "DERIVED: STEP_CONFIDENCE",
        "P=? [F service_complete]",
        "P=? [F pickup_delay]",
        "P=? [F dropoff_delay]",
        "P=? [X vehicle_busy_before_assignment]",
        "P=? [X dropoff_slack_le_15]",
        "P=? [X dropoff_slack_le_30]",
        "P=? [X service_path_event_affected]",
        "P=? [X service_path_bridge_affected]",
        "DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE",
        "DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE",
        "DERIVED: DROPOFF_SLACK_BY_VEHICLE",
        "DERIVED: SERVICE_PATH_DISRUPTION_BY_VEHICLE",
    ],

    # Type 6 (why is the closest vehicle not assigned) merged into type 1 —
    # same two-vehicle comparison intent and same PCTL+metric set.
    # Also absorbed former type 11 (why is a farther vehicle chosen).
    # 6: [
    #     "P=? [F service_complete]",
    #     "P=? [F<=1 service_complete]",
    #     "P=? [F<=2 pickup_delay]",
    #     "P=? [F<=2 dropoff_delay]",
    #     "P=? [F capacity_violation]",
    #     "P=? [F pickup_delay]",
    #     "P=? [F dropoff_delay]",
    #     "P=? [X pickup_delay_ge_30]",
    #     "P=? [X pickup_delay_ge_60]",
    #     "P=? [X dropoff_delay_ge_30]",
    #     "P=? [X dropoff_delay_ge_60]",
    #     "P=? [F<=2 pickup_delay_ge_30]",
    #     "P=? [F<=2 pickup_delay_ge_60]",
    #     "P=? [F<=2 dropoff_delay_ge_30]",
    #     "P=? [F<=2 dropoff_delay_ge_60]",
    #     "P=? [X vehicle_busy_before_assignment]",
    #     "P=? [X clear_current_route_ge_15]",
    #     "P=? [X clear_current_route_ge_30]",
    #     "P=? [X deadhead_to_pickup_ge_15]",
    #     "P=? [X deadhead_to_pickup_ge_30]",
    #     "P=? [X dropoff_slack_le_15]",
    #     "P=? [X dropoff_slack_le_30]",
    #     "P=? [X service_path_event_affected]",
    #     "P=? [X service_path_bridge_affected]",
    #     "DERIVED: ETA_PICKUP_BY_VEHICLE",
    #     "DERIVED: ETA_DROPOFF_BY_VEHICLE",
    #     "DERIVED: PENDING_REQUESTS_BY_VEHICLE",
    #     "DERIVED: TRAFFIC_LEVEL",
    #     "DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE",
    #     "DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE",
    #     "DERIVED: DROPOFF_SLACK_BY_VEHICLE",
    #     "DERIVED: SERVICE_PATH_DISRUPTION_BY_VEHICLE",
    # ],

    7: [  # What is the scheduled pickup/dropoff time for request {}?
        "DERIVED: ETA_PICKUP_BY_VEHICLE",
        "DERIVED: ETA_DROPOFF_BY_VEHICLE",
        "DERIVED: TRAFFIC_LEVEL",
        "DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE",
        "DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE",
        "DERIVED: DROPOFF_SLACK_BY_VEHICLE",
    ],

    8: [  # What is the passenger count/capacity pressure for Vehicle {} at epoch {}?
        "P=? [F carpool_active]",
        "P=? [F capacity_full]",
        "P=? [F capacity_violation]",
        "DERIVED: VEHICLE_OCCUPANCY"
    ],

    # Type 9 (how do two vehicles compare on delay risk/reward) merged into
    # type 1 — same two-vehicle comparison intent, metric set is a subset of 1.
    # 9: [
    #     "P=? [F service_complete]",
    #     "P=? [F<=2 pickup_delay]",
    #     "P=? [F<=2 dropoff_delay]",
    #     "P=? [F capacity_violation]",
    #     "P=? [F pickup_delay]",
    #     "P=? [F dropoff_delay]",
    #     "P=? [X pickup_delay_ge_30]",
    #     "P=? [X pickup_delay_ge_60]",
    #     "P=? [X dropoff_delay_ge_30]",
    #     "P=? [X dropoff_delay_ge_60]",
    #     "P=? [F<=2 pickup_delay_ge_30]",
    #     "P=? [F<=2 pickup_delay_ge_60]",
    #     "P=? [F<=2 dropoff_delay_ge_30]",
    #     "P=? [F<=2 dropoff_delay_ge_60]",
    #     "P=? [X vehicle_busy_before_assignment]",
    #     "P=? [X clear_current_route_ge_15]",
    #     "P=? [X clear_current_route_ge_30]",
    #     "P=? [X deadhead_to_pickup_ge_15]",
    #     "P=? [X deadhead_to_pickup_ge_30]",
    #     "P=? [X dropoff_slack_le_15]",
    #     "P=? [X dropoff_slack_le_30]",
    #     "P=? [X service_path_event_affected]",
    #     "P=? [X service_path_bridge_affected]",
    #     "DERIVED: PENDING_REQUESTS_BY_VEHICLE",
    #     "DERIVED: ETA_PICKUP_BY_VEHICLE",
    #     "DERIVED: ETA_DROPOFF_BY_VEHICLE",
    #     "DERIVED: TRAFFIC_LEVEL",
    #     "DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE",
    #     "DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE",
    #     "DERIVED: DROPOFF_SLACK_BY_VEHICLE",
    #     "DERIVED: SERVICE_PATH_DISRUPTION_BY_VEHICLE",
    # ],

    10: [  # At request {}, how many vehicles are available right now?
        "DERIVED: VEHICLE_OCCUPANCY"
    ],
    
    # Type 11 (why is a farther vehicle chosen) merged into type 6 — same counterfactual
    # reasoning as "why isn't the closest vehicle assigned", just phrased from the other side.
    # 11: [
    #     "P=? [X pickup_delay]",
    #     "P=? [X dropoff_delay]",
    #     "P=? [F<=2 pickup_delay]",
    #     "P=? [F<=2 dropoff_delay]",
    #     "P=? [F service_complete]",
    #     "P=? [X pickup_delay_ge_30]",
    #     "P=? [X pickup_delay_ge_60]",
    #     "P=? [X dropoff_delay_ge_30]",
    #     "P=? [X dropoff_delay_ge_60]",
    #     "P=? [F<=2 pickup_delay_ge_30]",
    #     "P=? [F<=2 pickup_delay_ge_60]",
    #     "P=? [F<=2 dropoff_delay_ge_30]",
    #     "P=? [F<=2 dropoff_delay_ge_60]",
    #     "P=? [X vehicle_busy_before_assignment]",
    #     "P=? [X clear_current_route_ge_15]",
    #     "P=? [X clear_current_route_ge_30]",
    #     "P=? [X deadhead_to_pickup_ge_15]",
    #     "P=? [X deadhead_to_pickup_ge_30]",
    #     "P=? [X dropoff_slack_le_15]",
    #     "P=? [X dropoff_slack_le_30]",
    #     "P=? [X service_path_event_affected]",
    #     "P=? [X service_path_bridge_affected]",
    #     "DERIVED: ETA_PICKUP_BY_VEHICLE",
    #     "DERIVED: ETA_DROPOFF_BY_VEHICLE",
    #     "DERIVED: PENDING_REQUESTS_BY_VEHICLE",
    #     "DERIVED: TRAFFIC_LEVEL",
    #     "DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE",
    #     "DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE",
    #     "DERIVED: DROPOFF_SLACK_BY_VEHICLE",
    #     "DERIVED: SERVICE_PATH_DISRUPTION_BY_VEHICLE",
    # ],

    12: [  # Case 2 (event-based non-stationarity): how did the event affect assignment at epoch {}?
        "P=? [X pickup_delay]",
        "P=? [X dropoff_delay]",
        "P=? [F service_complete]",
        "P=? [X pickup_delay_ge_30]",
        "P=? [X pickup_delay_ge_60]",
        "P=? [X dropoff_delay_ge_30]",
        "P=? [X dropoff_delay_ge_60]",
        "P=? [F<=2 pickup_delay_ge_30]",
        "P=? [F<=2 pickup_delay_ge_60]",
        "P=? [F<=2 dropoff_delay_ge_30]",
        "P=? [F<=2 dropoff_delay_ge_60]",
        "P=? [X vehicle_busy_before_assignment]",
        "P=? [X clear_current_route_ge_15]",
        "P=? [X clear_current_route_ge_30]",
        "P=? [X deadhead_to_pickup_ge_15]",
        "P=? [X deadhead_to_pickup_ge_30]",
        "P=? [X dropoff_slack_le_15]",
        "P=? [X dropoff_slack_le_30]",
        "P=? [X service_path_event_affected]",
        "P=? [X service_path_bridge_affected]",
        "DERIVED: EVENT_MULTIPLIER",
        "DERIVED: ETA_PICKUP_BY_VEHICLE",
        "DERIVED: ETA_DROPOFF_BY_VEHICLE",
        "DERIVED: TRAFFIC_LEVEL",
        "DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE",
        "DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE",
        "DERIVED: DROPOFF_SLACK_BY_VEHICLE",
        "DERIVED: SERVICE_PATH_DISRUPTION_BY_VEHICLE",
    ],

    # Type 14 (how do traffic and route burden affect assigned vs closest service timing)
    # merged into type 26 — 14 was a superset framing of 26's ETA-gap question;
    # 26's metric list has been expanded to cover both.
    # 14: [
    #     "DERIVED: ETA_PICKUP_BY_VEHICLE",
    #     "DERIVED: ETA_DROPOFF_BY_VEHICLE",
    #     "DERIVED: PENDING_REQUESTS_BY_VEHICLE",
    #     "DERIVED: TRAFFIC_LEVEL",
    #     "DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE",
    #     "DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE",
    #     "DERIVED: DROPOFF_SLACK_BY_VEHICLE",
    #     "DERIVED: SERVICE_PATH_DISRUPTION_BY_VEHICLE",
    # ],

    15: [  # At event epoch {}, how much does congestion affect assigned timing?
        "P=? [F pickup_delay]",
        "P=? [F dropoff_delay]",
        "P=? [X dropoff_slack_le_15]",
        "P=? [X dropoff_slack_le_30]",
        "P=? [X service_path_event_affected]",
        "P=? [X service_path_bridge_affected]",
        "DERIVED: EVENT_MULTIPLIER",
        "DERIVED: ETA_PICKUP_BY_VEHICLE",
        "DERIVED: ETA_DROPOFF_BY_VEHICLE",
        "DERIVED: TRAFFIC_LEVEL",
        "DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE",
        "DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE",
        "DERIVED: DROPOFF_SLACK_BY_VEHICLE",
        "DERIVED: SERVICE_PATH_DISRUPTION_BY_VEHICLE",
    ],

    16: [  # At request {}, what is the current traffic level?
        "DERIVED: TRAFFIC_LEVEL"
    ],

    17: [  # At request {}, how many pending requests are on the assigned vehicle?
        "DERIVED: PENDING_REQUESTS_BY_VEHICLE"
    ],

    18: [  # At request {}, how many pending requests are on the closest vehicle?
        "DERIVED: PENDING_REQUESTS_BY_VEHICLE"
    ],

    19: [  # At request {}, what is the assigned vehicle ETA to pickup?
        "DERIVED: ETA_PICKUP_BY_VEHICLE"
    ],

    20: [  # At request {}, what is the assigned vehicle ETA to dropoff?
        "DERIVED: ETA_DROPOFF_BY_VEHICLE"
    ],

    21: [  # At request {}, what is the closest vehicle ETA to pickup?
        "DERIVED: ETA_PICKUP_BY_VEHICLE"
    ],

    22: [  # At request {}, what is the closest vehicle ETA to dropoff?
        "DERIVED: ETA_DROPOFF_BY_VEHICLE"
    ],

    23: [  # How often does the model update its belief?
        "DERIVED: N_MIN",
        "DERIVED: N_INTERVAL"
    ],

    24: [  # How does the event affect the traffic? 
        "DERIVED: EVENT_MULTIPLIER"
    ],

    25: [  # At request {}, how do the assigned and closest vehicles differ in workload and route burden?
        "DERIVED: PENDING_REQUESTS_BY_VEHICLE",
        "DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE",
        "DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE",
    ],

    26: [  # At request {}, how much later is the assigned vehicle's pickup/dropoff ETA
           # than the closest vehicle's? (Also covers the broader "how do traffic and
           # route burden shape assigned vs closest timing" flavor — formerly type 14.)
        "DERIVED: ETA_PICKUP_BY_VEHICLE",
        "DERIVED: ETA_DROPOFF_BY_VEHICLE",
        "DERIVED: PENDING_REQUESTS_BY_VEHICLE",
        "DERIVED: TRAFFIC_LEVEL",
        "DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE",
        "DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE",
        "DERIVED: DROPOFF_SLACK_BY_VEHICLE",
        "DERIVED: SERVICE_PATH_DISRUPTION_BY_VEHICLE",
    ],

    27: [  # At request {}, what is the assigned vehicle's expected ride time (pickup to dropoff)?
        "DERIVED: ETA_PICKUP_BY_VEHICLE",
        "DERIVED: ETA_DROPOFF_BY_VEHICLE",
        "DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE",
        "DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE",
        "DERIVED: DROPOFF_SLACK_BY_VEHICLE",
    ],

    28: [  # At request {}, what is the closest vehicle's expected ride time (pickup to dropoff)?
        "DERIVED: ETA_PICKUP_BY_VEHICLE",
        "DERIVED: ETA_DROPOFF_BY_VEHICLE",
        "DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE",
        "DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE",
        "DERIVED: DROPOFF_SLACK_BY_VEHICLE",
    ],

    -1: [
        "P=? [F service_complete]",
        "P=? [F capacity_violation]",
        "P=? [F pickup_delay]",
        "P=? [F dropoff_delay]",
        "P=? [F carpool_active]",
        "P=? [F capacity_full]",
        "P=? [F any_vehicle_idle]",
        "P=? [G all_vehicles_busy]",
        "P=? [X pickup_delay_ge_30]",
        "P=? [X pickup_delay_ge_60]",
        "P=? [X dropoff_delay_ge_30]",
        "P=? [X dropoff_delay_ge_60]",
        "P=? [F<=2 pickup_delay_ge_30]",
        "P=? [F<=2 pickup_delay_ge_60]",
        "P=? [F<=2 dropoff_delay_ge_30]",
        "P=? [F<=2 dropoff_delay_ge_60]",
        "P=? [F<=2 pickup_delay]",
        "P=? [F<=2 dropoff_delay]",
        "P=? [X vehicle_busy_before_assignment]",
        "P=? [X clear_current_route_ge_15]",
        "P=? [X clear_current_route_ge_30]",
        "P=? [X deadhead_to_pickup_ge_15]",
        "P=? [X deadhead_to_pickup_ge_30]",
        "P=? [X dropoff_slack_le_15]",
        "P=? [X dropoff_slack_le_30]",
        "P=? [X service_path_event_affected]",
        "P=? [X service_path_bridge_affected]",
        "DERIVED: PENDING_REQUESTS_BY_VEHICLE",
        "DERIVED: ETA_PICKUP_BY_VEHICLE",
        "DERIVED: ETA_DROPOFF_BY_VEHICLE",
        "DERIVED: TRAFFIC_LEVEL",
        "P=? [G (!pickup_delay & !dropoff_delay)]",
        "P=? [F<=1 service_complete]",
        "DERIVED: TURNING_INTERVAL",
        "DERIVED: STEP_CONFIDENCE",
        "DERIVED: VEHICLE_OCCUPANCY",
        "P=? [X pickup_delay]",
        "P=? [X dropoff_delay]",
        "DERIVED: EVENT_MULTIPLIER",
        "DERIVED: ACCIDENT_MULTIPLIER",
        "DERIVED: EVENT_EPOCH",
        "DERIVED: ACCIDENT_EPOCH",
        "DERIVED: N_MIN",
        "DERIVED: N_INTERVAL",
        "DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE",
        "DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE",
        "DERIVED: DROPOFF_SLACK_BY_VEHICLE",
        "DERIVED: SERVICE_PATH_DISRUPTION_BY_VEHICLE",
    ]
}
