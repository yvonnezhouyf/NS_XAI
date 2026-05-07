"""
Non-Stationary Paratransit Vehicle Assignment Environment
Compatible with ADA-MCTS framework (similar to NSFrozenLakeV0)
"""

import numpy as np
import pandas as pd
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from copy import deepcopy


@dataclass
class PassengerRequest:
    """A passenger request with time windows."""
    request_id: int
    pickup_node: int
    dropoff_node: int
    request_time: float
    earliest_pickup: float
    latest_dropoff: float
    original_id: int = -1  # Original CSV index (preserved for reference)
    
    def __hash__(self):
        return hash(self.request_id)


@dataclass
class VehicleState:
    """Current state of a vehicle (similar to LogiEx Vehicle class)."""
    vehicle_id: int
    current_location: int
    current_time: float
    current_occupancy: int
    capacity: int
    route: list = None  # List of (request_id, location) to visit
    next_time: float = None  # Next time when vehicle will reach next location

    def __post_init__(self):
        if self.route is None:
            self.route = []
        if self.next_time is None:
            self.next_time = self.current_time

    def __hash__(self):
        return hash((self.vehicle_id, self.current_location, int(self.current_time), self.current_occupancy))


@dataclass
class ParatransitState:
    """Complete paratransit system state."""
    decision_epoch: int  # Which request we're assigning (0 to n_requests-1)
    current_request: PassengerRequest
    vehicles: List[VehicleState]
    traffic_level: float  # Non-stationary traffic condition (0.0 to 1.0)
    
    def __hash__(self):
        return hash((self.decision_epoch, self.current_request, tuple(self.vehicles), int(self.traffic_level * 1000)))
    
    def __eq__(self, other):
        if not isinstance(other, ParatransitState):
            return False
        return (self.decision_epoch == other.decision_epoch and 
                self.current_request == other.current_request and
                self.vehicles == other.vehicles and
                abs(self.traffic_level - other.traffic_level) < 0.001)


class NSParatransitV0:
    """
    Non-Stationary Paratransit Environment
    
    Compatible with ADA-MCTS framework:
    - state: ParatransitState object
    - nS: number of decision epochs (= n_requests)
    - nA: number of actions (= n_vehicles)
    - nT: number of timesteps (= n_requests)
    - tau: timestep duration
    - L_p: Lipschitz constant for transition
    - L_r: Lipschitz constant for reward
    """
    
    def __init__(self,
                 n_vehicles: int = 5,
                 vehicle_capacity: int = 3,
                 n_nodes: int = 20,
                 n_requests: int = 10,
                 max_time: int = 1440,  # Increased to 1440 min (24 hours) for real data
                 traffic_condition: float = 0.3,
                 seed: Optional[int] = None,
                 w_fulfillment: float = 1.0,
                 w_timing: float = 2.0,
                 delay_scale: float = 20.0,
                 data_path: Optional[str] = None,
                 use_real_data: bool = True,
                 fixed_request_ids: Optional[List[int]] = None,
                 time_window_scale: float = 1.0,
                 requests_csv_path: Optional[str] = None):
        """
        Args:
            fixed_request_ids: If provided, use these specific request IDs (CSV indices) from
                              train_chains.csv instead of random sampling. Note: requests will
                              still be SORTED by request_time to avoid time-travel issues.
                              Use request.original_id to trace back to CSV index.
            requests_csv_path: If provided, load requests from this CSV file instead of
                              train_chains.csv. CSV must have columns: request_id, pickup_node_id,
                              dropoff_node_id, pickup_time_since_midnight, dropoff_time_since_midnight.
                              When set, fixed_request_ids is ignored and ALL rows from CSV are used.
        """
        self.n_vehicles = n_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.n_nodes = n_nodes
        self.fixed_request_ids = fixed_request_ids
        self.requests_csv_path = requests_csv_path
        # If requests_csv_path provided, n_requests will be set after loading CSV
        # If fixed_request_ids provided, use its length as n_requests
        self.n_requests = len(fixed_request_ids) if fixed_request_ids and not requests_csv_path else n_requests
        self.max_time = max_time
        self.base_traffic_condition = traffic_condition
        self.use_real_data = use_real_data
        self.time_window_scale = time_window_scale  # Scale factor for time windows (1.0=normal, 2.0=2x wider)

        # Data path - default to use_cases/paratransit/data
        if data_path is None:
            # Get project root (go up from ADA-MCTS-main to ns_explainer)
            current_dir = os.path.dirname(os.path.abspath(__file__))  # nsparatransit/
            ada_mcts_dir = os.path.dirname(current_dir)  # ADA-MCTS-main/
            algo_dir = os.path.dirname(ada_mcts_dir)  # algo/
            project_root = os.path.dirname(algo_dir)  # ns_explainer/
            self.data_path = os.path.join(project_root, "use_cases", "paratransit", "data")
        else:
            self.data_path = data_path

        # Design 3: Reward function weights
        self.w_fulfillment = w_fulfillment  # Weight for fulfillment component
        self.w_timing = w_timing  # Weight for timing penalty
        self.delay_scale = delay_scale  # Delay normalization scale for tanh (minutes, TODO: may need tuning)

        # ADA-MCTS required parameters
        self.nS = self.n_requests  # Number of states (decision epochs, matches len of requests)
        self.nA = n_vehicles  # Number of actions (= n_vehicles)
        # NOTE: self.nT removed - MCTS now uses n_requests + 1 directly for terminal state
        self.tau = 1  # Timestep duration
        self.L_p = 5.0  # Lipschitz constant for transition (traffic changes)
        self.L_r = 0.0  # Lipschitz constant for reward (deterministic)

        # Random state
        self.np_random = np.random.RandomState(seed)
        self._initial_seed = seed

        # Request status tracking for reward calculation
        # Status: "pending", "in-transit", "dropped-off"
        self.request_status = {}  # {request_id: status}
        self.request_assignments = {}  # {request_id: vehicle_id}
        self.pickup_times = {}  # {request_id: actual_pickup_time}
        self.dropoff_times = {}  # {request_id: actual_dropoff_time}
        
        # Generate static data
        self._generate_travel_time_matrix()
        self._generate_requests()
        self._initialize_vehicles()
        
        # Current state
        self.state = None
        self.current_request_idx = 0

        # Design 2: Travel time noise parameter
        self.sigma_tt = 0.03  # 3% standard deviation for travel time noise

        # Track current epoch for event timing
        self.current_decision_epoch = 0

        # Event configuration (for event scenario - node-based congestion)
        # These can be set via set_event_config() method
        # Event affects ANY route where pickup OR dropoff is an event node
        self.event_epoch = None  # Epoch when event occurs (None = no event)
        self.event_nodes = set()  # Set of affected nodes {node_id, ...}
        self.event_multiplier = 3.0  # Travel time multiplier for routes involving event nodes

        # Bridge accident configuration (for Case 1 - directional cross-river congestion)
        # Can be set via set_bridge_accident_config() method
        self.bridge_accident_epoch = None  # Epoch when accident occurs (None = no accident)
        self.bridge_accident_multiplier = 2.0  # Travel time multiplier for affected direction
        self.south_nodes = set()  # Nodes on south bank of Tennessee River
        self.north_nodes = set()  # Nodes on north bank of Tennessee River

        # Citywide congestion configuration (Case 3)
        # Can be set via set_congestion_config() method
        self.congestion_epoch = None  # Epoch when citywide congestion begins (None = no congestion)
        self.congestion_traffic_level = 1.0  # Traffic level after congestion onset

        # Traffic level history (for visualization)
        self.traffic_level_history = {}  # {epoch: traffic_level}

    def set_event_config(self, event_epoch: Optional[int], event_nodes: Optional[list] = None,
                         event_multiplier: float = 3.0):
        """
        Configure event parameters for the event scenario (node-based congestion).

        Event affects ANY route where the pickup OR dropoff node is in the event_nodes set.
        This simulates congestion at specific locations (e.g., a stadium event, road closure).

        Args:
            event_epoch: Epoch at which the event occurs (None to disable)
            event_nodes: List/set of affected node IDs, e.g., [5, 12, 3]
                        Any route with from_node OR to_node in this set will be affected.
            event_multiplier: Travel time multiplier for affected routes (default 3x)
        """
        self.event_epoch = event_epoch
        self.event_multiplier = event_multiplier
        
        if event_nodes is not None:
            # Convert to set for O(1) lookup
            self.event_nodes = set(event_nodes) if not isinstance(event_nodes, set) else event_nodes
        else:
            self.event_nodes = set()

    def set_bridge_accident_config(self, accident_epoch: int, multiplier: float = 2.0,
                                   river_bank_csv_path: Optional[str] = None):
        """
        Configure a bridge accident that affects south-to-north cross-river travel.

        Only S→N direction trips get the multiplier applied (simulating a northbound
        lane accident on the Tennessee River bridge in Chattanooga).

        Args:
            accident_epoch: Epoch at which the bridge accident occurs
            multiplier: Travel time multiplier for S→N cross-river routes (default 2.0)
            river_bank_csv_path: Path to river_bank_nodes.csv. If None, auto-detected.
        """
        self.bridge_accident_epoch = accident_epoch
        self.bridge_accident_multiplier = multiplier

        # Load south/north node sets from CSV
        if river_bank_csv_path is None:
            river_bank_csv_path = os.path.join(self.data_path, "river_bank_nodes.csv")

        river_df = pd.read_csv(river_bank_csv_path)
        self.south_nodes = set(river_df[river_df['bank'] == 'south']['node_id'].values)
        self.north_nodes = set(river_df[river_df['bank'] == 'north']['node_id'].values)

    def set_congestion_config(self, congestion_epoch: int, congestion_traffic_level: float = 1.0):
        """
        Configure citywide congestion that changes traffic_level at a specific epoch.

        At congestion_epoch, traffic_level jumps from its base value to
        congestion_traffic_level, affecting ALL OD pairs through the
        interpolation formula: t = (1-c)*base + c*cong.

        Args:
            congestion_epoch: Epoch at which citywide congestion begins
            congestion_traffic_level: Traffic level after congestion onset (default 1.0)
        """
        self.congestion_epoch = congestion_epoch
        self.congestion_traffic_level = congestion_traffic_level

    def _generate_travel_time_matrix(self):
        """Load real travel time matrix (NO FALLBACK)."""
        if self.use_real_data:
            # BUG FIX: No fallback - fail if real data is missing
            # Load base travel time matrix (traffic_level = 0)
            ttm_path = os.path.join(self.data_path, "travel_time_matrix.csv")
            if not os.path.exists(ttm_path):
                raise FileNotFoundError(
                    f"Real data file not found: {ttm_path}\n"
                    f"This environment requires REAL Chattanooga data when use_real_data=True.\n"
                    f"Please ensure the data file exists or set use_real_data=False."
                )

            ttm_df = pd.read_csv(ttm_path, header=None)
            # CRITICAL: Convert from seconds to minutes!
            self.base_travel_times = ttm_df.values / 60.0

            # Load congestion travel time matrix (traffic_level = 1)
            ttm_cong_path = os.path.join(self.data_path, "travel_time_matrix_cong.csv")
            if not os.path.exists(ttm_cong_path):
                raise FileNotFoundError(
                    f"Real data file not found: {ttm_cong_path}\n"
                    f"This environment requires REAL Chattanooga congestion data when use_real_data=True.\n"
                    f"Please ensure the data file exists or set use_real_data=False."
                )

            ttm_cong_df = pd.read_csv(ttm_cong_path, header=None)
            self.cong_travel_times = ttm_cong_df.values / 60.0

            # Update n_nodes based on actual data
            actual_nodes = self.base_travel_times.shape[0]
            if actual_nodes != self.n_nodes:
                # Only print once per session to avoid spam
                if not hasattr(NSParatransitV0, '_printed_node_info'):
                    print(f"[INFO] Using {actual_nodes} nodes from real data (requested {self.n_nodes})")
                    NSParatransitV0._printed_node_info = True
                self.n_nodes = actual_nodes

            # Verify congestion matrix has same shape
            if self.cong_travel_times.shape != self.base_travel_times.shape:
                raise ValueError(
                    f"Congestion matrix shape {self.cong_travel_times.shape} "
                    f"does not match base matrix shape {self.base_travel_times.shape}"
                )

            # Only print once per session to avoid spam
            if not hasattr(NSParatransitV0, '_printed_ttm_info'):
                print(f"[INFO] Loaded real travel time matrix: {self.base_travel_times.shape}")
                print(f"[INFO] Base travel time range: {self.base_travel_times[self.base_travel_times > 0].min():.1f} - {self.base_travel_times[self.base_travel_times > 0].max():.1f} minutes")
                print(f"[INFO] Congestion travel time range: {self.cong_travel_times[self.cong_travel_times > 0].min():.1f} - {self.cong_travel_times[self.cong_travel_times > 0].max():.1f} minutes")
                NSParatransitV0._printed_ttm_info = True
        else:
            self._generate_random_travel_times()
    
    def _generate_random_travel_times(self):
        """Generate random travel time matrix (fallback)."""
        self.base_travel_times = np.random.uniform(5, 30, (self.n_nodes, self.n_nodes))
        np.fill_diagonal(self.base_travel_times, 0)
        # Make symmetric
        self.base_travel_times = (self.base_travel_times + self.base_travel_times.T) / 2
        
    def _generate_requests(self):
        """Load real passenger requests (NO FALLBACK)."""
        if self.use_real_data:
            # BUG FIX: No fallback - fail if real data is missing
            chains_path = os.path.join(self.data_path, "train_chains.csv")
            if not os.path.exists(chains_path):
                raise FileNotFoundError(
                    f"Real data file not found: {chains_path}\n"
                    f"This environment requires REAL Chattanooga data when use_real_data=True.\n"
                    f"Please ensure the data file exists or set use_real_data=False."
                )

            # If requests_csv_path is provided, use that CSV directly (all rows)
            if self.requests_csv_path is not None:
                if not os.path.exists(self.requests_csv_path):
                    raise FileNotFoundError(
                        f"Custom requests CSV not found: {self.requests_csv_path}"
                    )
                chains_df = pd.read_csv(self.requests_csv_path)
                sampled_chains = chains_df  # Use all rows from custom CSV
                self.n_requests = len(sampled_chains)
                print(f"[INFO] Loaded {self.n_requests} requests from custom CSV: {self.requests_csv_path}")
            else:
                chains_df = pd.read_csv(chains_path)

                # Use fixed_request_ids if provided (for reproducibility)
                if self.fixed_request_ids is not None:
                    # Select specific rows by index, preserving the given order
                    try:
                        sampled_chains = chains_df.loc[self.fixed_request_ids]
                    except KeyError as e:
                        raise ValueError(
                            f"Invalid request IDs in fixed_request_ids: {e}. "
                            f"Available indices: 0 to {len(chains_df)-1}"
                        )
                # Otherwise, sample randomly
                elif len(chains_df) < self.n_requests:
                    print(f"[WARNING] Only {len(chains_df)} chains available, requested {self.n_requests}")
                    sampled_chains = chains_df
                else:
                    sampled_chains = chains_df.sample(n=self.n_requests, random_state=self._initial_seed)

            self.requests = []

            # Get min request time for normalization (use request_time_since_midnight)
            min_request_time = sampled_chains['request_time_since_midnight'].min() / 60.0

            for idx, row in sampled_chains.iterrows():
                # CSV columns: request_id, pickup_node_id, dropoff_node_id, 
                #              request_time_since_midnight, pickup_time_since_midnight, dropoff_time_since_midnight
                pickup_node = int(row['pickup_node_id'])
                dropoff_node = int(row['dropoff_node_id'])

                # Convert times from seconds since midnight to minutes, then normalize to start from 0
                # request_time = when the request appears (passenger calls in)
                # earliest_pickup = earliest time passenger can be picked up
                # latest_dropoff = deadline for dropoff
                request_time = (float(row['request_time_since_midnight']) / 60.0) - min_request_time
                earliest_pickup = (float(row['pickup_time_since_midnight']) / 60.0) - min_request_time
                latest_dropoff = (float(row['dropoff_time_since_midnight']) / 60.0) - min_request_time

                # BUG FIX: Ensure nodes are within bounds - FAIL if invalid data
                if pickup_node >= self.n_nodes or dropoff_node >= self.n_nodes:
                    raise ValueError(
                        f"Invalid node IDs in real data: pickup={pickup_node}, dropoff={dropoff_node}, "
                        f"but only {self.n_nodes} nodes available. "
                        f"Data may be corrupted or incompatible with travel time matrix."
                    )

                request = PassengerRequest(
                    request_id=int(idx),  # Temporary, will be reassigned below
                    pickup_node=pickup_node,
                    dropoff_node=dropoff_node,
                    request_time=request_time,
                    earliest_pickup=earliest_pickup,
                    latest_dropoff=latest_dropoff,
                    original_id=int(idx)  # Preserve original CSV index
                )
                self.requests.append(request)

            # Only print once per session to avoid spam
            if not hasattr(NSParatransitV0, '_printed_requests_info'):
                print(f"[INFO] Loaded {len(self.requests)} real passenger requests")
                NSParatransitV0._printed_requests_info = True
        else:
            self._generate_random_requests()

        # ALWAYS sort by request_time to avoid time-travel issues in evolve_all_vehicles_to_time
        # (even for fixed_request_ids, requests must be processed in chronological order)
        self.requests.sort(key=lambda r: r.request_time)
        
        # ALWAYS reassign sequential IDs (0, 1, 2, ...) for correct indexing in _calculate_reward
        # Original CSV index is preserved in original_id field
        for i, req in enumerate(self.requests):
            req.request_id = i
    
    def _generate_random_requests(self):
        """Generate random passenger requests (fallback)."""
        self.requests = []
        for i in range(self.n_requests):
            pickup_node = self.np_random.randint(0, self.n_nodes)
            dropoff_node = self.np_random.randint(0, self.n_nodes)
            while dropoff_node == pickup_node:
                dropoff_node = self.np_random.randint(0, self.n_nodes)
            
            request_time = self.np_random.uniform(0, self.max_time * 0.7)
            # Scale random time windows by time_window_scale
            earliest_pickup = request_time + self.np_random.uniform(0, 10) * self.time_window_scale
            trip_time = self.base_travel_times[pickup_node, dropoff_node]
            latest_dropoff = earliest_pickup + trip_time + self.np_random.uniform(15, 30) * self.time_window_scale
            
            request = PassengerRequest(
                request_id=i,
                pickup_node=pickup_node,
                dropoff_node=dropoff_node,
                request_time=request_time,
                earliest_pickup=earliest_pickup,
                latest_dropoff=latest_dropoff
            )
            self.requests.append(request)
    
    # Default vehicle start positions: 3 south bank + 2 north bank nodes
    # Spread across clusters, avoiding request pickup/dropoff nodes in case1_chains.csv
    VEHICLE_START_NODES = [235, 9, 175, 454, 242]  # S(C2.1), S(C3), S(C4.0), N(C2.0), N(C5.0)

    def _initialize_vehicles(self):
        """Initialize vehicle states with distributed start positions."""
        self.initial_vehicles = []
        for i in range(self.n_vehicles):
            start_node = self.VEHICLE_START_NODES[i] if i < len(self.VEHICLE_START_NODES) else 0
            vehicle = VehicleState(
                vehicle_id=i,
                current_location=start_node,
                current_time=0.0,
                current_occupancy=0,
                capacity=self.vehicle_capacity
            )
            self.initial_vehicles.append(vehicle)
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_request_idx = 0
        self.current_decision_epoch = 0  # Reset epoch counter for event timing

        # Reset traffic level history
        self.traffic_level_history = {}

        # Reset vehicles
        vehicles = [deepcopy(v) for v in self.initial_vehicles]

        # Reset request tracking
        self.request_status = {i: "pending" for i in range(self.n_requests)}
        self.request_assignments = {}
        self.pickup_times = {}
        self.dropoff_times = {}

        # Initial traffic (non-stationary)
        traffic_level = self.base_traffic_condition

        # Record initial traffic level
        self.traffic_level_history[0] = traffic_level
        
        # Create initial state
        self.state = ParatransitState(
            decision_epoch=0,
            current_request=self.requests[0],
            vehicles=vehicles,
            traffic_level=traffic_level
        )
        
        return self.state
    
    def compute_travel_time(self, from_node: int, to_node: int, traffic_level: float,
                            deterministic: bool = False) -> float:
        """
        Compute travel time with traffic and event/accident effects.

        Args:
            from_node: Origin node
            to_node: Destination node
            traffic_level: Current traffic level (0.0 to 1.0)
            deterministic: If True, no noise added (for PCTL checks). If False, adds noise.

        Returns:
            Travel time in minutes
        """
        base_time = self.base_travel_times[from_node, to_node]
        cong_time = self.cong_travel_times[from_node, to_node]

        interpolated_time = base_time * (1.0 - traffic_level) + cong_time * traffic_level

        # Apply event multiplier if from_node OR to_node is in event_nodes set
        if (self.event_epoch is not None and
            self.event_nodes and
            self.current_decision_epoch >= self.event_epoch and
            (from_node in self.event_nodes or to_node in self.event_nodes)):
            interpolated_time = interpolated_time * self.event_multiplier

        # Apply bridge accident multiplier for S→N cross-river trips
        if (self.bridge_accident_epoch is not None and
            self.current_decision_epoch >= self.bridge_accident_epoch and
            from_node in self.south_nodes and to_node in self.north_nodes):
            interpolated_time = interpolated_time * self.bridge_accident_multiplier

        if not deterministic:
            noise = self.np_random.normal(0, self.sigma_tt * interpolated_time)
            interpolated_time = interpolated_time + noise

        return max(interpolated_time, 0.5)

    def state_to_observation(self, state):
        """
        Convert state to observation for BNN.

        PAPER-COMPLIANT: Following the paper's logic, we HIDE traffic_level from the agent.
        Agent must learn travel time dynamics from observed (s,a,s') transitions.

        For paratransit, we use 16D encoding (observable state features only):
        [decision_epoch,
         V0.loc, V0.time, V0.occ, V1.loc, V1.time, V1.occ, ..., V4.loc, V4.time, V4.occ]

        Dimension: 1 (global) + 3*n_vehicles (fleet) = 1 + 15 = 16

        NOT included (hidden parameters like Frozen Lake's intend_prob):
        - traffic_level (this is the hidden environmental parameter!)

        The action is separately one-hot encoded and concatenated by BNN as per paper:
        input = [state_obs, action_one_hot, w_b]

        Args:
            state: ParatransitState or state index
        """
        if isinstance(state, ParatransitState):
            # Global features (only observable info)
            norm_epoch = float(state.decision_epoch) / self.n_requests

            # ALL vehicle features (sorted by vehicle_id for consistency)
            vehicle_features = []
            for v in sorted(state.vehicles, key=lambda x: x.vehicle_id):
                vehicle_features.extend([
                    float(v.current_location) / (self.n_nodes - 1) if self.n_nodes > 1 else 0.0,
                    float(v.current_time) / self.max_time,
                    float(v.current_occupancy) / float(self.vehicle_capacity)
                ])

            # 16D: [epoch] + 15 vehicle features
            # NO traffic_level! (hidden parameter)
            observation = np.array([norm_epoch] + vehicle_features)
            return observation
        else:
            # If state is just an index, return minimal encoding (16D with zeros)
            return np.zeros(1 + 3 * self.n_vehicles)
    
    def get_valid_actions(self, state):
        """
        Get valid actions with capacity pruning.

        Returns:
            List of valid action IDs (vehicles with available capacity)
        """
        valid_vehicles = []

        for vehicle_id, vehicle in enumerate(state.vehicles):
            # Only check capacity constraint - prune vehicles at max capacity
            if vehicle.current_occupancy < vehicle.capacity:
                valid_vehicles.append(vehicle_id)

        return valid_vehicles if valid_vehicles else [0]  # At least one action

    def evolve_vehicle_time(self, vehicle, current_time, target_time):
        """
        Evolve a single vehicle from current_time to target_time.
        Similar to LogiEx's Vehicle.evolve_time() method.

        During this time, the vehicle:
        - Completes its route (pickup/dropoff)
        - Updates its location and time
        """
        # If vehicle has no route, just update time to target
        if len(vehicle.route) == 0:
            vehicle.current_time = target_time
            vehicle.next_time = target_time
            return

        # BUG FIX: Initialize next_time based on current_time if not set properly
        # If next_time is None or less than current_time, we need to compute travel time to first waypoint
        if vehicle.next_time is None or vehicle.next_time < vehicle.current_time:
            if len(vehicle.route) > 0:
                next_location = vehicle.route[0][1]
                # Design 2: Use compute_travel_time with traffic and noise
                adjusted_travel_time = self.compute_travel_time(
                    vehicle.current_location, next_location, self.state.traffic_level
                )
                vehicle.next_time = vehicle.current_time + adjusted_travel_time

        # Process vehicle's route up to target_time
        while vehicle.next_time <= target_time and len(vehicle.route) > 0:
            # Remove completed waypoint and extract request_id
            completed = vehicle.route.pop(0)
            request_id = completed[0]  # Extract request_id from (request_id, location) tuple

            # Vehicle reaches next location in route
            vehicle.current_location = completed[1]  # location
            vehicle.current_time = vehicle.next_time

            # Handle pickup/dropoff occupancy logic
            # Determine if this waypoint is a pickup or dropoff by checking request status

            # BUG FIX: Ensure request_id is in request_status
            # The bug was that request_id might not be in request_status due to initialization issues
            if request_id not in self.request_status:
                # Initialize missing request_id
                self.request_status[request_id] = "pending"

            current_status = self.request_status.get(request_id, "pending")

            # If request is pending, this must be a pickup
            if current_status == "pending":
                vehicle.current_occupancy += 1
                self.request_status[request_id] = "picked-up"
            # If request is picked-up, this must be a dropoff
            elif current_status == "picked-up":
                vehicle.current_occupancy = max(0, vehicle.current_occupancy - 1)
                self.request_status[request_id] = "dropped-off"

            # Calculate next travel time if route continues
            if len(vehicle.route) > 0:
                next_location = vehicle.route[0][1]
                # Design 2: Use compute_travel_time with traffic and noise
                adjusted_travel_time = self.compute_travel_time(
                    vehicle.current_location, next_location, self.state.traffic_level
                )
                vehicle.next_time = vehicle.current_time + adjusted_travel_time
            else:
                # No more route, vehicle is idle at current location
                vehicle.next_time = target_time

        # If still haven't reached target_time, advance to it
        if vehicle.current_time < target_time:
            vehicle.current_time = target_time
            if len(vehicle.route) == 0:
                vehicle.next_time = target_time

    def evolve_all_vehicles_to_time(self, vehicles, target_time):
        """
        Evolve all vehicles from their current times to target_time.
        Similar to LogiEx's move_vehicle() method.
        """
        for vehicle in vehicles:
            self.evolve_vehicle_time(vehicle, vehicle.current_time, target_time)

    def step(self, action: int):
        """
        Take action (assign vehicle to current request).

        Args:
            action: vehicle_id (0 to n_vehicles-1)

        Returns:
            next_state: ParatransitState
            reward: float
            done: bool
            info: dict
        """
        request = self.state.current_request
        vehicle = self.state.vehicles[action]

        # Calculate when the vehicle will actually be available (after finishing existing route)
        if len(vehicle.route) > 0:
            # Vehicle has pending tasks — estimate completion time of entire existing route
            eta_available = vehicle.next_time  # Time to reach first pending waypoint
            prev_loc = vehicle.route[0][1]     # First waypoint location
            for wp_req_id, wp_loc in vehicle.route[1:]:
                leg_time = self.compute_travel_time(prev_loc, wp_loc, self.state.traffic_level)
                eta_available += leg_time
                prev_loc = wp_loc
            # Vehicle departs from last route waypoint to new pickup
            last_route_loc = vehicle.route[-1][1]
            adjusted_travel_time_to_pickup = self.compute_travel_time(
                last_route_loc, request.pickup_node, self.state.traffic_level
            )
            eta_pickup = eta_available + adjusted_travel_time_to_pickup
        else:
            # Vehicle is idle — go directly from current location
            adjusted_travel_time_to_pickup = self.compute_travel_time(
                vehicle.current_location, request.pickup_node, self.state.traffic_level
            )
            eta_pickup = vehicle.current_time + adjusted_travel_time_to_pickup

        adjusted_travel_time_to_dropoff = self.compute_travel_time(
            request.pickup_node, request.dropoff_node, self.state.traffic_level
        )
        eta_dropoff = eta_pickup + adjusted_travel_time_to_dropoff

        # Update request tracking for this assignment
        request_id = request.request_id
        self.request_assignments[request_id] = action
        self.pickup_times[request_id] = eta_pickup
        self.dropoff_times[request_id] = eta_dropoff
        # Status stays "pending" — evolve_vehicle_time() will transition it to
        # "picked-up" (occupancy +1) at pickup, then "dropped-off" (occupancy -1) at dropoff.

        # Design 3: Compute instant reward for this step
        # Component 1: Fulfillment - small positive reward for completing one request
        fulfillment_step = 1.0 / self.n_requests
        fulfillment_component = self.w_fulfillment * fulfillment_step

        # Component 2: Timing penalty - soft penalty for lateness
        pickup_delay = eta_pickup - request.earliest_pickup
        dropoff_delay = eta_dropoff - request.latest_dropoff
        lateness = max(0, pickup_delay) + max(0, dropoff_delay)  # Only penalize late, not early
        timing_penalty = np.tanh(lateness / self.delay_scale)  # Normalized to [0, 1]

        # Instant reward: fulfillment + timing (capacity violation handled by pruning)
        reward = fulfillment_component - self.w_timing * timing_penalty

        # Update vehicle state (LogiEx-style)
        new_vehicles = [deepcopy(v) for v in self.state.vehicles]

        # Add pickup and dropoff to assigned vehicle's route
        assigned_vehicle = new_vehicles[action]

        # BUG FIX: Only set next_time if vehicle has no existing route
        # If vehicle already has route, next_time should remain as arrival time to FIRST waypoint
        had_route_before = len(assigned_vehicle.route) > 0

        assigned_vehicle.route.append((request_id, request.pickup_node))
        assigned_vehicle.route.append((request_id, request.dropoff_node))

        # Calculate next_time for assigned vehicle (when it will reach pickup)
        # Only update if this is the first assignment to an idle vehicle
        if not had_route_before:
            assigned_vehicle.next_time = eta_pickup

        # CRITICAL: Evolve all vehicles to the next decision time
        # This is the key difference from before - ALL vehicles advance in time
        self.current_request_idx += 1
        done = self.current_request_idx >= self.n_requests

        if not done:
            # Get next request's arrival time
            next_request = self.requests[self.current_request_idx]
            next_decision_time = next_request.request_time

            # Evolve ALL vehicles from current time to next decision time
            # This simulates the passage of real time (like LogiEx)
            self.evolve_all_vehicles_to_time(new_vehicles, next_decision_time)
        else:
            # If this is the last request, evolve to dropoff completion time
            # Use the maximum next_time among all vehicles
            final_time = max(v.next_time for v in new_vehicles)
            self.evolve_all_vehicles_to_time(new_vehicles, final_time)

        # Update current_decision_epoch for event timing
        self.current_decision_epoch = self.current_request_idx

        # Citywide congestion: change traffic_level at congestion_epoch
        if (self.congestion_epoch is not None and
                self.current_decision_epoch >= self.congestion_epoch):
            next_traffic_level = self.congestion_traffic_level
        else:
            next_traffic_level = self.state.traffic_level

        # Record traffic level for this epoch (for visualization)
        self.traffic_level_history[self.current_decision_epoch] = next_traffic_level

        if not done:
            next_request = self.requests[self.current_request_idx]
            next_state = ParatransitState(
                decision_epoch=self.current_request_idx,
                current_request=next_request,
                vehicles=new_vehicles,
                traffic_level=next_traffic_level
            )
        else:
            # Terminal state
            next_state = ParatransitState(
                decision_epoch=self.current_request_idx,
                current_request=self.requests[-1],  # Keep last request
                vehicles=new_vehicles,
                traffic_level=next_traffic_level
            )
        
        self.state = next_state

        # Calculate delays for info
        pickup_delay = eta_pickup - request.earliest_pickup
        dropoff_delay = eta_dropoff - request.latest_dropoff

        info = {
            'eta_pickup': eta_pickup,
            'eta_dropoff': eta_dropoff,
            'pickup_delay': pickup_delay,
            'dropoff_delay': dropoff_delay,
            'has_late_pickup': pickup_delay > 0,
            'has_late_dropoff': dropoff_delay > 0,
            'vehicle_occupancy': new_vehicles[action].current_occupancy,
            'vehicle_capacity_remaining': new_vehicles[action].capacity - new_vehicles[action].current_occupancy,
            'request_id': request_id,
            'assigned_vehicle': action
        }

        return next_state, reward, done, info
    
    def _calculate_reward(self):
        """
        Calculate reward using two components: fulfillment and timing.

        Fulfillment component: Ratio of fulfilled requests
        - Fulfilled = requests with status "in-transit" or "dropped-off"

        Timing component: Sum of pickup and dropoff delays
        - For "in-transit": only pickup delay
        - For "dropped-off": pickup delay + dropoff delay
        - Delay = actual_time - expected_time (positive = late, negative = early)

        Returns:
            reward: float = w_fulfillment * fulfillment_ratio + w_timing * timing_penalty
        """
        # Component 1: Trip Fulfillment
        num_fulfilled = sum(1 for status in self.request_status.values()
                           if status in ["picked-up", "in-transit", "dropped-off"])
        fulfillment_ratio = num_fulfilled / self.n_requests

        # Component 2: Timing
        timing_penalty = 0.0
        for request_id, status in self.request_status.items():
            if status in ["picked-up", "in-transit", "dropped-off"]:
                request = self.requests[request_id]

                # Pickup delay (negative = early, positive = late)
                if request_id in self.pickup_times:
                    actual_pickup = self.pickup_times[request_id]
                    # Use earliest_pickup as reference (ideal pickup time)
                    pickup_delay = actual_pickup - request.earliest_pickup
                    timing_penalty += pickup_delay

                # Dropoff delay (only for dropped-off requests)
                if status == "dropped-off" and request_id in self.dropoff_times:
                    actual_dropoff = self.dropoff_times[request_id]
                    # Calculate expected dropoff time based on pickup and travel time
                    # Use latest_dropoff as reference (deadline)
                    dropoff_delay = actual_dropoff - request.latest_dropoff
                    timing_penalty += dropoff_delay

        # Combined reward
        # Note: timing_penalty is added (not subtracted) because negative delays (early) are good
        # and positive delays (late) are bad, so we want to minimize the penalty
        reward = self.w_fulfillment * fulfillment_ratio - self.w_timing * timing_penalty

        return reward

    def is_done(self):
        """Check if episode is complete."""
        return self.current_request_idx >= self.n_requests
    
    def is_terminal(self, state):
        """Check if state is terminal."""
        if isinstance(state, ParatransitState):
            return state.decision_epoch >= self.n_requests
        else:
            # If state is an index
            return state >= self.n_requests

    def _compute_assignment_path_stats(self, current_state, action) -> dict:
        """
        Compute assignment path statistics for a given (state, action) pair.

        Shared helper used by evaluate_atomic_props() (post-MCTS annotation only).
        NOT used by compute_rollout_reward() — that function has its own ETA logic
        using vehicle.next_time to preserve MCTS reproducibility.

        Path definition:
        - If vehicle has route: finish entire remaining route, then deadhead to
          pickup, then pickup to dropoff.
        - If vehicle is idle: go from current location to pickup, then to dropoff.

        Returns dict with keys:
            vehicle_busy_before_assignment (bool)
            clear_current_route_time      (float, minutes)
            deadhead_to_pickup            (float, minutes)
            eta_pickup                    (float, absolute time)
            eta_dropoff                   (float, absolute time)
            dropoff_slack                 (float, minutes; negative = late)
            service_path_event_affected   (bool)
            service_path_bridge_affected  (bool)
        """
        request = current_state.current_request
        vehicle = current_state.vehicles[action]
        traffic_level = getattr(current_state, 'traffic_level', self.base_traffic_condition)

        vehicle_busy = False
        clear_current_route_time = 0.0
        deadhead_to_pickup = 0.0
        event_affected = False
        bridge_affected = False

        # Collect all legs of the service path for disruption checks
        service_legs = []  # list of (from_node, to_node)

        if hasattr(vehicle, 'route') and len(vehicle.route) > 0:
            vehicle_busy = True

            # --- clear current route (fully deterministic) ---
            # Compute every leg via compute_travel_time so the metric is
            # independent of the persisted next_time value.
            eta_available = vehicle.current_time
            prev_loc = vehicle.current_location
            for _, wp_loc in vehicle.route:
                service_legs.append((prev_loc, wp_loc))
                leg_time = self.compute_travel_time(
                    prev_loc, wp_loc, traffic_level, deterministic=True)
                eta_available += leg_time
                prev_loc = wp_loc
            clear_current_route_time = max(0.0, eta_available - vehicle.current_time)

            # --- deadhead: last route waypoint -> pickup ---
            last_route_loc = vehicle.route[-1][1]
            deadhead_to_pickup = self.compute_travel_time(
                last_route_loc, request.pickup_node, traffic_level, deterministic=True)
            service_legs.append((last_route_loc, request.pickup_node))
            eta_pickup = eta_available + deadhead_to_pickup
        else:
            # Vehicle idle
            deadhead_to_pickup = self.compute_travel_time(
                vehicle.current_location, request.pickup_node, traffic_level, deterministic=True)
            service_legs.append((vehicle.current_location, request.pickup_node))
            eta_pickup = vehicle.current_time + deadhead_to_pickup

        # --- pickup -> dropoff ---
        travel_to_dropoff = self.compute_travel_time(
            request.pickup_node, request.dropoff_node, traffic_level, deterministic=True)
        service_legs.append((request.pickup_node, request.dropoff_node))
        eta_dropoff = eta_pickup + travel_to_dropoff

        # --- dropoff slack ---
        dropoff_slack = request.latest_dropoff - eta_dropoff

        # --- disruption checks along full service path ---
        eval_epoch = getattr(current_state, 'decision_epoch', self.current_decision_epoch)
        # Event check
        if (self.event_epoch is not None and
                self.event_nodes and
                eval_epoch >= self.event_epoch):
            for from_n, to_n in service_legs:
                if from_n in self.event_nodes or to_n in self.event_nodes:
                    event_affected = True
                    break
        # Bridge S→N check
        if (self.bridge_accident_epoch is not None and
                eval_epoch >= self.bridge_accident_epoch):
            for from_n, to_n in service_legs:
                if from_n in self.south_nodes and to_n in self.north_nodes:
                    bridge_affected = True
                    break

        return {
            'vehicle_busy_before_assignment': vehicle_busy,
            'clear_current_route_time': clear_current_route_time,
            'deadhead_to_pickup': deadhead_to_pickup,
            'eta_pickup': eta_pickup,
            'eta_dropoff': eta_dropoff,
            'dropoff_slack': dropoff_slack,
            'service_path_event_affected': event_affected,
            'service_path_bridge_affected': bridge_affected,
        }

    def compute_rollout_reward(self, state, action, next_state):
        """
        Compute rollout reward based on (s, a, s') for MCTS simulation.

        This method estimates the reward using:
        1. Fulfillment component
        2. Timing penalty (estimated from state, action, next_state)

        IMPORTANT: This is on the MCTS hot path. Do NOT refactor to use
        _compute_assignment_path_stats() — it computes the first route leg
        differently (fully deterministic vs next_time), which would change
        reward values and break MCTS reproducibility.

        Args:
            state: Current ParatransitState
            action: Action taken (vehicle_id)
            next_state: Next ParatransitState (from BNN prediction)

        Returns:
            Estimated reward for this transition
        """

        # For vehicle assignment actions, compute reward similar to step()
        request = state.current_request

        # Get assigned vehicle from state
        if action < len(state.vehicles):
            vehicle = state.vehicles[action]
        else:
            # Invalid action, return large negative penalty
            return -1.0

        # CONSISTENT WITH evaluate_atomic_props: Align epoch for correct event multiplier
        # Without this, compute_travel_time(deterministic=True) may use wrong epoch for events
        eval_epoch = getattr(state, 'decision_epoch', self.current_decision_epoch)
        saved_epoch = self.current_decision_epoch
        self.current_decision_epoch = eval_epoch

        try:
            traffic_level = getattr(state, 'traffic_level', self.base_traffic_condition)

            # Account for vehicle's existing route before going to new pickup
            if hasattr(vehicle, 'route') and len(vehicle.route) > 0:
                # Estimate when vehicle finishes its current route
                eta_available = getattr(vehicle, 'next_time', vehicle.current_time)
                prev_loc = vehicle.route[0][1]
                for _, wp_loc in vehicle.route[1:]:
                    leg_time = self.compute_travel_time(
                        prev_loc, wp_loc, traffic_level, deterministic=True)
                    eta_available += leg_time
                    prev_loc = wp_loc
                last_route_loc = vehicle.route[-1][1]
                travel_to_pickup = self.compute_travel_time(
                    last_route_loc, request.pickup_node, traffic_level, deterministic=True)
                eta_pickup = eta_available + travel_to_pickup
            else:
                travel_to_pickup = self.compute_travel_time(
                    vehicle.current_location, request.pickup_node, traffic_level, deterministic=True)
                eta_pickup = vehicle.current_time + travel_to_pickup

            travel_to_dropoff = self.compute_travel_time(
                request.pickup_node, request.dropoff_node, traffic_level, deterministic=True)
            eta_dropoff = eta_pickup + travel_to_dropoff
        finally:
            self.current_decision_epoch = saved_epoch

        # Design 3: Compute reward (same as step())
        fulfillment_step = 1.0 / self.n_requests
        fulfillment_component = self.w_fulfillment * fulfillment_step

        # Timing penalty
        pickup_delay = eta_pickup - request.earliest_pickup
        dropoff_delay = eta_dropoff - request.latest_dropoff
        lateness = max(0, pickup_delay) + max(0, dropoff_delay)
        timing_penalty = np.tanh(lateness / self.delay_scale)

        reward = fulfillment_component - self.w_timing * timing_penalty
        return reward

    def evaluate_atomic_props(self, current_state, action, next_state, ctx=None) -> Dict[str, bool]:
        """
        Evaluate atomic propositions for trace-based PCTL evaluation.

        This is the core interface for the trace-based PCTL system.
        Returns truth values of all atomic propositions at this step.

        Atomic propositions covered:
        - service_complete, violation, capacity_violation, time_window_violation
        - pickup_delay, dropoff_delay, any_delay
        - pickup_delay_ge_5/15/30/60, dropoff_delay_ge_5/15/30/60
        - carpool_active, capacity_full, any_vehicle_idle, all_vehicles_busy
        - vehicle_busy_before_assignment: action vehicle has pending route
        - clear_current_route_ge_15/30: clearing existing route takes >= threshold
        - deadhead_to_pickup_ge_15/30: deadhead from route end to pickup >= threshold
        - dropoff_slack_le_15/30: time margin before deadline <= threshold
        - service_path_event_affected: service path crosses event nodes
        - service_path_bridge_affected: service path crosses bridge S→N

        Args:
            current_state: ParatransitState before action
            action: Vehicle ID assigned
            next_state: ParatransitState after action (from BNN prediction)
            ctx: Optional context dict (unused, for future extension)

        Returns:
            Dict[str, bool]: {ap_name: truth_value} for all atomic propositions
        """
        props = {
            # Core completion/violation
            'service_complete': False,
            'violation': False,
            'capacity_violation': False,
            'time_window_violation': False,
            # Delay events
            'pickup_delay': False,
            'dropoff_delay': False,
            'any_delay': False,
            # Delay threshold events (5/15/30/60 minutes)
            'pickup_delay_ge_5': False,
            'pickup_delay_ge_15': False,
            'pickup_delay_ge_30': False,
            'pickup_delay_ge_60': False,
            'dropoff_delay_ge_5': False,
            'dropoff_delay_ge_15': False,
            'dropoff_delay_ge_30': False,
            'dropoff_delay_ge_60': False,
            # Vehicle state events
            'carpool_active': False,
            'capacity_full': False,
            'any_vehicle_idle': False,
            'all_vehicles_busy': False,
            # Assignment path APs (new)
            'vehicle_busy_before_assignment': False,
            'clear_current_route_ge_15': False,
            'clear_current_route_ge_30': False,
            'deadhead_to_pickup_ge_15': False,
            'deadhead_to_pickup_ge_30': False,
            'dropoff_slack_le_15': False,
            'dropoff_slack_le_30': False,
            'service_path_event_affected': False,
            'service_path_bridge_affected': False,
        }

        try:
            # 1. Service complete: all requests processed
            if hasattr(next_state, 'decision_epoch'):
                if next_state.decision_epoch >= self.n_requests:
                    props['service_complete'] = True

            # 2. Vehicle-level checks from next_state (scan ALL vehicles)
            has_idle = False
            if hasattr(next_state, 'vehicles'):
                for vehicle in next_state.vehicles:
                    if hasattr(vehicle, 'current_occupancy') and hasattr(vehicle, 'capacity'):
                        if vehicle.current_occupancy > vehicle.capacity:
                            props['capacity_violation'] = True
                            props['violation'] = True
                        if vehicle.current_occupancy == vehicle.capacity:
                            props['capacity_full'] = True
                        if vehicle.current_occupancy >= 2:
                            props['carpool_active'] = True

                    if hasattr(vehicle, 'route'):
                        if len(vehicle.route) == 0:
                            has_idle = True
                    else:
                        has_idle = True

                props['any_vehicle_idle'] = has_idle
                props['all_vehicles_busy'] = not has_idle

            # 3. Timing checks via shared helper
            if hasattr(current_state, 'current_request') and current_state.current_request is not None:
                request = current_state.current_request

                eval_epoch = getattr(current_state, 'decision_epoch', self.current_decision_epoch)
                saved_epoch = self.current_decision_epoch
                self.current_decision_epoch = eval_epoch

                try:
                    if action < len(current_state.vehicles):
                        stats = self._compute_assignment_path_stats(current_state, action)
                        eta_pickup = stats['eta_pickup']
                        eta_dropoff = stats['eta_dropoff']

                        # --- existing delay APs ---
                        pickup_delay_mins = 0.0
                        dropoff_delay_mins = 0.0

                        if hasattr(request, 'earliest_pickup'):
                            pickup_delay_mins = max(0.0, eta_pickup - request.earliest_pickup)
                            if pickup_delay_mins > 0:
                                props['pickup_delay'] = True
                                for th in (5, 15, 30, 60):
                                    if pickup_delay_mins >= th:
                                        props[f'pickup_delay_ge_{th}'] = True

                        if hasattr(request, 'latest_dropoff'):
                            dropoff_delay_mins = max(0.0, eta_dropoff - request.latest_dropoff)
                            if dropoff_delay_mins > 0:
                                props['dropoff_delay'] = True
                                for th in (5, 15, 30, 60):
                                    if dropoff_delay_mins >= th:
                                        props[f'dropoff_delay_ge_{th}'] = True

                        # --- new assignment path APs ---
                        props['vehicle_busy_before_assignment'] = stats['vehicle_busy_before_assignment']

                        ccrt = stats['clear_current_route_time']
                        props['clear_current_route_ge_15'] = ccrt >= 15
                        props['clear_current_route_ge_30'] = ccrt >= 30

                        dtp = stats['deadhead_to_pickup']
                        props['deadhead_to_pickup_ge_15'] = dtp >= 15
                        props['deadhead_to_pickup_ge_30'] = dtp >= 30

                        ds = stats['dropoff_slack']
                        props['dropoff_slack_le_15'] = ds <= 15
                        props['dropoff_slack_le_30'] = ds <= 30

                        props['service_path_event_affected'] = stats['service_path_event_affected']
                        props['service_path_bridge_affected'] = stats['service_path_bridge_affected']

                    # Derive compound events
                    props['any_delay'] = props['pickup_delay'] or props['dropoff_delay']
                    props['time_window_violation'] = props['dropoff_delay']
                    if props['time_window_violation']:
                        props['violation'] = True

                finally:
                    self.current_decision_epoch = saved_epoch

        except Exception:
            pass

        return props

    def evaluate_terminal_props(self, terminal_state) -> Dict[str, bool]:
        """
        Evaluate atomic propositions for a terminal state (no action/next_state).

        This is called when a rollout starts from or reaches a terminal state.
        We evaluate the state's properties to ensure traces are never empty.

        Args:
            terminal_state: ParatransitState that is terminal (decision_epoch >= n_requests)

        Returns:
            Dict[str, bool]: {ap_name: truth_value} for all atomic propositions
        """
        props = {
            # Core completion/violation
            'service_complete': True,  # Terminal = all requests processed
            'violation': False,
            'capacity_violation': False,
            'time_window_violation': False,
            # Delay events
            'pickup_delay': False,
            'dropoff_delay': False,
            'any_delay': False,
            # Delay threshold events (5/15/30/60 minutes)
            'pickup_delay_ge_5': False,
            'pickup_delay_ge_15': False,
            'pickup_delay_ge_30': False,
            'pickup_delay_ge_60': False,
            'dropoff_delay_ge_5': False,
            'dropoff_delay_ge_15': False,
            'dropoff_delay_ge_30': False,
            'dropoff_delay_ge_60': False,
            # Vehicle state events
            'carpool_active': False,
            'capacity_full': False,
            'any_vehicle_idle': False,
            'all_vehicles_busy': False
        }

        try:
            # Check vehicle states in terminal state
            if hasattr(terminal_state, 'vehicles'):
                all_busy = True
                has_idle = False
                for vehicle in terminal_state.vehicles:
                    # Check occupancy
                    occ = getattr(vehicle, 'current_occupancy', 0)
                    cap = getattr(vehicle, 'capacity', 4)
                    route = getattr(vehicle, 'route', [])

                    if occ > cap:
                        props['capacity_violation'] = True
                        props['violation'] = True
                    if occ >= 2:
                        props['carpool_active'] = True
                    if occ >= cap:
                        props['capacity_full'] = True

                    if len(route) == 0:
                        has_idle = True
                        all_busy = False

                props['any_vehicle_idle'] = has_idle
                props['all_vehicles_busy'] = all_busy

            # Check accumulated violations from completed requests
            # Note: Threshold APs (pickup_delay_ge_*, dropoff_delay_ge_*) are not updated here
            # because terminal state only has boolean flags, not numeric delay values.
            # Thresholds are evaluated during transition steps in evaluate_atomic_props().
            if hasattr(terminal_state, 'completed_requests'):
                for req_id in terminal_state.completed_requests:
                    if hasattr(terminal_state, 'request_violations'):
                        violations = terminal_state.request_violations.get(req_id, {})
                        if violations.get('time_window'):
                            props['time_window_violation'] = True
                            props['violation'] = True
                        if violations.get('pickup_delay'):
                            props['pickup_delay'] = True
                            props['any_delay'] = True
                        if violations.get('dropoff_delay'):
                            props['dropoff_delay'] = True
                            props['any_delay'] = True

        except Exception:
            pass

        return props

    def instant_reward_byindex(self, state_index):
        """
        Get instant reward for state index (used by MCTS rollout).

        For paratransit, we use a simplified reward signal:
        - Non-terminal steps: small positive reward for progress (fulfillment component)
        - Terminal state: bonus if good service quality

        This allows MCTS to differentiate between actions during rollout.
        The actual detailed reward (with timing penalties) is still used for
        BNN training via the step() method.

        Args:
            state_index: Can be integer (epoch) or ParatransitState object
        """
        # Extract decision_epoch from ParatransitState if needed
        if hasattr(state_index, 'decision_epoch'):
            epoch = state_index.decision_epoch
        else:
            epoch = int(state_index)

        if epoch >= self.n_requests:
            # Terminal state: calculate episode quality
            # This is a simplified evaluation for MCTS rollout
            return 1.0  # Completed all requests
        else:
            # Non-terminal: small reward for making progress
            # This gives MCTS a signal to prefer actions that complete requests
            return 0.02  # Equivalent to fulfillment component (1/50)


# Utility functions for distribution (similar to nsfrozenlake)
def distribution_from_dict(state_to_prob_dict):
    """Convert state->prob dict to distribution."""
    return state_to_prob_dict


def uniformly_random_policy(state):
    """Return uniform distribution over actions."""
    # Not used in ADA-MCTS but included for compatibility
    return {}
