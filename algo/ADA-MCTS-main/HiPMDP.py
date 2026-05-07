from BNN.BayesianNeuralNetwork import *
import pickle

# Model imports will be done dynamically based on domain
class HiPMDP(object):
    """
    The Hidden Parameters-MDP
    """

    def __init__(self, domain, preset_hidden_params, run_type='full', episode_count=500, bnn_hidden_layer_size=25, bnn_num_hidden_layers=2, bnn_network_weights=None,
                 eps_min=0.15, test_inst=None, create_exp_batch=False, save_results=False,
                 grid_beta=0.23, print_output=False, paratransit_requests_csv_path=None):
        """
        Initialize framework.
        """

        self.__initialize_params()

        # Store arguments
        self.domain = domain
        self.run_type = run_type
        self.preset_hidden_params = preset_hidden_params
        self.bnn_hidden_layer_size = bnn_hidden_layer_size
        self.bnn_num_hidden_layers = bnn_num_hidden_layers
        self.bnn_network_weights = bnn_network_weights
        self.eps_min = eps_min
        self.test_inst = test_inst
        self.create_exp_batch = create_exp_batch
        self.save_results = save_results
        self.grid_beta = grid_beta
        self.print_output = print_output
        self.paratransit_requests_csv_path = paratransit_requests_csv_path
        # Set domain specific hyperparameters
        self.__set_domain_hyperparams()
        self.episode_count = episode_count

    def __initialize_params(self):
        """Initialize standard framework settings."""
        self.instance_count = 1  # number of task instances
        self.weight_count = 5  # number of latent weights
        self.eps_max = 1.0  # initial epsilon value for e-greedy policy
        self.bnn_and_latent_update_interval = 10  # Number of episodes between BNN and latent weight updates
        self.num_strata_samples = 5  # The number of samples we take from each strata of the experience buffer
        self.bnn_num_samples = 100  # number of samples of network weights drawn to get each BNN prediction
        self.bnn_batch_size = 32
        self.bnn_v_prior = 3  # Prior variance on the BNN parameters
        self.bnn_training_epochs = 100  # number of epochs of SGD in each BNN update
        self.num_episodes_avg = 30  # number of episodes used in moving average reward to determine whether to stop DQN training
        self.wb_learning_rate = 0.0005  # latent weight learning rate
        self.bnn_alpha = 0.5  # BNN alpha divergence parameter
        self.eps_decay = 0.999  # Epsilon decay rate
        self.ddqn_batch_size = 50  # DDQN batch size
        # Prioritized experience replay hyperparameters
        self.PER_alpha = 0.2
        self.PER_beta_zero = 0.1
        self.wb_num_epochs = 100  # number of epochs of SGD in each latent weight update

    def __set_domain_hyperparams(self):
        # Import model class dynamically based on domain
        if self.domain == 'frozenlake':
            from nsfrozenlake.nsfrozenlake_v0 import NSFrozenLakeV0 as model
            self.task = model()
        elif self.domain == 'paratransit':
            from nsparatransit.nsparatransit_v0 import NSParatransitV0 as model
            if self.paratransit_requests_csv_path is None:
                raise ValueError("paratransit_requests_csv_path is required for paratransit domain")
            self.task = model(
                w_fulfillment=1.0,
                w_timing=2.0,
                delay_scale=20.0,
                n_requests=10,      # Match data collection
                max_time=480,       # Match data collection
                n_vehicles=5,
                vehicle_capacity=3,
                n_nodes=20,
                traffic_condition=0.3,
                requests_csv_path=self.paratransit_requests_csv_path
            )
        elif self.domain == 'grid':
            from nsfrozenlake.nsfrozenlake_v0 import NSFrozenLakeV0 as model
            self.task = model(beta=self.grid_beta)
        elif self.domain == 'discretegrid':
            from nsfrozenlake.nsfrozenlake_v0 import NSFrozenLakeV0 as model
            self.task = model(time=0)
        else:
            # Default to frozen lake for backward compatibility
            from nsfrozenlake.nsfrozenlake_v0 import NSFrozenLakeV0 as model
            self.task = model()

        # Set num_actions and num_dims
        if hasattr(self.task, 'num_actions'):
            self.num_actions = self.task.num_actions
        elif hasattr(self.task, 'nA'):
            self.num_actions = self.task.nA
        else:
            raise ValueError(f"Cannot determine number of actions for domain: {self.domain}")

        # Get number of state dimensions (input to BNN)
        # For paratransit, we need to initialize the state first
        if self.domain == 'paratransit':
            self.task.reset()
            # PAPER-COMPLIANT: BNN learns p(s'|s,a) where s,s' are observations
            # Input: state_to_observation() = 16D (NO traffic_level!)
            # [decision_epoch, V0.loc, V0.time, V0.occ, ..., V4.loc, V4.time, V4.occ]
            # = 1 + 3*5 = 16
            # Action is separately one-hot encoded (5D: 5 vehicles only, no reject)
            # Total BNN input: 16 (state) + 5 (action) + 5 (latent) = 26D
            self.num_dims = 16
            # PAPER-COMPLIANT: BNN output is next_state_observation (16D)
            # BNN learns complete state transition WITHOUT knowing traffic_level
            self.num_output_dims = 16
        else:
            # PAPER-COMPLIANT: Use state_to_observation() to avoid leaking hidden parameters
            # For frozen lake, this returns observation without intend_prob
            dummy_state_obs = self.task.state_to_observation(self.task.state)
            self.num_dims = len(dummy_state_obs)  # number of state dimensions
            self.num_output_dims = self.num_dims  # For frozen lake, output = input dims

    def __initialize_BNN(self):
        """Initialize the BNN and set pretrained network weights (if supplied)."""
        # Generate BNN layer sizes
        # Option D: Use num_output_dims for output layer size (domain-specific)
        if self.run_type != 'full_linear':
            bnn_layer_sizes = [self.num_dims + self.num_actions + self.weight_count] + [
                self.bnn_hidden_layer_size] * self.bnn_num_hidden_layers + [self.num_output_dims * 2]
        else:
            bnn_layer_sizes = [self.num_dims + self.num_actions] + [
                self.bnn_hidden_layer_size] * self.bnn_num_hidden_layers + [self.num_output_dims * self.weight_count]
        # activation function
        self.bnn_learning_rate = 0.0005
        relu = lambda x: np.maximum(x, 0.0)
        # Gather parameters
        param_set = {
            'bnn_layer_sizes': bnn_layer_sizes,
            'weight_count': self.weight_count,
            # Option D: Use num_output_dims for BNN output dimension
            'num_state_dims': self.num_output_dims,
            'bnn_num_samples': self.bnn_num_samples,
            'bnn_batch_size': self.bnn_batch_size,
            'num_strata_samples': self.num_strata_samples,
            'bnn_training_epochs': self.bnn_training_epochs,
            'bnn_v_prior': self.bnn_v_prior,
            'bnn_learning_rate': self.bnn_learning_rate,
            'bnn_alpha': self.bnn_alpha,
            'wb_learning_rate': self.wb_learning_rate,
            'wb_num_epochs': self.wb_num_epochs
        }
        if self.run_type != 'full_linear':
            self.network = BayesianNeuralNetwork(param_set, nonlinearity=relu)
        else:
            self.network = BayesianNeuralNetwork(param_set, nonlinearity=relu, linear_latent_weights=True)
        # Use previously trained network weights
        if self.bnn_network_weights is not None:
            self.network.weights = self.bnn_network_weights


def train_model(seed, domain, bnn2, weight_set2, best_net_work_error, best_latent_error, local_converge_count, use_global_buffer=False, Nu=3):
    # Load buffer: use global buffer D if available, otherwise use episode buffer D_b
    buffer_file = 'data_buffer/{}_{}_exp_buffer_D'.format(domain, seed) if use_global_buffer else 'data_buffer/{}_{}_exp_buffer_Db'.format(domain, seed)

    # Fall back to D_b if D doesn't exist yet
    import os
    if not os.path.exists(buffer_file):
        buffer_file = 'data_buffer/{}_{}_exp_buffer_Db'.format(domain, seed)

    with open(buffer_file, 'rb') as f:
        exp_buffer = pickle.load(f)

    # Paper: Buffer stores (s, a, r, s', w_b) where w_b is the sampled latent weight
    exp_buffer_np = np.vstack(exp_buffer)

    # Extract components from buffer
    # exp_buffer_np[:, 0] = states
    # exp_buffer_np[:, 1] = actions
    # exp_buffer_np[:, 2] = rewards
    # exp_buffer_np[:, 3] = next_states
    # exp_buffer_np[:, 4] = w_b (latent weights used for each transition)

    X = np.array([np.hstack([exp_buffer_np[tt, 0], exp_buffer_np[tt, 1]]) for tt in range(exp_buffer_np.shape[0])])
    y = np.array([exp_buffer_np[tt, 3] for tt in range(exp_buffer_np.shape[0])])
    # Extract w_b for each transition
    wb_per_transition = np.array([exp_buffer_np[tt, 4] for tt in range(exp_buffer_np.shape[0])])

    # Domain-specific dimensions
    # PAPER-COMPLIANT: BNN learns p(s'|s,a) where s,s' are observable state features
    # - num_input_dims: dimension of state observation (input to BNN)
    # - num_output_dims: dimension of next state observation (output of BNN)
    if domain == 'paratransit':
        num_input_dims = bnn2.num_dims  # 16D state observation (NO traffic_level)
        num_output_dims = bnn2.network.num_state_dims  # 16D next state observation
    else:
        # For frozen lake, input = output (both 2D coordinates)
        num_input_dims = bnn2.network.num_state_dims
        num_output_dims = bnn2.network.num_state_dims

    num_actions = bnn2.num_actions
    num_wb = 5
    relu = lambda x: np.maximum(x, 0.)
    state_diffs = False

    # PAPER-COMPLIANT: For paratransit, buffer should contain next_state_obs (16D)
    # Agent learns travel time dynamics from observed (s,a,s') transitions
    if domain == 'paratransit':
        print(f"[train_model] PAPER-COMPLIANT Paratransit: BNN learns p(s'|s,a)")
        print(f"[train_model] y.shape = {y.shape} (should be [N, 16] for next_state_obs)")
        print(f"[train_model] Sample: exp_buffer_np[0, 3] type={type(exp_buffer_np[0, 3])}, shape={np.array(exp_buffer_np[0, 3]).shape}")
        # Verify dimension
        if y.shape[1] != 16:
            print(f"[ERROR] Expected y.shape[1] = 16 (next_state_obs), got {y.shape[1]}.")
            print(f"[ERROR] Buffer contains old data format. Please re-run data collection with updated state_to_observation().")
            raise ValueError(f"Buffer data incompatible with paper-compliant implementation. Expected 16D next_state_obs, got {y.shape[1]}D")
    bnn_learning_rate = 0.00025
    wb_learning_rate = 0.00025
    if local_converge_count >= 2:
        tuples_list = [(0.0001, 0.0001), (0.0001, 0.0007), (0.0002, 0.0002)]
        chosen_tuple = random.choice(tuples_list)
        bnn_learning_rate, wb_learning_rate = chosen_tuple
    param_set = {
        # BUG FIX: Use num_input_dims for input layer, num_output_dims for output layer
        'bnn_layer_sizes': [num_input_dims + num_actions + num_wb] + [bnn2.bnn_hidden_layer_size] * bnn2.bnn_num_hidden_layers + [
            num_output_dims * 2],  # *2 for mean and variance
        'weight_count': num_wb,
        'num_state_dims': num_output_dims,  # Output dimension
        'bnn_num_samples': 1000,
        'bnn_batch_size': 32,
        'num_strata_samples': 5,
        'bnn_training_epochs': 1,
        'bnn_v_prior': 1,
        'bnn_learning_rate': bnn_learning_rate,
        'bnn_alpha': 0.5,
        'wb_num_epochs': 1,
        'wb_learning_rate': wb_learning_rate
    }
    network_training = BayesianNeuralNetwork(param_set, nonlinearity=relu)
    network_training.weights = bnn2.network.weights
    output_network_weights = bnn2.network.weights
    output_latent_weights = weight_set2

    def get_random_sample(start, stop, size):
        indices_set = set()
        while len(indices_set) < size:
            indices_set.add(np.random.randint(start, stop))
            if len(indices_set) >= stop:
                break
        return np.array(list(indices_set))

    sample_size = min(1000, X.shape[0])

    # Paper Algorithm 1, Line 7: Update w_b from D_b
    # Initialize w_b with mean of sampled weights from buffer
    current_wb = np.mean(wb_per_transition, axis=0).reshape(1, -1)

    # PAPER-COMPLIANT (Issue 2): TuneModel loop uses Nu parameter
    # Paper Algorithm 1, Lines 16-19: for k = 0 to Nu updates
    print(f"[train_model] TuneModel: running Nu={Nu} update iterations")
    for i in range(Nu):
        # Update BNN network weights
        network_training.fit_network(exp_buffer_np, None, 0, state_diffs=state_diffs, use_all_exp=True)

        # Update latent weights w_b using gradient optimization
        # This is the correct implementation of "Update w_b from D_b"
        current_wb = network_training.optimize_latent_weighting_stochastic(
            exp_buffer_np, current_wb, 0, state_diffs=state_diffs, use_all_exp=True
        )

        if i % 1 == 0:
            sample_indices = get_random_sample(0, X.shape[0], sample_size)
            # Use optimized w_b for evaluation
            X_with_wb = np.array([np.hstack([X[idx], current_wb[0]]) for idx in sample_indices])
            l2_errors = network_training.get_td_error(X_with_wb, y[sample_indices],
                                                      location=0.0, scale=1.0, by_dim=False)
            if (np.mean(l2_errors) + np.std(l2_errors)) < best_net_work_error:
                best_net_work_error = (np.mean(l2_errors) + np.std(l2_errors))
            best_latent_error = np.mean(l2_errors)
            output_network_weights = network_training.weights
            output_latent_weights = current_wb[0]  # Use optimized w_b

    # Also compute variance of w_b samples for P_W update
    latent_variance = np.var(wb_per_transition, axis=0)

    return output_network_weights, output_latent_weights, best_net_work_error, best_latent_error, latent_variance