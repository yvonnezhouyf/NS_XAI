import gymnasium as gym
import numpy as np
import pickle
import random


class RLExperienceCollector:
    def __init__(self, env_name, model_class, transition_number=1000, domain='frozenlake', **model_kwargs):
        """
        Initialize experience collector.

        Args:
            env_name: Gym environment name (only for frozen lake)
            model_class: Task model class
            transition_number: Number of transitions to collect per state-action pair
            domain: 'frozenlake' or 'paratransit'
            **model_kwargs: Additional arguments for model initialization
        """
        self.domain = domain

        if domain == 'frozenlake':
            self.env = gym.make(env_name)
            self.task = model_class(intended_prob=model_kwargs.get('intended_prob', 0.4))
        elif domain == 'paratransit':
            self.env = None  # Paratransit doesn't use gym
            self.task = model_class(**model_kwargs)
        else:
            raise ValueError(f"Unknown domain: {domain}")

        self.transition_number = transition_number
        self.exp_buffer = []
        self.state_list = []
        self.episode_buffer_bnn = []
        self.model_class_name = model_class.__name__

    def __encode_action(self, action):
        """One-hot encodes the integer action supplied."""
        if self.domain == 'frozenlake':
            a = np.zeros(self.env.action_space.n, dtype=int)
            a[action] = 1
        elif self.domain == 'paratransit':
            a = np.zeros(self.task.nA, dtype=int)
            a[action] = 1
        return a

    def __encode_state(self, state):
        """Encodes the state in a generic way. This method can be overridden by the model class if needed."""
        encode_method = f"_{self.model_class_name}__encode_state"
        if hasattr(self.task, encode_method):
            return getattr(self.task, encode_method)(state)
        return state

    def collect_experiences_frozenlake(self):
        """Collect experiences for frozen lake environment."""
        action_space = list(range(self.env.action_space.n))
        # Collect valid states that are not terminal
        for state in range(self.env.observation_space.n):
            row, col = self.task.to_m(state)
            letter = self.task.desc[row, col]
            if (letter != b'H') and (letter != b'G'):  # state is not a Hole or Goal (Frozen Lake specific)
                self.state_list.append(state)

        # Collect experiences
        for state in self.state_list:
            for action in action_space:
                for _ in range(self.transition_number):
                    self.task.set_state(state, 1)
                    next_state, reward, done, _ = self.task.step(action)
                    self.episode_buffer_bnn.append(
                        np.reshape(np.array([self.__encode_state(state), self.__encode_action(action),
                                             reward, next_state, 0], dtype=object), [1, 5]))

        exp_list = np.reshape(self.episode_buffer_bnn, [-1, 5])
        self.exp_buffer.extend(exp_list)

    def collect_experiences_paratransit(self):
        """Collect experiences for paratransit environment using random rollouts."""
        # For paratransit, we collect experiences by running random episodes
        num_episodes = 1000  # Number of episodes to collect

        for episode in range(num_episodes):
            state = self.task.reset()
            done = False

            while not done:
                # Get valid actions for current state
                valid_actions = self.task.get_valid_actions(state)
                action = random.choice(valid_actions)

                # Get state observation for BNN (PAPER-COMPLIANT: NO traffic_level)
                state_obs = self.task.state_to_observation(state)

                # Take action
                next_state, reward, done, info = self.task.step(action)

                # Get next state observation (PAPER-COMPLIANT: NO traffic_level)
                next_state_obs = self.task.state_to_observation(next_state)

                # PAPER-COMPLIANT: Store complete next_state_obs for all domains
                # BNN learns p(s'|s,a) where s,s' are observable features (NO hidden params)
                # For paratransit: 16D [epoch, v0_loc, v0_time, v0_occ, ..., v4_loc, v4_time, v4_occ]
                # For frozen lake: 2D [row, col]
                target = next_state_obs

                # Store experience: [state_obs, action_one_hot, reward, next_state_obs, instance_id]
                # BNN learns complete state transition from observed (s,a,s') pairs
                self.episode_buffer_bnn.append(
                    np.reshape(np.array([state_obs, self.__encode_action(action),
                                         reward, target, 0], dtype=object), [1, 5]))

                state = next_state

        exp_list = np.reshape(self.episode_buffer_bnn, [-1, 5])
        self.exp_buffer.extend(exp_list)

    def collect_experiences(self):
        """Collect experiences based on domain."""
        if self.domain == 'frozenlake':
            self.collect_experiences_frozenlake()
        elif self.domain == 'paratransit':
            self.collect_experiences_paratransit()
        else:
            raise ValueError(f"Unknown domain: {self.domain}")


    def save_experiences(self, filename='data_buffer/exp_buffer.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.exp_buffer, f)


# Usage Example
if __name__ == "__main__":
    import sys

    # Get domain from command line argument (default to frozenlake)
    domain = sys.argv[1] if len(sys.argv) > 1 else 'frozenlake'

    if domain == 'frozenlake':
        from nsfrozenlake.nsfrozenlake_v0 import NSFrozenLakeV0

        collector = RLExperienceCollector(
            env_name="FrozenLake-v1",
            model_class=NSFrozenLakeV0,
            domain='frozenlake'
        )
        collector.collect_experiences()
        collector.save_experiences('data_buffer/frozenlake_exp_buffer.pkl')

    elif domain == 'paratransit':
        from nsparatransit.nsparatransit_v0 import NSParatransitV0

        collector = RLExperienceCollector(
            env_name=None,
            model_class=NSParatransitV0,
            domain='paratransit',
            n_vehicles=5,
            vehicle_capacity=3,
            n_nodes=20,
            n_requests=10,
            max_time=480,   # Increased max_time proportionally
            traffic_condition=0.3,
            seed=42,
            w_fulfillment=1.0,  # Weight for fulfillment component
            w_timing=1.0        # Weight for timing component
        )
        collector.collect_experiences()
        collector.save_experiences('data_buffer/paratransit_exp_buffer.pkl')

    else:
        print(f"Unknown domain: {domain}")
        print("Usage: python -m data_collection.data_generation [frozenlake|paratransit]")
        sys.exit(1)