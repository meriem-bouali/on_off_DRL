from .utils.custom_abc_meta import CustomABCMeta
from .utils.sum_tree_disk import SumTree


import os
import numpy as np
# from collections import deque

import os


class ReplayMemory(metaclass=CustomABCMeta):
    """
    Abstract base class for replay memory
    This class defines a common interface for different types of replay buffers, including methods for storing and sampling transitions.

    Attributes:
        batch_size (int): Number of transitions to sample.
        buffer_capacity (int): Maximum number of transitions the buffer can hold.
    """

    def __init__(self, buffer_capacity: int, batch_size: int, state_shape: tuple, model: str):
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.cache = os.path.join(os.path.dirname(os.path.dirname(__file__)), "save", "cache_buffer", model)
        self.mode = "r+" if os.path.exists(self.cache + "/states.dat") else "w+"

        os.makedirs(self.cache, exist_ok=True)
        self.state_shape = state_shape

    def store_transitions(self, obses, actions, rews, dones, new_obses):
        """Stores a batch of transitions in the replay buffer"""
        raise NotImplementedError

    def sample_transitions(self, step: int):
        """Samples a batch of transitions from the replay buffer"""
        raise NotImplementedError


class ReplayMemoryNaive(ReplayMemory):
    """
    A naive implementation of ReplayMemory using a deque data structure.
    This implementation stores transitions in a first-in, first-out manner and samples them uniformly.

    Attributes:
        replay_buffer (deque): A deque that holds the transitions. The maximum length of the deque is defined by buffer_capacity.

     Methods:
        store_transitions: Stores a batch of transitions in the deque. When the buffer is full, the oldest transitions are automatically discarded.
        sample_transitions: Randomly samples a batch of transitions from the deque.
        from_dic: Reconstructs the ReplayMemoryNaive instance from a dictionary.
        to_dic: Converts the ReplayMemoryNaive instance to a dictionary for serialization.
        len: Returns the number of transitions currently stored in the replay buffer.


    """

    def __init__(self, *args, **kwargs):
        super(ReplayMemoryNaive, self).__init__(*args, **kwargs)

        self.data_pointer = 0  # Points to the next self.data_pointer/index to write in the data array.
        self.size = 0

        # self.replay_buffer: deque = deque(maxlen=self.buffer_capacity)
        # Allocate memory-mapped arrays on disk
        self.states = np.memmap(
            os.path.join(self.cache, "states.dat"), dtype=np.float32, mode=self.mode, shape=(self.buffer_capacity, *self.state_shape)
        )
        self.actions = np.memmap(os.path.join(self.cache, "actions.dat"), dtype=np.int64, mode=self.mode, shape=(self.buffer_capacity,))
        self.rewards = np.memmap(os.path.join(self.cache, "rewards.dat"), dtype=np.float32, mode=self.mode, shape=(self.buffer_capacity,))
        self.next_states = np.memmap(
            os.path.join(self.cache, "next_states.dat"), dtype=np.float32, mode=self.mode, shape=(self.buffer_capacity, *self.state_shape)
        )
        self.dones = np.memmap(os.path.join(self.cache, "dones.dat"), dtype=np.float32, mode=self.mode, shape=(self.buffer_capacity,))

    def to_dict(self):
        """
        Convert the instance of ReplayMemoryNaive to a dictionary.

        This method is useful for saving the state of the SumTree instance to a file.
        The dictionary representation allows for easy serialization and storage of
        the ReplayMemoryNaive's state, which can later be used to reconstruct the instance.

        Returns:
            dict: A dictionary representation of the ReplayMemoryNaive instance containing all essential attributes.

        """

        return {"self.data_pointer": self.data_pointer, "size": self.size}

    def from_dict(self, buffer_dict: dict):
        """
        Convert a dictionary back to an instance of ReplayMemoryNaive

        This method reconstructs a ReplayMemoryNaive instance from a dictionary representation.
        The dictionary, typically loaded from a file, contains all necessary attributes
        to restore the state of the ReplayMemoryNaive

        Args:
            buffer_dict (dict): A dictionary containing the attributes of the SumTree instance.
        """
        self.data_pointer = buffer_dict["self.data_pointer"]
        self.size = buffer_dict["size"]

    def len(self):
        """Returns the number of stored experiences in the replay buffer"""
        return self.size

    def store_transitions(self, obses: np.ndarray, actions: list, rews: np.ndarray, dones: np.ndarray, new_obses: np.ndarray):
        """
        Stores each transition (observation, action, reward, done flag, and next observation) in the replay buffer.

        Yields:
            int: The index of the transition where an episode ends.
        """
        for e, (obs, action, rew, done, new_obs) in enumerate(zip(obses, actions, rews, dones, new_obses)):
            idx = self.data_pointer
            self.states[idx] = obs
            self.actions[idx] = action
            self.rewards[idx] = rew
            self.next_states[idx] = new_obs
            self.dones[idx] = done

            self.data_pointer = (self.data_pointer + 1) % self.buffer_capacity
            self.size = min(self.size + 1, self.buffer_capacity)

            #  When an episode ends, yield the index of the corresponding transition
            if done:
                yield e

    def sample_transitions(self, step=None) -> list:
        """
        Samples a random batch of transitions from the replay buffer.

        Args:
            step: Not used in this implementation. Included for compatibility with the base class.

        """

        indices = np.random.choice(self.size, self.batch_size, replace=False)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        next_states = self.next_states[indices]

        batch = list(zip(states, actions, rewards, dones, next_states))  # return batch of transition
        return batch


# https://danieltakeshi.github.io/2019/07/14/per/
class ReplayMemoryPrioritized(ReplayMemory):
    def __init__(self, buffer_capacity: int, batch_size: int, eps_dec: int, state_shape: tuple, model: str):
        """
        A prioritized implementation of ReplayMemory using a SumTree data structure.
        This implementation gives higher sampling probability to transitions with higher temporal-difference (TD) error

        Args:
            buffer_capacity (int): The maximum number of transitions the buffer can hold.
            batch_size (int): The number of transitions to sample from the buffer.
            eps_dec (float): The decay rate for the beta parameter
                (`beta` controls the probability correction weight of each transition).

        Attributes:
            replay_buffer (SumTree): A SumTree that stores transitions with their associated priorities.
            epsilon (float): Small constant added to TD errors to ensure non-zero priorities.
            alpha (float): Controls the degree of prioritization
                (0 corresponds to uniform sampling, 1 to full prioritization).
            beta_start (float): Initial value of the importance-sampling correction factor.
            beta_end (float): Final value of the importance-sampling correction factor.
            beta_inc (float): Increment rate of beta over training steps.
            max_priority_high (float): Initial high priority assigned to new experiences.

        Methods:
            from_dict: Reconstructs the ReplayMemoryPrioritized instance from a dictionary.
            to_dict: Converts the ReplayMemoryPrioritized instance to a dictionary for serialization.
            len: Returns the number of transitions currently stored in the replay buffer.
        """
        super(ReplayMemoryPrioritized, self).__init__(buffer_capacity, batch_size, state_shape, model)

        # Initialize the SumTree that stores transitions with their associated priorities.
        self.replay_buffer: SumTree = SumTree(self.buffer_capacity, state_shape=self.state_shape, cache=self.cache, mode=self.mode)

        # Small constant added to TD error to avoid zero priority.
        self.epsilon = 0.0001

        # Hyperparameter that controls prioritization strength
        # (0 corresponds to uniform sampling, 1 to full prioritization).
        self.alpha = 0.6

        # Initial importance-sampling correction factor. For addressing the bias introduced by non-uniform sampling.
        self.beta_start = 0.4

        # Final value of importance-sampling weight factor
        self.beta_end = 1.0

        # Number of steps used to gradually reduce beta from beta_start to beta_end
        self.beta_inc = eps_dec

        # Default high priority assigned to newly added experiences.
        self.max_priority_high = 1.0

    def len(self):
        """Returns the number of stored experiences in the replay buffer"""
        return self.replay_buffer.len()

    def to_dict(self):
        """
        Convert the ReplayMemoryPrioritized instance to a dictionary.

        This method is useful for saving the state of the ReplayMemoryPrioritized to a file.
        The dictionary representation allows easy serialization and storage
        of the ReplayMemoryPrioritized state, which can later be used to reconstruct the ReplayMemoryPrioritized .

        Returns:.
            dict: A dictionary representation of the ReplayMemoryPrioritized instance containing all essential attributes.

        """
        return {
            "max_priority_high": self.max_priority_high,
            "replay_buffer": self.replay_buffer.to_dict(),
        }

    def from_dict(self, buffer_dict: dict):
        """
        Restore a ReplayMemoryPrioritized instance from a dictionary.

        This method reconstructs a ReplayMemoryPrioritized instance from a dictionary representation.
        The dictionary,loaded from a file, contains all necessary attributes to restore the replay buffer state.

        Args:
            buffer_dict (dict):  Dictionary containing serialized replay buffer attributes.
        """
        self.max_priority_high = buffer_dict["max_priority_high"]
        self.replay_buffer.from_dict(buffer_dict["replay_buffer"])

    def store_transitions(self, obses, actions, rews, dones, new_obses):
        """
        Store a batch of transitions in prioritized replay memory.

        Each new experience is assigned the maximum current priority so it has  a high probability of being sampled at least once before its priority
        is updated based on TD error.

        Yields:
            int: The index of the transition where the episode ended.
        """

        max_priority = self.replay_buffer.max_priority

        # If the tree is empty (max priority = 0), assign `max_priority_high` to new experiences.
        # Otherwise, new experiences will directly receive `max_priority`.
        if max_priority == 0:
            max_priority = self.max_priority_high

        # Add each transition to the SumTree.
        for e, (obs, action, rew, done, new_obs) in enumerate(zip(obses, actions, rews, dones, new_obses)):
            self.replay_buffer.add(max_priority, obs, action, rew, done, new_obs)

            # yields indices of transitions where episodes end.
            if done:
                yield e

    def sample_transitions(self, step: int):
        """
        Sample a batch of transitions from prioritized replay memory.
        Importance-sampling weights (beta) are computed for each sampled transition to compensate for the bias introduced by prioritized sampling.

        Args:
            step (int): Current training step, used to gradually reduce the beta parameter.
        """
        is_weights, tree_indices, transitions = [], [], []

        # Calculate the size of each segment in the priority distribution
        priority_segment = self.replay_buffer.total_priority / self.batch_size

        # Linearly interpolate beta from beta_start to beta_end.
        beta = np.interp(step, [0, self.beta_inc], [self.beta_start, self.beta_end])

        # Calculate the maximum importance-sampling weight.
        prob_min = self.replay_buffer.min_priority / self.replay_buffer.total_priority
        max_is_weight = pow(self.replay_buffer.size * prob_min, -beta)

        # Sample transitions based on their priority
        for i in range(self.batch_size):
            v = np.random.uniform(priority_segment * i, priority_segment * (i + 1))

            tree_index, priority, transition = self.replay_buffer.get_leaf(v)

            # Calculate the probability of the sampled transition and its importance-sampling weight.
            prob_i = priority / self.replay_buffer.total_priority
            is_weight_i = pow(self.replay_buffer.size * prob_i, -beta) / max_is_weight

            # Store the weights, indices, and transitions.
            is_weights.append(is_weight_i)
            tree_indices.append(tree_index)
            transitions.append(transition)

        return is_weights, tree_indices, transitions

    def update_batch_priorities(self, tree_indices: list, abs_td_errors_np: np.ndarray):
        """
        Update the priorities of a previously sampled batch of transitions in the SumTree

        Args:
            tree_indices (list): Indices of the sampled transitions in the SumTree.
            abs_td_errors_np (numpy.ndarray): Absolute TD errors of the sampled transitions.
        """

        # Calculate the new priorities using the absolute TD errors.
        priorities = list(np.power(np.minimum(abs_td_errors_np + self.epsilon, self.max_priority_high), self.alpha))

        # Update each transition's priority in the SumTree.
        for i, p in zip(tree_indices, priorities):
            self.replay_buffer.update(i, p)
