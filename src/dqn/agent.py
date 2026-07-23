from .utils.custom_abc_meta import CustomABCMeta, abstract_attribute
from .network import DeepQNetwork, DuelingDeepQNetwork
from .dqn_config import HYPER_PARAMS

if HYPER_PARAMS.buffer_location == "disk":
    from .replay_memory_disk import ReplayMemoryNaive, ReplayMemoryPrioritized
else:
    from .replay_memory_ram import ReplayMemoryNaive, ReplayMemoryPrioritized


from colorama import Fore
import time
import math
import random
import numpy as np
from collections import deque

from typing import Union
from .agent_mixin import AgentMixin

import torch as T
from torch.utils.tensorboard.writer import SummaryWriter

"""
This module implements different Deep Q-Network (DQN) algorithms.
    - The `Agent` class defines the common attributes and methods required by the DQN algorithms, such as model saving/loading, logging,
      storing transitions, action selection, etc.
    - The `SimpleAgent`, `DoubleAgent`, and `PerDoubleAgent` classes inherit from the `Agent` class and define the `learn()` function, 
      which performs a single learning step for a given DQN algorithm.
    - The `DQNAgent`, `DoubleDQNAgent`, `DuelingDoubleDQNAgent`, and `PerDuelingDoubleDQNAgent` classes initialize the neural networks 
      and replay memory required for each algorithm.
"""


class Agent(AgentMixin, metaclass=CustomABCMeta):
    """
    Abstract base class for DQN agent. It must be extended by specific DQN agent implementations.
    Define the common attributes and methods required by the DQN algorithmes.
    These functions include model saving/loading, logging, storing transitions, selecting actions, etc.

    Args:
        n_env (int): Number of environments used for Multi-processing/parallel learning.
        lr (float): Learning rate.
        gamma (float): Discount factor.
        policy (str) : Policy name or identifier used by the agent.
        eps_start (float): Initial value of epsilon .
        eps_min (float): Minimum (end) value of epsilon to which epsilon will decay.
        eps_dec (int): Number of steps for epsilon to decay to its minimum value.
        eps_dec_exp (bool): Whether to use exponential decay for epsilon.
        nn_conf_func (function): Function to configure the neural network.
        input_dim (tuple): Dimensions of the input data (state space).
        output_dim (int): Dimension of the output data (action space).
        batch_size (int): Size of the mini-batches.
        min_buffer_size (int): Minimum size of the replay memory buffer before training begins.
        buffer_capacity (int): Replay memory buffer capacity.
        target_update_freq (int): Target network update frequency (in steps).
        target_soft_update (bool): # Whether to use target network soft update.
        tau (float): Soft update rate, which controls how quickly the target network's parameters are updated
        save_freq (int): Frequency (in steps) at which the model is saved
        log_freq (int): Frequency (in steps) at which training metrics are logged.
        save_dir (str): Directory where the model is saved.
        log_dir (str): Directory where TensorBoard logs are stored.
        load (bool): Whether to load a pre-trained model and resume training.
        algo (str): Name/identifier of the DQN algorithm.
        gpu (str): identifier of GPU device .

    Abstract Methods:
        replay_memory_buffer (ReplayMemoryNaive | ReplayMemoryPrioritized): The buffer used to store and sample experiences.
        online_network (DeepQNetwork | DuelingDeepQNetwork): The current network.
        target_network (DeepQNetwork | DuelingDeepQNetwork): The target network.


    Methods:
        learn(): Method to be implemented by derived classes to define the learning process.
        transitions_to_tensor(transitions): Converts a list of transitions to PyTorch tensors.
        store_transitions(obses, actions, rews, dones, new_obses, infos): Stores transitions in the replay memory buffer.
        epsilon(): Computes the current value of epsilon.
        choose_actions(obses): Chooses actions based on the current policy and epsilon-greedy strategy.
        update_target_network(force=False): Updates the target network.
        load_model(): Loads a saved model to resume training.
        save_model(): Saves the current model.
        log(): Logs training statistics to TensorBoard.
        info_mean(i): Computes the mean value of a specified metric in the episode info buffer.
    """

    def __init__(
        self,
        policy,  # policy Name/identifier
        nn_conf_func,  # Function that return NN Congig (architecture, loss, and optimizer)
        algo: str,  # Name/identifier of the DQN algorithm.
        input_dim: tuple[int],  # Dimensions of the input data (state space).
        output_dim: int,  # Dimension of the output data (action space).
        n_env: int = HYPER_PARAMS.n_env,  # Number of environments used for Multi-processing/parallel learning
        lr: float = HYPER_PARAMS.lr,  # Learning rate
        gamma: float = HYPER_PARAMS.gamma,  # Discount factor
        eps_start: float = HYPER_PARAMS.eps_start,  # Initial value of epsilon
        eps_min: float = HYPER_PARAMS.eps_min,  # Minimum (end) value of epsilon
        eps_dec: int = HYPER_PARAMS.eps_dec,  # Number of steps for epsilon to decay to its minimum value
        eps_dec_exp: bool = HYPER_PARAMS.eps_dec_exp,  # Whether to use exponential decay for epsilon
        batch_size: int = HYPER_PARAMS.batch_size,  # Size of the mini-batches.
        min_buffer_size: int = HYPER_PARAMS.min_buffer_size,  # Minimum size of the replay memory buffer before training begins.
        buffer_capacity: int = HYPER_PARAMS.buffer_capacity,  # Replay memory buffer capacity.
        target_update_freq: int = HYPER_PARAMS.target_update_freq,  # Target network update frequency (in steps)
        target_soft_update: bool = HYPER_PARAMS.target_soft_update,  #  Whether to use target network soft update.
        tau: float = HYPER_PARAMS.tau,  #  Soft update rate
        save_freq: int = HYPER_PARAMS.save_freq,  # Frequency (in steps) at which the model is saved
        log_freq: int = HYPER_PARAMS.log_freq,  # Frequency (in steps) at which training metrics are logged.
        save_dir: str = HYPER_PARAMS.save_dir,  # Directory where the model is saved.
        log_dir: str = HYPER_PARAMS.log_dir+"train/",  # Directory where TensorBoard logs are stored.
        load: bool = HYPER_PARAMS.load,  # Whether to load a pre-trained model and resume training.
        gpu: str = HYPER_PARAMS.gpu,  # identifier of GPU device
    ):
        # Assertions for the attributes
        assert isinstance(gpu, str), "GPU identifier should be a string"
        assert isinstance(n_env, int) and n_env >= 0, "Number of environments should be a positive integer"
        assert isinstance(lr, float) and lr > 0, "Learning rate should be between 0 and 1"
        assert isinstance(gamma, float) and 0 < gamma <= 1, "Gamma should be between 0 and 1"
        assert isinstance(eps_start, float) and 0 < eps_start <= 1, "Epsilon start should be between 0 and 1"
        assert isinstance(eps_min, float) and 0 <= eps_min < eps_start, (
            "Epsilon min should be a positive float and lower than epsilon start."
        )
        assert isinstance(eps_dec, int) and eps_dec > 0, "Epsilon decay should be a positive integer"
        assert isinstance(eps_dec_exp, bool), "Epsilon exponential decay flag should be a boolean"
        assert isinstance(batch_size, int) and batch_size > 0, "Batch size should be a positive integer"
        assert isinstance(min_buffer_size, int) and min_buffer_size > 0, "Minimum memory size should be a positive integer"
        assert isinstance(buffer_capacity, int) and buffer_capacity >= min_buffer_size, (
            "Maximum memory size should be at least the minimum memory size"
        )
        assert isinstance(target_update_freq, int) and target_update_freq > 0, "Target update frequency should be a positive integer"
        assert isinstance(target_soft_update, bool), "Target soft update flag should be a boolean"
        assert isinstance(tau, float) and 0 < tau <= 1, "Tau should be a positive float between 0 and 1"
        assert isinstance(save_freq, int) and save_freq > 0, "Save frequency should be a positive integer"
        assert isinstance(log_freq, int) and log_freq > 0, "Log frequency should be a positive integer"
        assert isinstance(save_dir, str), "Save directory should be a string"
        assert isinstance(log_dir, str), "Log directory should be a string"
        assert isinstance(load, bool), "Load model flag should be a boolean"
        assert algo in [
            "DQNAgent",
            "DoubleDQNAgent",
            "DuelingDoubleDQNAgent",
            "PerDuelingDoubleDQNAgent",
        ], "Algorithm should be one of the specified types"

        ## Initialize agent hyperparameters
        self.n_env = n_env
        self.lr = lr
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.eps_dec_exp = eps_dec_exp
        self.nn_conf_func = nn_conf_func
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.buffer_capacity = buffer_capacity
        self.target_update_freq = target_update_freq
        self.target_soft_update = target_soft_update
        self.tau = tau
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.load = load
        self.buffer_location = HYPER_PARAMS.buffer_location

        ## Initialize internal state variables
        self.step = 0
        self.resume_step = 0
        self.episode_count: int = 0
        self.ep_info_buffer: deque = deque([], maxlen=HYPER_PARAMS.ep_info_buffer_capacity)

        ## Set up paths for saving models and logs
        self.path = algo[:-5] + "_lr_" + str(lr) + "_" + policy + "_model.pack"  # file name
        self.save_path = save_dir + self.path
        self.save_training_state_path = save_dir + "training_state_" + HYPER_PARAMS.buffer_location + "/" + self.path
        self.summary_writer = SummaryWriter(log_dir + algo[:-5] + "_" + policy + "/")

        self.loss_per_set_iteration = 0

        ## Set device (GPU or CPU) for computation
        self.device = T.device(("cuda:" + gpu) if T.cuda.is_available() else "cpu")
        print(
            "DEVICE",
            "=",
            self.device,
            "" if not T.cuda.is_available() else T.cuda.get_device_name(self.device),
        )

        ## Record the start time for logging
        self.start_time = time.time()

    @abstract_attribute
    def replay_memory_buffer(self):
        pass

    @abstract_attribute
    def online_network(self):
        pass

    @abstract_attribute
    def target_network(self):
        pass

    def learn(self):
        raise NotImplementedError

    def transitions_to_tensor(self, transitions: list[tuple]):
        """
        Convert a list of transitions to tensors.
        A transition is a tuple of (obs, action, reward, done, new_obs).

        Takes as input:
            - transitions: list of transitions/tuples.

        Returns:
            - tuple of tensors (obses_t, actions_t, rews_t, dones_t, new_obses_t).
        """

        # # This function is used to convert transitions sampled from the replay buffer to tensor
        # # It is not used to convert transitions collected directly from agent-environment interaction
        # # because those transitions are stored on replay buffer as np.array

        # Check that transitions is a list
        assert isinstance(transitions, list), "transitions must be a list"

        # Check that each transition in the list is a tuple
        assert all(isinstance(t, Union[tuple, list]) for t in transitions), "Each item in transitions must be a tuple"

        obses_t = T.as_tensor(np.asarray([t[0] for t in transitions]), dtype=T.float32).to(self.device)
        actions_t = T.as_tensor(np.asarray([t[1] for t in transitions]), dtype=T.int64).to(self.device).unsqueeze(-1)
        rews_t = T.as_tensor(np.asarray([t[2] for t in transitions]), dtype=T.float32).to(self.device).unsqueeze(-1)
        dones_t = T.as_tensor(np.asarray([t[3] for t in transitions]), dtype=T.float32).to(self.device).unsqueeze(-1)
        new_obses_t = T.as_tensor(np.asarray([t[4] for t in transitions]), dtype=T.float32).to(self.device)

        return obses_t, actions_t, rews_t, dones_t, new_obses_t

    def store_transitions(
        self,
        obses: np.ndarray,
        actions: list,
        rews: np.ndarray,
        dones: np.ndarray,
        new_obses: np.ndarray,
        infos: Union[tuple, None, list],
    ):
        """
        Store transitions and log episode information when episodes end.

        Args:
            obses (np.ndarray): Array of observations (states).
            actions (list): List of actions taken by the agent.
            rews (np.ndarray): Array of rewards received after taking the actions.
            dones (np.ndarray): Array of boolean values indicating episode termination.
            new_obses (np.ndarray): Array of new observations (next states) after actions.
            infos (Union[tuple, None]): Tuple containing additional episode information,
                                        such as total reward ('r') and length ('l'), or None if not available.

        Side Effects:
            - Appends episode information to `self.ep_info_buffer` when an episode ends.
            - Increments `self.episode_count` by 1 for each completed episode.

        """

        # # Check that all inputs are of expected types
        # assert isinstance(obses, np.ndarray), f"obses must be a numpy array, but got {type(obses).__name__}"
        # assert isinstance(actions, list), f"actions must be a list, but got {type(actions).__name__}"
        # assert isinstance(rews, np.ndarray), f"rews must be a numpy array, but got {type(rews).__name__}"
        # assert isinstance(dones, np.ndarray), f"dones must be a numpy array, but got {type(dones).__name__}"
        # assert isinstance(new_obses, np.ndarray), f"new_obses must be a numpy array, but got {type(new_obses).__name__}"
        # assert isinstance(infos, (tuple, type(None), list)), f"infos must be a tuple or list or None, but got {type(infos).__name__}"
        # # tuple when n_env > 1;  None during buffer initialization; list when n_env = 1

        for i in self.replay_memory_buffer.store_transitions(obses, actions, rews, dones, new_obses):
            # The method self.replay_memory_buffer.store_transitions yields indices of transitions where episodes end.
            if infos:
                # When initializing the replay buffer, infos==None. Once training starts, infos!=None, and logging begins.
                # save episode information () only when the episode ends and infos is not None.
                self.ep_info_buffer.append({"r": infos[i]["r"], "l": infos[i]["l"]})
                self.episode_count += 1  # Increment the episode count by 1.
                infos[i]["episode_num"] = self.episode_count

    def epsilon(self):
        """
        Calculate the current value of epsilon based on the current time-step, adjusting between the start and minimum epsilon values over
        a defined range of time steps (epsilon decay)

        This function provides two decay methods:
            - Exponential decay: Epsilon decreases following an exponential curve, where initially epsilon drops quickly, but the rate of
            decrease slows down as it approaches the minimum value.
            - Linear decay: Epsilon decreases linearly over time. The rate of decrease is constant from start to finish.

        Returns:
            float: The current value of epsilon.
        """

        if self.eps_dec_exp:  #  exponential decay
            return np.exp(
                np.interp(
                    self.step * self.n_env,  # The current time step,  used as interpolation input.
                    [0, self.eps_dec],  # The x-coordinates of the interpolation range.
                    [np.log(self.eps_start), np.log(self.eps_min)],  # the corresponding y-coordinates (log scale).
                )
            )
        else:  # linear decay
            return np.interp(
                self.step * self.n_env,  # The current time step,  interpolation input.
                [0, self.eps_dec],  # The x-coordinates of the interpolation range.
                [self.eps_start, self.eps_min],  # the corresponding y-coordinates (linear scale).
            )

    def choose_actions(self, obses: np.ndarray) -> list:
        """Choose actions based on the current policy (online network) with epsilon-greedy method"""

        # Check that the input is a numpy array
        assert isinstance(obses[0], np.ndarray), f"obses must be a numpy array, but got {type(obses).__name__}."

        # Get actions from the curent policy (online network) based on observations
        actions = self.online_network.actions(obses)

        # Iterate through each action
        for i in range(len(actions)):
            # With probability epsilon, replace the i-th action with a random action
            if random.random() <= self.epsilon():
                actions[i] = random.randint(0, self.output_dim - 1)

        return actions

    def update_target_network(self, force=False):
        """
        Updates the target network's parameters (weights) based on the online network's parameters.

        Args:
            force (bool): If True, apply hard update at each time step.

        This function handles two types of updates:
            1. Hard Update: Replaces the target network's parameters entirely with those of the online network:
                - at specific intervals, or
                - at each time step if 'force' is True.

            2. Soft Update: Gradually blends the online network's parameters into the target network's parameters at each step:
                - Instead of waiting C steps to make an update, the target network is updated in each step, using the formula:
                        - Qtarget = τ * Qcurrent + (1 - τ) * Qtarget
                        - where τ is a small value (e.g., 0.001) that determines the rate of the update
                - For example, with τ=0.001 the new weights for the target network will take
                    - 0.1% of the main network's weights, and
                    - 99.9% of the old target network weights
        """
        # Hard Update: Replaces the target network’s parameters entirely with those of the online network at specific intervals.
        if (not self.target_soft_update and self.step % (self.target_update_freq // self.n_env) == 0) or force:
            self.target_network.load_state_dict(self.online_network.state_dict())

        # Soft Update: Gradually blends the online network’s parameters into the target network’s parameters on #* each step *#.
        elif self.target_soft_update:
            # Iterate through the parameters of both networks
            for target_network_param, online_network_param in zip(self.target_network.parameters(), self.online_network.parameters()):
                # Update target network parameters using the soft update formula
                target_network_param.data.copy_(
                    (self.tau * self.n_env) * online_network_param.data + (1.0 - (self.tau * self.n_env)) * target_network_param.data
                )

    def info_mean(self, i: str):
        """
        Computes the mean of the specified metric from the episode information buffer.

        Parameters:
        - i (str): Key for which the mean is to be calculated (e.g., "r" for reward, "l" for length).

        Returns:
        - float: The mean of the specified metric, or 0.0 if the mean is NaN.
        """

        # Check that the input is of the expected type
        assert isinstance(i, str), "Key metric must be a string"

        # Calculate the mean of the specified metric
        i_mean = np.mean([e[i] for e in self.ep_info_buffer])

        # Return the mean or 0.0 if the mean is NaN
        # If self.ep_info_buffer is empty, i_mean will be NaN
        return i_mean if not math.isnan(i_mean) else 0.0

    def __str__(self):
        return (
            f"‣ NN :\n {Fore.LIGHTGREEN_EX} {self.online_network} {Fore.RESET} \n"
            "‣ DQN Hyperparameters\n"
            f"  • Learning rate (lr): {self.lr}\n"
            f"  • Gamma: {self.gamma}\n"
            f"  • Epsilon start: {self.eps_start}\n"
            f"  • Epsilon min: {self.eps_min}\n"
            f"  • Epsilon decay: {self.eps_dec}\n"
            f"  • Exp decay: {self.eps_dec_exp}\n"
            f"  • Batch size: {self.batch_size}\n"
            f"  • Buffer capacity: {self.buffer_capacity}\n"
            f"  • Buffer min size: {self.min_buffer_size} \n"
            f"  • Target update: {'Soft' if self.target_soft_update else 'Hard'}\n"
            f"  • Update frequency: {self.target_update_freq}\n"
            f"  • tau:{self.tau if self.target_soft_update else ''}\n"
            f"  • Save frequency: {self.save_freq}\n"
            f"  • Save path: {self.save_path}\n"
            f"  • Log frequency: {self.log_freq}\n"
            f"  • Log directory: {self.summary_writer.log_dir}\n"
            f"  • n_env: {self.n_env}\n"
            f"  • Device: {self.device}\n"
            f"  • Resume from step: {self.resume_step}\n"
            f"  • Episode count: {self.episode_count}\n"
            "\n"
            f"  • Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}\n"
        )


class SimpleAgent(Agent):
    """
    Implementation of the Vanilla DQN algorithm.
    """

    def __init__(self, *args, **kwargs):
        super(SimpleAgent, self).__init__(*args, **kwargs)

    def learn(self):
        """
        Implements a single step of the learning process for the Vanilla DQN algorithm.
        """
        # Sample transitions from the replay memory buffer.
        transitions = self.replay_memory_buffer.sample_transitions()

        # print("00000\n", transitions[0], type(transitions[0]))

        # Convert the sampled transitions into tensors for processing.
        obses_t, actions_t, rews_t, dones_t, new_obses_t = self.transitions_to_tensor(transitions)

        # print("11111111111\n", obses_t, actions_t, rews_t, dones_t, new_obses_t)
        # exit(0)

        with T.no_grad():
            # Compute target Q-values using the target network.
            target_q_values = self.target_network(new_obses_t)

            # Get the maximum Q-value for the next state.
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

            # Calculate the targets for the loss function.
            targets = rews_t + (1 - dones_t) * self.gamma * max_target_q_values

        # Get the Q-values from the online network for the current state.
        online_q_values = self.online_network(obses_t)

        # Select the Q-values corresponding to the taken actions.
        action_q_values = T.gather(input=online_q_values, dim=1, index=actions_t)

        # Compute the loss between the predicted and target Q-values
        loss = self.online_network.loss(action_q_values, targets).to(self.device)

        # Perform gradient descent to minimize the loss.
        self.online_network.optimizer.zero_grad()
        loss.backward()  # computes the gradient of the loss with respect to each parameter (weight and bias)
        self.online_network.optimizer.step()  # Update the online network parameters.

        self.loss_per_set_iteration += loss.item()


class DoubleAgent(Agent):
    """
    Implementation of the Double DQN algorithm.
    DoubleAgent extends the SimpleAgent to reduce the overestimation present in Vanilla DQN by using the online network to select actions
    and the target network to evaluate those actions.
    """

    def __init__(self, *args, **kwargs):
        super(DoubleAgent, self).__init__(*args, **kwargs)

    def learn(self):
        """
        Implements a single step of the learning process for the Double DQN algorithms
        """
        # Sample transitions from the replay memory buffer.
        transitions = self.replay_memory_buffer.sample_transitions()

        # Convert the sampled transitions into tensors for processing.
        obses_t, actions_t, rews_t, dones_t, new_obses_t = self.transitions_to_tensor(transitions)

        with T.no_grad():
            # Compute Q-values for the next state using the online network.
            targets_online_q_values = self.online_network(new_obses_t)

            # Get the indices of the actions with the highest Q-values.
            targets_online_best_q_indices = targets_online_q_values.argmax(dim=1, keepdim=True)

            # Compute Q-values for the next state using target network.
            targets_target_q_values = self.target_network(new_obses_t)

            # Select the Q-values corresponding to the taken greedy actions
            targets_selected_q_values = T.gather(
                input=targets_target_q_values,
                dim=1,
                index=targets_online_best_q_indices,
            )

            # Calculate the targets
            targets = rews_t + (1 - dones_t) * self.gamma * targets_selected_q_values

        # Get the Q-values from the online network for the current state.
        online_q_values = self.online_network(obses_t)

        # Select the Q-values corresponding to the taken actions.
        action_q_values = T.gather(input=online_q_values, dim=1, index=actions_t)

        # Compute the loss between the predicted and target Q-values.
        loss = self.online_network.loss(action_q_values, targets).to(self.device)

        # Perform gradient descent to minimize the loss.
        self.online_network.optimizer.zero_grad()
        loss.backward()  # computes the gradient of the loss with respect to each parameter (weight and bias)
        self.online_network.optimizer.step()  # Update the online network parameters.

        self.loss_per_set_iteration += loss.item()


class PerDoubleAgent(Agent):
    """
    Implementation of the Prioritized Double DQN algorithm.
    - PerDoubleAgent extends the DoubleAgent by incorporating Prioritized Experience Replay (PER).
    - Prioritized Experience Replay samples transitions with higher temporal-difference (TD) errors more frequently, helping the agent
      to learn from the most informative experiences.
    """

    def __init__(self, *args, **kwargs):
        super(PerDoubleAgent, self).__init__(*args, **kwargs)

    def learn(self):
        """
        Implements a single step of the learning process for the Prioritized Double DQN algorithm.
        """
        # Sample transitions with importance sampling weights and tree indices.
        is_weights, tree_indices, transitions = self.replay_memory_buffer.sample_transitions(self.step * self.n_env)

        # Convert the importance sampling weights to tensors and add an extra dimension.
        is_weights_t = T.as_tensor(np.asarray(is_weights), dtype=T.float32).to(self.device).unsqueeze(-1)

        # Convert the sampled transitions into tensors for processing.
        obses_t, actions_t, rews_t, dones_t, new_obses_t = self.transitions_to_tensor(transitions)

        with T.no_grad():
            # Compute Q-values for the next state using the online network.
            targets_online_q_values = self.online_network(new_obses_t)

            # Get the indices of the actions with the highest Q-values.
            targets_online_best_q_indices = targets_online_q_values.argmax(dim=1, keepdim=True)

            # Compute Q-values for the next state using target network.
            targets_target_q_values = self.target_network(new_obses_t)

            # Select the Q-values corresponding to the taken greedy actions
            targets_selected_q_values = T.gather(
                input=targets_target_q_values,
                dim=1,
                index=targets_online_best_q_indices,
            )

            # Calculate the targets for the loss function.
            targets = rews_t + (1 - dones_t) * self.gamma * targets_selected_q_values

        # Get the Q-values from the online network for the current state.
        online_q_values = self.online_network(obses_t)

        # Select the Q-values corresponding to the taken actions.
        action_q_values = T.gather(input=online_q_values, dim=1, index=actions_t)

        with T.no_grad():
            # Compute the absolute TD errors
            abs_td_errors_np = T.abs(targets - action_q_values).detach().cpu().numpy()

            # update the priorities in the replay buffer.
            self.replay_memory_buffer.update_batch_priorities(tree_indices, abs_td_errors_np)

        # Compute the weighted loss using importance sampling weights.
        loss = T.mean(is_weights_t * self.online_network.loss(action_q_values, targets)).to(self.device)

        # Perform gradient descent to minimize the loss.
        self.online_network.optimizer.zero_grad()
        loss.backward()  # computes the gradient of the loss with respect to each parameter (weight and bias)
        self.online_network.optimizer.step()  # Update the online network parameters.

        return loss


class DQNAgent(SimpleAgent):
    """
    Initialize the neural network and replay memory required for the Vanilla DQN algorithm.
    """

    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

        # Initialize the replay memory buffer for storing transitions.
        self.replay_memory_buffer = ReplayMemoryNaive(
            self.buffer_capacity, self.batch_size, state_shape=self.input_dim.shape, model=self.path
        )

        # Initialize the online and target networks.
        self.online_network = DeepQNetwork(self.device, self.lr, self.nn_conf_func, self.input_dim, self.output_dim)
        self.target_network = DeepQNetwork(self.device, self.lr, self.nn_conf_func, self.input_dim, self.output_dim)

        # Set the target network's parameters equal to the online network's parameters (using hard update).
        self.update_target_network(force=True)


class DoubleDQNAgent(DoubleAgent):
    """
    Initializes the neural network and replay memory required for the Double DQN algorithm.
    """

    def __init__(self, *args, **kwargs):
        super(DoubleDQNAgent, self).__init__(*args, **kwargs)

        # Initialize the replay memory buffer for storing transitions.
        self.replay_memory_buffer = ReplayMemoryNaive(
            self.buffer_capacity, self.batch_size, state_shape=self.input_dim.shape, model=self.path
        )

        # Initialize the online and target networks.
        self.online_network = DeepQNetwork(self.device, self.lr, self.nn_conf_func, self.input_dim, self.output_dim)
        self.target_network = DeepQNetwork(self.device, self.lr, self.nn_conf_func, self.input_dim, self.output_dim)
        # Set the target network's parameters equal to the online network's parameters (using hard update).
        self.update_target_network(force=True)


class DuelingDoubleDQNAgent(DoubleAgent):
    """
    Initialize the neural network and replay memory required for the Dueling Double DQN algorithm.
    """

    def __init__(self, *args, **kwargs):
        super(DuelingDoubleDQNAgent, self).__init__(*args, **kwargs)

        # Initialize the replay memory buffer for storing transitions.
        self.replay_memory_buffer = ReplayMemoryNaive(
            self.buffer_capacity, self.batch_size, state_shape=self.input_dim.shape, model=self.path
        )

        # Initialize the online and target networks.
        self.online_network = DuelingDeepQNetwork(self.device, self.lr, self.nn_conf_func, self.input_dim, self.output_dim)
        self.target_network = DuelingDeepQNetwork(self.device, self.lr, self.nn_conf_func, self.input_dim, self.output_dim)
        # Set the target network's parameters equal to the online network's parameters (using hard update).
        self.update_target_network(force=True)


class PerDuelingDoubleDQNAgent(PerDoubleAgent):
    """
    Initialize the neural network and replay memory required for the Prioritized Dueling Double DQN algorithm.
    """

    def __init__(self, *args, **kwargs):
        super(PerDuelingDoubleDQNAgent, self).__init__(*args, **kwargs)

        # Initialize the prioritized replay memory buffer for storing transitions.
        self.replay_memory_buffer = ReplayMemoryPrioritized(
            self.buffer_capacity, self.batch_size, self.eps_dec, state_shape=self.input_dim.shape, model=self.path
        )

        # Initialize the online and target networks.
        self.online_network = DuelingDeepQNetwork(
            self.device,
            self.lr,
            self.nn_conf_func,
            self.input_dim,
            self.output_dim,
            reduction="none",
        )
        self.target_network = DuelingDeepQNetwork(
            self.device,
            self.lr,
            self.nn_conf_func,
            self.input_dim,
            self.output_dim,
            reduction="none",
        )
        # Set the target network's parameters equal to the online network's parameters (using hard update).
        self.update_target_network(force=True)
