from .utils.custom_abc_meta import CustomABCMeta, abstract_attribute
from .network import DeepQNetwork, DuelingDeepQNetwork
from .dqn_config import HYPER_PARAMS
from .agent_off_mixin import AgentOffMixin

import os
from colorama import Fore
import time
import torch as T
from torch.utils.tensorboard.writer import SummaryWriter


# // if HYPER_PARAMS.buffer_location == "disk":
# //     from .replay_memory_disk import ReplayMemoryNaive, ReplayMemoryPrioritized
# // else:
# //     from .replay_memory_ram import ReplayMemoryNaive, ReplayMemoryPrioritized


"""
This module implements different Deep Q-Network (DQN) algorithms.
    - The `Agent` class defines the common attributes and methods required by the DQN algorithms, such as model saving/loading, logging,
      storing transitions, action selection, etc.
    - The `SimpleAgent`, `DoubleAgent`, and `PerDoubleAgent` classes inherit from the `Agent` class and define the `learn()` function, 
      which implements a single step of the learning process for a given DQN algorithm.
    - The `DQNAgent`, `DoubleDQNAgent`, `DuelingDoubleDQNAgent`, and `PerDuelingDoubleDQNAgent` classes initialize the neural networks 
      and replay memory required for each algorithm.
"""


class AgentOff(AgentOffMixin, metaclass=CustomABCMeta):
    """
    Abstract base class for Offline DQN agent. It must be extended by specific DQN agent implementations.
    Define the common attributes and methods required by the DQN algorithmes.
    These functions include model saving/loading, logging, storing transitions, selecting actions, etc.

    Args:
        lr (float): Learning rate.
        gamma (float): Discount factor.
        gpu (str): identifier of GPU device .
        csv_dir_name (str) :  Name of the directory containing the agent training data (CSV files).
        policy (str) : Policy name or identifier used by the agent.
        algo (str): Name/identifier of the DQN algorithm.
        input_dim (tuple): Shape of the input data (state space).
        output_dim (int): Shape of the output data (action space).
        batch_size (int) : Number of samples per training batch.
        target_update_freq (int) : Frequency (in training steps) for updating the target network.
        target_soft_update (bool) : Whether target network parameters are updated using soft updates.
        tau (float) : Soft update rate used when `target_soft_update=True`, it controls how quickly the target network's parameters are updated
        nn_conf_func (function): Function to configure the neural network (architecture, loss function, and optimizer).

        save_freq (int) : Frequency (in training steps) at which model checkpoints are saved.
        log_freq (int) : Frequency (in training steps) at which training metrics are logged.
        save_dir (str) : Directory path for saving model checkpoints.
        log_dir (str):  Directory path for storing TensorBoard logs.
        load (bool) : Whether to load a pre-trained model and checkpoint and resume training.

        shuffle (bool): Whether to shuffle training data before batching.


    Abstract Methods:
        online_network (DeepQNetwork | DuelingDeepQNetwork): The current network.
        target_network (DeepQNetwork | DuelingDeepQNetwork): The target network.


    Methods:
        learn(): Method to be implemented by derived classes to define the learning process.
        transitions_to_tensor(transitions): Converts a list of transitions to PyTorch tensors.
        update_target_network(force=False): Updates the target network.
        load_model(): Loads a saved model to resume training.
        save_model(): Saves the current model.
        log(): Logs training statistics to TensorBoard.
        info_mean(i): Computes the mean value of a specified metric in the epoch info buffer.
    """

    def __init__(
        self,
        csv_dir_name: str,  # Directory name where the agent data is located
        policy,  # policy Name/identifier
        nn_conf_func,  # Function that return NN Congig (architecture, loss, and optimizer)
        algo: str,  # Name/identifier of the DQN algorithm.
        input_dim: tuple[int],  # Dimensions of the input data (state space).
        output_dim: int,  # Dimension of the output data (action space).
        lr: float = HYPER_PARAMS.lr,  # Learning rate
        gamma: float = HYPER_PARAMS.gamma,  # Discount factor
        batch_size: int = HYPER_PARAMS.batch_size,  # Size of the mini-batches.
        target_update_freq: int = HYPER_PARAMS.target_update_freq,  # Target network update frequency (in steps)
        target_soft_update: bool = HYPER_PARAMS.target_soft_update,  #  Whether to use target network soft update.
        tau: float = HYPER_PARAMS.tau,  #  Soft update rate
        save_freq: int = HYPER_PARAMS.save_freq,  # Frequency (in steps) at which the model is saved
        log_freq: int = HYPER_PARAMS.log_freq,  # Frequency (in steps) at which training metrics are logged.
        save_dir: str = HYPER_PARAMS.save_dir,  # Directory where the model is saved.
        log_dir: str = HYPER_PARAMS.log_dir,  # Directory where TensorBoard logs are stored.
        load: bool = HYPER_PARAMS.load,  # Whether to load a pre-trained model and resume training.
        gpu: str = HYPER_PARAMS.gpu,  # identifier of GPU device
        shuffle: bool = False,  # Whether to shuffle agent data or not
    ):
        # Assertions for the attributes
        assert isinstance(gpu, str), "GPU identifier should be a string"
        assert isinstance(lr, float) and lr > 0, "Learning rate should be between 0 and 1"
        assert isinstance(gamma, float) and 0 < gamma <= 1, "Gamma should be between 0 and 1"
        assert isinstance(batch_size, int) and batch_size > 0, "Batch size should be a positive integer"
        assert isinstance(target_update_freq, int) and target_update_freq > 0, "Target update frequency should be a positive integer"
        assert isinstance(target_soft_update, bool), "Target soft update flag should be a boolean"
        assert isinstance(tau, float) and 0 < tau <= 1, "Tau should be a positive float between 0 and 1"
        assert isinstance(save_freq, int) and save_freq > 0, "Save frequency should be a positive integer"
        assert isinstance(log_freq, int) and log_freq > 0, "Log frequency should be a positive integer"
        assert isinstance(save_dir, str), "Save directory should be a string"
        assert isinstance(log_dir, str), "Log directory should be a string"
        assert isinstance(load, bool), "Load model flag should be a boolean"
        assert algo in [
            "DQNAgentOff",
            "DoubleDQNAgentOff",
            "DuelingDoubleDQNAgentOff",
        ], "Algorithm should be one of the specified types"

        ## Initialize agent hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.nn_conf_func = nn_conf_func
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.target_soft_update = target_soft_update
        self.tau = tau
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.load = load
        self.input_dim = input_dim
        self.output_dim = output_dim

        ## Set up paths for saving models and logs
        self.path = algo[:-8] + "_lr_" + str(lr) + "_" + policy + "_model.pack"  # file name
        self.save_path = save_dir + self.path
        self.save_training_state_path = save_dir + "training_state_" + HYPER_PARAMS.buffer_location + "/" + self.path
        self.summary_writer = SummaryWriter(log_dir + algo[:-8] + "_" + policy + "/")

        ## Set device (GPU or CPU) for computation
        self.device = T.device(("cuda:" + gpu) if T.cuda.is_available() else "cpu")
        print(
            "DEVICE",
            "=",
            self.device,
            "" if not T.cuda.is_available() else T.cuda.get_device_name(self.device),
        )

        self.start_time = time.time()  # Record the start time for logging
        self.resume_iteration = 0  # Iteration from which to resume training
        self.iteration = 0  # training iteration counter
        self.epoch = 0  # training epoch counter

        self.csv_dir_path = os.path.join(HYPER_PARAMS.agent_data_dir, csv_dir_name)  # set the path to the agent data
        self.shuffle = shuffle
        self.size_agent_data = self.loading_agent_data(self.resume_iteration)

        # Initialize variables for tracking loss and training iterations
        self.loss_per_epoch = 0
        self.loss_per_set_iteration = 0
        self.iter_per_epoch = int(self.size_agent_data / self.batch_size)  # Number of data samples divided by batch size

    @abstract_attribute
    def online_network(self):
        pass

    @abstract_attribute
    def target_network(self):
        pass

    def learn(self):
        raise NotImplementedError

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
                target_network_param.data.copy_(self.tau * online_network_param.data + (1.0 - self.tau) * target_network_param.data)

    def __str__(self):
        return (
            f"‣ NN :\n {Fore.LIGHTGREEN_EX} {self.online_network} {Fore.RESET} \n"
            "‣ DQN Hyperparameters\n"
            f"  • Learning rate (lr): {self.lr}\n"
            f"  • Gamma: {self.gamma}\n"
            f"  • Batch size: {self.batch_size}\n"
            f"  • Target update: {'Soft' if self.target_soft_update else 'Hard'}\n"
            f"  • Update frequency: {self.target_update_freq}\n"
            f"  • tau:{self.tau if self.target_soft_update else ''}\n"
            f"  • Save frequency: {self.save_freq}\n"
            f"  • Save path: {self.save_path}\n"
            f"  • Log frequency: {self.log_freq}\n"
            f"  • Log directory: {self.summary_writer.log_dir}\n"
            f"  • Device: {self.device}\n"
            f"  • Resume from iteration: {self.resume_iteration}\n"
            "\n"
            f"  • Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}\n"
        )


class SimpleAgentOff(AgentOff):
    """
    Implementation of the Vanilla DQN algorithm.
    """

    def __init__(self, *args, **kwargs):
        super(SimpleAgentOff, self).__init__(*args, **kwargs)

    def learn(self, transitions):
        """
        Implements a single step of the learning process for the Vanilla DQN algorithm.
        """

        obses_t, actions_t, rews_t, dones_t, new_obses_t = transitions

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

        # //self.loss_info_buffer.append(loss.item())
        self.loss_per_epoch += loss.item()
        self.loss_per_set_iteration += loss.item()


class DoubleAgentOff(AgentOff):
    """
    Implementation of the Double DQN algorithm.
    DoubleAgent extends the SimpleAgent to reduce the overestimation present in Vanilla DQN by using the online network to select actions
    and the target network to evaluate those actions.
    """

    def __init__(self, *args, **kwargs):
        super(DoubleAgentOff, self).__init__(*args, **kwargs)

    def learn(self, transitions):
        """
        Implements a single step of the learning process for the Double DQN algorithm.
        """

        obses_t, actions_t, rews_t, dones_t, new_obses_t = transitions

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

        self.loss_per_epoch += loss.item()
        self.loss_per_set_iteration += loss.item()


class DQNAgentOff(SimpleAgentOff):
    """
    Initialize the neural network and replay memory required for the Vanilla DQN algorithm.
    """

    def __init__(self, *args, **kwargs):
        super(DQNAgentOff, self).__init__(*args, **kwargs)

        # Initialize the online and target networks.
        self.online_network = DeepQNetwork(self.device, self.lr, self.nn_conf_func, self.input_dim, self.output_dim)
        self.target_network = DeepQNetwork(self.device, self.lr, self.nn_conf_func, self.input_dim, self.output_dim)

        # Set the target network's parameters equal to the online network's parameters (using hard update).
        self.update_target_network(force=True)


class DoubleDQNAgentOff(DoubleAgentOff):
    """
    Initializes the neural network and replay memory required for the Double DQN algorithm.
    """

    def __init__(self, *args, **kwargs):
        super(DoubleDQNAgentOff, self).__init__(*args, **kwargs)

        # Initialize the online and target networks.
        self.online_network = DeepQNetwork(self.device, self.lr, self.nn_conf_func, self.input_dim, self.output_dim)
        self.target_network = DeepQNetwork(self.device, self.lr, self.nn_conf_func, self.input_dim, self.output_dim)
        # Set the target network's parameters equal to the online network's parameters (using hard update).
        self.update_target_network(force=True)


class DuelingDoubleDQNAgentOff(DoubleAgentOff):
    """
    Initialize the neural network and replay memory required for the Dueling Double DQN algorithm.
    """

    def __init__(self, *args, **kwargs):
        super(DuelingDoubleDQNAgentOff, self).__init__(*args, **kwargs)

        # Initialize the online and target networks.
        self.online_network = DuelingDeepQNetwork(self.device, self.lr, self.nn_conf_func, self.input_dim, self.output_dim)
        self.target_network = DuelingDeepQNetwork(self.device, self.lr, self.nn_conf_func, self.input_dim, self.output_dim)
        # Set the target network's parameters equal to the online network's parameters (using hard update).
        self.update_target_network(force=True)
