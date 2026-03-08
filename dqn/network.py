import os

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from gymnasium.spaces import Space
import numpy as np

import msgpack
import msgpack_numpy as m



import copy


m.patch()  # automatically force all msgpack serialization and deserialization routines


class Network(nn.Module):
    """
    Base class for defining neural network architectures.
    """

    def __init__(self, device: T.device, nn_conf_func: callable, input_dim: Space):
        """
        Initialize the Network class.

        Args:
            device (torch.device): The device (CPU or GPU) to which the network will be moved.
            nn_conf_func (function): A function that returns the neural network,
                                     output dimension, optimizer function, and loss function.
            input_dim (Space): The dimension of the input privided to the network.
        """
        super(Network, self).__init__()

        # Check that inputs are of expected types
        assert isinstance(device, T.device), f"Expected 'device' to be torch.device, but got {type(device).__name__}."
        assert callable(nn_conf_func), f"Expected 'nn_conf_func' to be callable, but got {type(nn_conf_func).__name__}."
        assert isinstance(input_dim, Space), f"Expected 'input_dim' to be an int, but got {type(input_dim).__name__}."

        # Initialize the network, output dimension, optimizer, and loss function using nn_conf_func
        self.net, self.fc_out_dim, optim_func, loss_func = nn_conf_func(input_dim)
        self.optim_func = lambda params, lr: optim_func(params, lr=lr)
        self.loss_func = lambda reduction: loss_func(reduction=reduction)

        self.device = device

    def forward(self, s: T.Tensor):
        """
        Forward pass through the network.
        """
        raise NotImplementedError

    def actions(self, obses: np.ndarray):
        """
        Select actions based on the current observations.
        """
        raise NotImplementedError


    def load(self, load_path: str):
        """
        Load the model parameters for testing #//and training progress from a file.
        #//This method deserializes the model parameters and training statistics from a file that was previously saved using the `save` method.

        Args:
            load_path (str): Path to the file from which to load the model and progress data.

        """

        # Check if the file exists, raise an error if it doesn't
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)

        with open(load_path, "rb") as f:
            params_dict: dict = msgpack.loads(f.read())  # Use msgpack to deserialize the data from the file

        # Convert the loaded parameters to tensors and move them to the correct device
        parameters = {k: T.as_tensor(v.copy(), device=self.device) for k, v in params_dict["online_network"].items()}
        # Load the parameters into the network
        self.load_state_dict(parameters)

        # Return the training progress statistics
        # return params_dict["step"], params_dict["episode_count"], params_dict["rew_mean"], params_dict["len_mean"]

class BCNetwork(Network):
    """
    
    """

    def __init__(
        self, device: T.device, lr: float, nn_conf_func: callable, input_dim: Space, output_dim: np.int64, reduction: str = "mean"
    ):
        # Check that input is of expected types
        assert isinstance(lr, float), f"Expected 'lr' to be a float, but got {type(lr).__name__}."

        """
        Initialize the BCNetwork class.

        Args:
            ....
            lr (float): Learning rate for the optimizer. how much the optimizer should change parameters at each step
            ....
            output_dim (np.int64): The dimension of the output (number of actions).
            reduction (str): The reduction method for the loss function.
                             Specifies how the computed loss should be aggregated across a batch of data
        """
        super(BCNetwork, self).__init__(device, nn_conf_func, input_dim)

        # Define the output layer for action values
        self.fc_out = nn.Linear(self.fc_out_dim, output_dim)
        self.fc_out_bc = nn.Linear(self.fc_out_dim, output_dim)  #! added 
        self.net_bc=copy.deepcopy(self.net)  # same architecture, independent weights

        # Set up optimizer and loss function
        self.optimizer = self.optim_func(self.parameters(), lr=lr)
        self.loss = self.loss_func(reduction=reduction)
       

        self.to(self.device)

    def forward(self, s: T.Tensor):
        """
        Forward pass through the network.

        Args:
            s (torch.Tensor): batch of observation.

        Returns:
            torch.Tensor: Output tensor representing Q-values for each action.
        """

        # # Check that input is of expected types
        # assert isinstance(s, T.Tensor), f"Expected 's' to be torch.Tensor , but got {type(s).__name__}."

        net = self.net(s)
        a = self.fc_out(net)
        net_bc=self.net_bc(s)
        i = self.fc_out_bc(net_bc)

        return a, F.log_softmax(i, dim=1), i
    
    def actions(self, obses: np.ndarray) -> list:
        """
        return the best actions based on the current observations.

        Args:
            obses (numpy.ndarray): List of observations, number of observation depend on the n_env.

        Returns:
            list: List of greedy actions.
        """

        # Check that input is of expected types
        assert isinstance(obses, np.ndarray), f"Expected 'obses' to be numpy.ndarray , but got {type(obses).__name__}."

        # Convert observations to a PyTorch tensor of type float32 and move it to the appropriate device
        obses_t = T.as_tensor(obses, dtype=T.float32).to(self.device)

        # compute the q_values of each possible actions
        q_values,_,_ = self(obses_t)

        # get greedy action for each observation
        max_q_indices = T.argmax(q_values, dim=1)

        actions = max_q_indices.detach().tolist()

        return actions

class DeepQNetwork(Network):
    """
    Deep Q-Network (DQN)
    """

    def __init__(
        self, device: T.device, lr: float, nn_conf_func: callable, input_dim: Space, output_dim: np.int64, reduction: str = "mean"
    ):
        # Check that input is of expected types
        assert isinstance(lr, float), f"Expected 'lr' to be a float, but got {type(lr).__name__}."

        """
        Initialize the DeepQNetwork class.

        Args:
            ....
            lr (float): Learning rate for the optimizer. how much the optimizer should change parameters at each step
            ....
            output_dim (np.int64): The dimension of the output (number of actions).
            reduction (str): The reduction method for the loss function.
                             Specifies how the computed loss should be aggregated across a batch of data
        """
        super(DeepQNetwork, self).__init__(device, nn_conf_func, input_dim)

        # Define the output layer for action values
        self.fc_out = nn.Linear(self.fc_out_dim, output_dim)

        # Set up optimizer and loss function
        self.optimizer = self.optim_func(self.parameters(), lr=lr)
        self.loss = self.loss_func(reduction=reduction)

        self.to(self.device)

    def forward(self, s: T.Tensor):
        """
        Forward pass through the network.

        Args:
            s (torch.Tensor): batch of observation.

        Returns:
            torch.Tensor: Output tensor representing Q-values for each action.
        """

        # # Check that input is of expected types
        # assert isinstance(s, T.Tensor), f"Expected 's' to be torch.Tensor , but got {type(s).__name__}."

        net = self.net(s)
        a = self.fc_out(net)

        return a

    def actions(self, obses: np.ndarray) -> list:
        """
        return the best actions based on the current observations.

        Args:
            obses (numpy.ndarray): List of observations, number of observation depend on the n_env.

        Returns:
            list: List of greedy actions.
        """

        # Check that input is of expected types
        assert isinstance(obses, np.ndarray), f"Expected 'obses' to be numpy.ndarray , but got {type(obses).__name__}."

        # Convert observations to a PyTorch tensor of type float32 and move it to the appropriate device
        obses_t = T.as_tensor(obses, dtype=T.float32).to(self.device)

        # compute the q_values of each possible actions
        q_values = self(obses_t)

        # get greedy action for each observation
        max_q_indices = T.argmax(q_values, dim=1)

        actions = max_q_indices.detach().tolist()

        return actions


class DuelingDeepQNetwork(Network):
    """
    Dueling Deep Q-Network for reinforcement learning.
    """

    def __init__(
        self, device: T.device, lr: float, nn_conf_func: callable, input_dim: Space, output_dim: np.int64, reduction: str = "mean"
    ):
        """
        Initialize the DuelingDeepQNetwork class.

        Args:
            device (torch.device): The device (CPU or GPU) to which the network will be moved.
            lr (float): Learning rate for the optimizer.
            nn_conf_func (function): A function that returns the neural network,
                                     output dimension, optimizer function, and loss function.
            input_dim (Space): The dimension of the input to the network.
            output_dim (int): The dimension of the output (number of actions).
            reduction (str): The reduction method for the loss function.
        """
        super(DuelingDeepQNetwork, self).__init__(device, nn_conf_func, input_dim)

        # Define the value and advantage layer
        # Define fully connected layer for computing state value function
        self.fc_val = nn.Linear(self.fc_out_dim, 1)

        # Define fully connected layer for computing advantage value function
        self.fc_adv = nn.Linear(self.fc_out_dim, output_dim)

        # Define a lambda function to combine value and advantage functions.
        self.aggregate_layer = lambda val, adv: T.add(val, (adv - adv.mean(dim=1, keepdim=True)))

        # Set up optimizer and loss function
        self.optimizer = self.optim_func(self.parameters(), lr=lr)
        self.loss = self.loss_func(reduction=reduction)

        self.to(self.device)

    def forward(self, s: T.Tensor):
        """
        Forward pass through the network.

        Args:
            s (torch.Tensor): batch of observation.

        Returns:
            torch.Tensor: Output tensor representing aggregated Q-values.
        """

        # # Check that input is of expected types
        # assert isinstance(s, T.Tensor), f"Expected 's' to be torch.Tensor , but got {type(s).__name__}."

        # Process the batch of states through the shared network
        net = self.net(s)

        # Compute the value for each state in the batch
        val = self.fc_val(net)

        # Compute advantages for all actions for each state in the batch
        adv = self.fc_adv(net)

        # Aggregate the value and advantage to get the final Q-values
        agg = self.aggregate_layer(val, adv)

        return agg

    def value(self, s: T.Tensor) -> T.Tensor:
        """
        Compute the state value V(s)

        Args:
            s (torch.Tensor): batch of observation.

        Returns:
            torch.Tensor: Value function output.
        """

        # # Check that input is of expected types
        # assert isinstance(s, T.Tensor), f"Expected 's' to be torch.Tensor , but got {type(s).__name__}."

        net = self.net(s)  #  processes the batch of states through the shared network
        val = self.fc_val(net)  # computes the value for each state in the batch

        return val

    def advantages(self, s: T.Tensor) -> T.Tensor:
        """
        Compute the advantage values A(s,a) for all actions

        Args:
            s (torch.Tensor): batch of observation.

        Returns:
            torch.Tensor: Advantage function output.
        """

        # # Check that input is of expected types
        # assert isinstance(s, T.Tensor), f"Expected 's' to be torch.Tensor , but got {type(s).__name__}."

        net = self.net(s)  #  processes the batch of states through the shared network
        adv = self.fc_adv(net)  # computes advantages for all actions for each state in the batch

        return adv

    def actions(self, obses: np.ndarray) -> list:
        """
        return the best actions based on the current observations.

        Args:
            obses (numpy.ndarray): List of observations, number of observation depend on the n_env.

        Returns:
            list: List of greedy actions.
        """
        # Convert observations to a PyTorch tensor of type float32 and move it to the appropriate device
        obses_t = T.as_tensor(obses, dtype=T.float32).to(self.device)

        # Compute advantage q-values A(s,a)
        adv_q_values = self.advantages(obses_t)

        # Choose actions with the highest advantage values
        max_adv_q_indices = T.argmax(adv_q_values, dim=1)

        actions = max_adv_q_indices.detach().tolist()

        return actions
