import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Space
from dataclasses import dataclass

import torch
import os

# from rl_env.custom_env.env_config import EnvConfig as EC
from dqn.utils.depthwise_separable_conv2d import DepthwiseSeparableConv2d


@dataclass
class HYPER_PARAMS:
    ## Comun hyperparameter
    action_n: int = 3
    observation_n: int = (17,)
    gpu: str = "0"  # identifier of GPU device
    lr: float = 1e-04  # Learning rate
    gamma: float = 0.99  # Discount factor
    batch_size: int = 64  # 32 # Size of the mini-batches
    target_update_freq: int = 1000  # Target network update frequency (in steps)
    target_soft_update: bool = True  # Whether to use target network soft update
    tau: float = 1e-03  # Target network soft update rate, which controls how quickly the target network's parameters are updated
    load: bool = True  # Whether to load a pre-trained model and resume training.
    algo: str = "DoubleDQNAgent"  # Name/identifier of the DQN algorithm
    # [DQNAgent,  DoubleDQNAgent, DuelingDoubleDQNAgent, PerDuelingDoubleDQNAgent]
    log_freq: int = 1000  # Frequency (in steps/iteration) at which training metrics are logged.
    save_freq: int = 1000  # Frequency (in steps/iterations) at which the model is saved
    save_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "save/")  # Directory where the model is saved.
    agent_data_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_data", "train")  #

    ## ON DQN hyperparametres
    n_env: int = 4  # Number of environments used for Multi-processing/parallel learning
    eps_start: float = 1.0  # Initial value of epsilon # DQN Paper
    eps_min: float = 0.01  # Minimum (end) value of epsilon # DQN Paper
    eps_dec: int = int(159e5)  # int(8e6)  # Number of steps for epsilon to decay to its minimum value
    eps_dec_exp: bool = True  # Whether to use exponential decay for epsilon
    min_buffer_size: int = 50000  # Replay memory buffer min size  # DQN Paper
    buffer_capacity: int = 1000000  # Replay memory buffer capacity  # DQN Paper
    ep_info_buffer_capacity: int = 100  # Maximum number of episodes to retain information in the buffer
    max_total_steps: int = int(16e6)  # int(8e6)  # Max total training steps if > 0, else inf training
    buffer_location: str = "disk"
    log_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs/train/")  # Directory where TensorBoard logs are stored.
    # log_episode_metrics_dir = os.path.join(
    #     os.path.dirname(os.path.dirname(__file__)), "logs/episode_metrics/"
    # )  # Directory where TensorBoard logs for episode_metrics are stored

    # ## OFF DQN Hyperparameters
    nb_total_iteration: int = int(max_total_steps / n_env) #- int(min_buffer_size / n_env)  # ? added

    ## BCQ Hyperparameters
    BCQ_threshold=0.3

    ##########################################################
    ############ Assertions for the attributes ###############
    assert isinstance(gpu, str), "GPU identifier should be a string"
    assert isinstance(n_env, int) and n_env > 0, "Number of environments should be a positive integer"
    assert isinstance(lr, float) and lr > 0, "Learning rate should be between 0 and 1"
    assert isinstance(gamma, float) and 0 < gamma <= 1, "Gamma should be between 0 and 1"
    assert isinstance(eps_start, float) and 0 < eps_start <= 1, "Epsilon start should be between 0 and 1"
    assert isinstance(eps_min, float) and 0 <= eps_min < eps_start, "Epsilon min should be a positive float and lower than epsilon start."
    assert isinstance(eps_dec, int) and eps_dec > 0, "Epsilon decay should be a positive integer"
    assert isinstance(eps_dec_exp, bool), "Epsilon exponential decay flag should be a boolean"
    assert isinstance(batch_size, int) and batch_size > 0, "Batch size should be a positive integer"
    assert isinstance(min_buffer_size, int) and min_buffer_size >= batch_size, (
        "Minimum memory size should be greater than or equal to batch size"
    )
    assert isinstance(buffer_capacity, int) and buffer_capacity >= min_buffer_size, (
        "Maximum memory size should be at least the minimum memory size"
    )
    assert isinstance(target_update_freq, int) and target_update_freq > 0, "Target update frequency should be a positive integer"
    assert isinstance(target_soft_update, bool), "Target soft update flag should be a boolean"
    assert isinstance(tau, float) and 0 < tau <= 1, "Tau should be a positive float between 0 and 1"
    assert isinstance(save_freq, int) and save_freq > 0, "Save frequency should be a positive integer"
    assert isinstance(log_freq, int) and log_freq > 0 and log_freq == save_freq, (
        "Log frequency should be a positive integer and log_freq==save_freq"
    )
    assert isinstance(save_dir, str), "Save directory should be a string"
    assert isinstance(log_dir, str), "Log directory should be a string"
    assert isinstance(load, bool), "Load model flag should be a boolean"
    assert isinstance(max_total_steps, int) and max_total_steps >= 0, "Max total steps should be a non-negative integer"
    assert algo in [
        "DQNAgent",
        "DoubleDQNAgent",
        "DuelingDoubleDQNAgent",
        "PerDuelingDoubleDQNAgent",
    ], "Algorithm should be one of the specified types"
    assert buffer_location in ["disk", "ram"]


def network_config(input_dim: Space):
    hidden_dims = (64, 128, 128, 64)
    # activation function
    activation = nn.Tanh()
    # Construct the neural network architecture
    net = nn.Sequential(
        nn.Linear(input_dim.shape[0], hidden_dims[0]),
        activation,
        nn.Linear(hidden_dims[0], hidden_dims[1]),
        activation,
        nn.Linear(hidden_dims[1], hidden_dims[2]),
        activation,
        nn.Linear(hidden_dims[2], hidden_dims[3]),
        activation,
    )

    # The optimizer function
    optim_func = optim.Adam

    # The loss function
    loss_func = nn.SmoothL1Loss

    return net, hidden_dims[3], optim_func, loss_func
