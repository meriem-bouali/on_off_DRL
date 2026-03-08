import os
import msgpack
import msgpack_numpy as m
import numpy as np
import torch as T
from collections import deque
from .dqn_config import HYPER_PARAMS
from colorama import Fore
from datetime import timedelta
import time
import torch

m.patch()  # automatically force all msgpack serialization and deserialization routines


class AgentMixin:
    def save_training_state(self):
        """
        Save the training state in pack file:
            - online and  target network parameters
            - replay buffer
            - episode info buffer
            - step count and episode count
        """
        print(Fore.LIGHTCYAN_EX, "\n Saving training state...", Fore.RESET)

        optimizer_state = self.online_network.optimizer.state_dict()
        params_dict = {
            "online_network": {
                k: v.detach().cpu().numpy() for k, v in self.online_network.state_dict().items()
            },  # Save online network parameters
            "target_network": {
                k: v.detach().cpu().numpy() for k, v in self.target_network.state_dict().items()
            },  # Save target network parameters
            "step": self.step,  # Current step number
            "episode_count": self.episode_count,  # Number of episodes completed
            "replay_buffer": self.replay_memory_buffer.to_dict(),
            "loss_per_set_iteration": self.loss_per_set_iteration,
            "ep_info_buffer": list(self.ep_info_buffer),
            "optimizer": {
                "state": {
                    str(k): {kk: vv.cpu().numpy() if torch.is_tensor(vv) else vv for kk, vv in v.items()}
                    for k, v in optimizer_state["state"].items()
                },
                "param_groups": optimizer_state["param_groups"],
            },
        }

        # Create directories if they do not exist
        os.makedirs(os.path.dirname(self.save_training_state_path), exist_ok=True)

        # Save the parameters to a file
        with open(self.save_training_state_path, "wb") as f:
            f.write(msgpack.dumps(params_dict))  # Use msgpack to serialize and write the data to the file
        print(Fore.LIGHTCYAN_EX, "OK!", Fore.RESET)  # Confirmation message after saving

    def save_model(self):
        """
        Save the current model and training state (step, episode_count, mean episode reward, and mean episode length) at a given interval (save_freq)
        Save the current state of the model and training progress.
        This method serializes the model's parameters and training statistics
        into a dictionary, which is then saved to a file using msgpack format
        The condition self.step > self.resume_step prevent the model from being saved immediately after resuming training.
        """

        print("\n Saving model...")
        params_dict = {
            "online_network": {
                k: v.detach().cpu().numpy() for k, v in self.online_network.state_dict().items()
            },  # Save all model parameters as numpy arrays
            "step": self.step,  # Current step number
            "episode_count": self.episode_count,  # Number of episodes completed
            "rew_mean": self.info_mean("r"),  # Average reward of episodes
            "len_mean": self.info_mean("l"),  # Average length of episodes
            "loss_per_set_iteration": self.loss_per_set_iteration,
        }

        # Create directories if they do not exist
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        # Save the parameters to a file
        with open(self.save_path, "wb") as f:
            f.write(msgpack.dumps(params_dict))  # Use msgpack to serialize and write the data to the file
        print(Fore.LIGHTYELLOW_EX, "OK!", Fore.RESET)  # Confirmation message after saving

    def load_training_state(self):
        """
        SLoads a previously saved training state if available.:
            - online and  target network parameters
            - replay buffer
            - episode info buffer
            - step count and episode count
        """

        print("\n Resume training from " + self.save_training_state_path + "...")

        with open(self.save_training_state_path, "rb") as f:
            params_dict: dict = msgpack.loads(f.read())  # Use msgpack to deserialize the data from the file

        # Convert the loaded NN parameters to tensors and move them to the correct device
        online_parameters = {k: T.as_tensor(v.copy(), device=self.device) for k, v in params_dict["online_network"].items()}
        target_parameters = {k: T.as_tensor(v.copy(), device=self.device) for k, v in params_dict["target_network"].items()}

        # Load the parameters into the online and target network's
        self.online_network.load_state_dict(online_parameters)
        self.target_network.load_state_dict(target_parameters)

        # load replay buffert
        self.replay_memory_buffer.from_dict(params_dict["replay_buffer"])

        # Load the training progress statistics
        self.resume_step = params_dict["step"]  # Current step number
        self.episode_count = params_dict["episode_count"]  # Number of episodes completed
        self.ep_info_buffer = deque(params_dict["ep_info_buffer"], maxlen=HYPER_PARAMS.ep_info_buffer_capacity)
        rew_mean = self.info_mean("r")  # Average reward of episodes
        len_mean = self.info_mean("l")  # Average length of episodes
        self.loss_per_set_iteration = params_dict["loss_per_set_iteration"]  # loss of current set of iteration

        # Load optimizer
        optimizer = params_dict["optimizer"]
        optimizer_state = {
            "state": {
                int(k): {kk: T.as_tensor(vv.copy(), device=self.device) if isinstance(vv, np.ndarray) else vv for kk, vv in v.items()}
                for k, v in optimizer["state"].items()
            },
            "param_groups": optimizer["param_groups"],
        }
        self.online_network.optimizer.load_state_dict(optimizer_state)

        # Print loaded details: the point where the training resumes
        print(
            "Step: ",  # the step where the model was last saved.
            self.resume_step * self.n_env,
            ", Episodes: ",  # Number of episodes completed
            self.episode_count,
            ", Avg Rew: ",  # Last computed average episode (total) reward
            rew_mean,
            ", Avg Ep Len: ",  # Last computed average episode length
            len_mean,
        )

        # Set the current step to the resume step.
        self.step = self.resume_step

    def load_model(self):
        """
        Loads a previously saved pre-trained model if available.

        Load the model parameters and training progress from a file.
        This method deserializes the model parameters and training statistics from a file that was previously saved using the `save` method.


        This function resumes training from the point (step and episode count) where it was left off.
        """

        print("\n Resume training from " + self.save_path + "...")

        with open(self.save_path, "rb") as f:
            params_dict: dict = msgpack.loads(f.read())  # Use msgpack to deserialize the data from the file

        # Convert the loaded parameters to tensors and move them to the correct device
        parameters = {k: T.as_tensor(v.copy(), device=self.device) for k, v in params_dict["online_network"].items()}

        # Load the parameters into the online network's
        self.online_network.load_state_dict(parameters)

        # Load the training progress statistics
        self.resume_step, self.episode_count, rew_mean, len_mean = (
            params_dict["step"],
            params_dict["episode_count"],
            params_dict["rew_mean"],
            params_dict["len_mean"],
        )
        self.loss_per_set_iteration = params_dict["loss_per_set_iteration"]  # loss of current set of iteration

        # Fill the episode information buffer with the loaded average reward and average episode length.
        # The buffer is filled up to the lesser of the total number of episodes or the buffer's maximum capacity.
        [
            self.ep_info_buffer.append({"r": rew_mean, "l": len_mean})
            for _ in range(
                np.min([self.episode_count, self.ep_info_buffer.maxlen])
            )  # Ensuring the buffer's capacity is not exceeded and avoiding trying to append more values than there are available episodes.
        ]

        # Print loaded details: the point where the training resumes
        print(
            "Step: ",  # the step where the model was last saved.
            self.resume_step * self.n_env,
            ", Episodes: ",  # Number of episodes completed
            self.episode_count,
            ", Avg Rew: ",  # Last computed average episode (total) reward
            rew_mean,
            ", Avg Ep Len: ",  # Last computed average episode length
            len_mean,
        )

        # Set the target network's parameters equal to the online network's parameters (using hard update).
        self.update_target_network(force=True)

        # Set the current step to the resume step.
        self.step = self.resume_step

    def resume_training(self):
        if self.load and (os.path.exists(self.save_training_state_path) or os.path.exists(self.save_path)):
            print(Fore.LIGHTYELLOW_EX, "\n load training state...", Fore.RESET)
            # Check if the file exists, raise an error if it doesn't
            if os.path.exists(self.save_training_state_path):
                self.load_training_state()

            elif os.path.exists(self.save_path):
                self.load_model()
            else:
                raise FileNotFoundError(self.save_path)

    def log(self):
        """
        Logs information about the training progress at a given interval (log_freq).
        Includes details such as average episode reward, average episode length, number of episodes, and elapsed time.
        The condition `self.step > self.resume_step` prevents logging immediately after resuming training.
        """
        if self.step % self.log_freq == 0 and self.step > self.resume_step:
            rew_mean, len_mean = self.info_mean("r"), self.info_mean("l")  # get mean episode reward and mean episode lenght

            print()
            print(
                "Step: ",
                self.step * self.n_env,  # Total steps across all environments
                " (" + str(self.step) + "x" + str(self.n_env) + ")",  # Detailed step count
            )
            print("Avg Ep Rew: ", rew_mean)  # Average episode reward
            print("Avg Ep Len: ", len_mean)  # Average episode length
            print("Episodes: ", self.episode_count)  # Total number of episodes completed
            print("loss: ", self.loss_per_set_iteration)  # Total number of episodes completed
            print(
                "---",
                str(timedelta(seconds=round((time.time() - self.start_time), 0))),  # Elapsed time since training started
                "---",
            )
            print(Fore.LIGHTMAGENTA_EX, time.strftime("%Y-%m-%d %H:%M", time.localtime()), Fore.RESET)

            # Logging metrics to TensorBoard
            self.summary_writer.add_scalar("AvgEpRew", rew_mean, global_step=(self.step * self.n_env))
            self.summary_writer.add_scalar("AvgEpLen", len_mean, global_step=(self.step * self.n_env))
            self.summary_writer.add_scalar("Episodes", self.episode_count, global_step=(self.step * self.n_env))
            self.summary_writer.add_scalar("loss_per_set_iteration", self.loss_per_set_iteration, global_step=self.step)
            self.loss_per_set_iteration = 0
