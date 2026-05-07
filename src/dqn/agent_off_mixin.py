import os
import msgpack
import msgpack_numpy as m
import numpy as np
import torch as T
from colorama import Fore
from datetime import timedelta
import time
import pandas as pd
from itertools import islice
from torch.utils.data import TensorDataset, DataLoader
from collections import deque

m.patch()  # automatically force all msgpack serialization and deserialization routines


class AgentOffMixin:
    def loading_agent_data(self, resume_iteration):
        # get csv files names
        csv_files = [os.path.join(self.csv_dir_path, f) for f in os.listdir(self.csv_dir_path) if f.endswith(".csv")]

        # Read all CSVs in one go
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

        # data to transition data
        state_cols = [
            "has_right_lane",
            "has_left_lane",
            "driving_in_weaving",
            "dist_to_onramp",
            "dist_to_offramp",
            "leader_gap",
            "leader_relatif_s",
            "follower_gap",
            "follower_relatif_s",
            "left_leader_gap",
            "left_leader_relatif_s",
            "left_follower_gap",
            "left_follower_relatif_s",
            "right_leader_gap",
            "right_leader_relatif_s",
            "right_follower_gap",
            "right_follower_relatif_s",
        ]

        next_state_cols = ["next_" + col for col in state_cols]

        # input_dim, output_dim = len(state_cols), len(df["action"].unique())
        # Convert to tensors
        obses_t = T.tensor(df[state_cols].values, dtype=T.float32).to(self.device)
        actions_t = T.tensor(df["action"].values, dtype=T.long).to(self.device).unsqueeze(1)
        rews_t = T.tensor(df["reward"].values, dtype=T.float32).to(self.device).unsqueeze(1)
        dones_t = T.tensor(df["done"].values, dtype=T.float32).to(self.device).unsqueeze(1)
        new_obses_t = T.tensor(df[next_state_cols].values, dtype=T.float32).to(self.device)

        # Create TensorDataset
        dataset = TensorDataset(obses_t, actions_t, rews_t, dones_t, new_obses_t)

        # Create DataLoader
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        self.dataloader_iter = iter(self.dataloader)

        # Skip batches if resuming (deque faster than for loop)
        batches_to_skip = resume_iteration % len(self.dataloader)
        deque(islice(self.dataloader_iter, batches_to_skip), maxlen=0)

        size_agent_data = len(df)
        df.drop(df.index, inplace=True)  # memory freeing

        return size_agent_data

    def save_model(self):
        print("\n Saving model...")
        params_dict = {
            "online_network": {
                k: v.detach().cpu().numpy() for k, v in self.online_network.state_dict().items()
            },  # Save all model parameters as numpy arrays
            "iteration": self.iteration,  # Current iteration number
            "epoch": self.epoch,  # Number of epoch completed
            "loss_per_epoch": self.loss_per_epoch,
            "loss_per_set_iteration": self.loss_per_set_iteration,
        }

        # Create directories if they do not exist
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        # Save the parameters to a file
        with open(self.save_path, "wb") as f:
            f.write(msgpack.dumps(params_dict))  # Use msgpack to serialize and write the data to the file
        print(Fore.LIGHTYELLOW_EX, "OK!", Fore.RESET)  # Confirmation message after saving

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
        self.resume_iteration, self.epoch = (params_dict["iteration"], params_dict["epoch"])
        self.loss_per_epoch = params_dict["loss_per_epoch"]  # loss of current epoch
        self.loss_per_set_iteration = params_dict["loss_per_set_iteration"]  # loss of current set of iteration

        # Print loaded details: the point where the training resumes
        print(
            "iteration: ",  # the iteration where the model was last saved.
            self.resume_iteration,
            ", Episodes: ",  # Number of episodes completed
            self.epoch,
        )

        # Set the target network's parameters equal to the online network's parameters (using hard update).
        self.update_target_network(force=True)

        # Set the current iteration to the resume iteration.
        self.iteration = self.resume_iteration

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
            # //"loss_info_buffer": list(self.loss_info_buffer),
            "iteration": self.iteration,  # Current iteration number
            "epoch": self.epoch,  # Current epoch number
            "loss_per_epoch": self.loss_per_epoch,
            "loss_per_set_iteration": self.loss_per_set_iteration,
            "optimizer": {
                "state": {
                    str(k): {kk: vv.cpu().numpy() if T.is_tensor(vv) else vv for kk, vv in v.items()}
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

    def load_training_state(self):
        """
        Loads a previously saved training state if available.:
            - online and  target network parameters
            - ietration count and epoch count
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

        # load loss
        # //self.loss_info_buffer = deque(params_dict["loss_info_buffer"], maxlen=HYPER_PARAMS.log_freq)

        # Load the training progress statistics
        self.resume_iteration = params_dict["iteration"]  # Current iteration number
        self.epoch = params_dict["epoch"]  # Current iteration number
        self.loss_per_epoch = params_dict["loss_per_epoch"]  # loss of current epoch
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
            "iteration: ",  # the iteration where the model was last saved.
            self.resume_iteration,
            ", epoch: ",  # Number of epochs completed
            self.epoch,
        )

        self.iteration = self.resume_iteration

    def resume_training(self):
        if self.load and os.path.exists(self.save_training_state_path):
            print(Fore.LIGHTYELLOW_EX, "\n load training state...", Fore.RESET)
            # Check if the file exists, raise an error if it doesn't
            if os.path.exists(self.save_training_state_path):
                self.load_training_state()
            else:
                raise FileNotFoundError(self.save_training_state_path)

    def log(self):
        """
        Logs information about the training progress at a given interval (log_freq).
        Includes details such as average episode reward, average episode length, number of episodes, and elapsed time.
        The condition `self.iteration > self.resume_iteration` prevents logging immediately after resuming training.
        """
        if self.iteration % self.log_freq == 0 and self.iteration > self.resume_iteration:
            print()
            print(
                "iteration: ",
                self.iteration,  # Total steps across all environments
            )
            print("Epoch: ", self.epoch)  # Total number of epoch completed
            print("loss: ", self.loss_per_set_iteration)
            print(
                "---",
                str(timedelta(seconds=round((time.time() - self.start_time), 0))),  # Elapsed time since training started
                "---",
            )
            print(Fore.LIGHTMAGENTA_EX, time.strftime("%Y-%m-%d %H:%M", time.localtime()), Fore.RESET)

            # Logging metrics to TensorBoard # log the loss per log freq
            # // self.summary_writer.add_scalar(
            # //     "loss", sum(self.loss_info_buffer) / len(self.loss_info_buffer) if self.loss_info_buffer else 0.0, self.iteration
            # // )
            self.summary_writer.add_scalar(
                "loss_per_set_iteration", round(self.loss_per_set_iteration / self.log_freq, 2), global_step=self.iteration
            )
            self.loss_per_set_iteration = 0
