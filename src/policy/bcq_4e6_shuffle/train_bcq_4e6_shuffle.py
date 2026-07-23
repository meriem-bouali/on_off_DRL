from dqn import BCQAgent, HYPER_PARAMS, network_config  # network_config_SumoGui
from gymnasium import spaces
import numpy as np

import os
import itertools

from policy.logger import Logger
from policy.utils import set_seed

"""
This module implements a training loop for a Deep Q-Network (DQN) agent
The main execution is triggered when the script is run directly.

The agent's state representation is derived directly from SUMO's graphical interface, using screenshots captured from the SUMO-GUI.
"""


class TrainBCQ4e6Shuffle:
    def __init__(self, csv_dir_path, algo="BCQAgent", max_total_iteration=HYPER_PARAMS.nb_total_iteration):
        # Set the environment variables for CUDA device configuration
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = HYPER_PARAMS.gpu
        self.policy = "BCQ4e6Shuffle"
        self.mode = "train"
        self.max_total_iteration = max_total_iteration

        # Create a Logger instance for storing metrics and agent transitions in HDF5/CSV files.
        self.logger = Logger(algo[:-5], self.mode, self.policy, store_trs=False)
        

        # Initialize the agent's hyperparameters and network configuration
        self.agent=BCQAgent(
            policy=self.policy,
            algo=algo,
            nn_conf_func=network_config,
            input_dim=spaces.Box(low=-2.0, high=1.0, shape=HYPER_PARAMS.observation_n, dtype=np.float32),
            output_dim=spaces.Discrete(HYPER_PARAMS.action_n).n,  # action space is type of gym.space.Discrete
            csv_dir_name=csv_dir_path,  # ? added
            shuffle=True
        )

        # Load the pre-trained model to resume training if it was interrupted previously
        self.agent.resume_training()

        # Print training setup
        print(" \n TRAIN \n")
        print("‣ Algo", algo)
        print(self.agent)

    def train_loop(self):
        for iteration in itertools.count(start=self.agent.resume_iteration):
            # Update the current step count
            self.agent.iteration = iteration

            try:
                transitions = next(self.agent.dataloader_iter)
            except StopIteration:
                self.agent.dataloader_iter = iter(self.agent.dataloader)  # Reset when exhausted
                transitions = next(self.agent.dataloader_iter)
                self.agent.epoch += 1

                # log the loss per epoch
                self.logger.log_loss("loss per epoch", self.agent.loss_per_epoch / self.agent.iter_per_epoch, self.agent.epoch)
                self.agent.loss_per_epoch = 0

            # Apply a single learning step
            self.agent.learn(transitions)

            # log the loss at each iteration
            # //self.logger.log_loss("loss per iteration", loss, iteration)

            # Update the target network periodically
            self.agent.update_target_network()

            # Log the training progress periodically
            self.agent.log()

            # Save the model periodically.
            if self.agent.iteration % self.agent.save_freq == 0 and self.agent.iteration > self.agent.resume_iteration:
                self.agent.save_training_state()
                # self.logger.log_loss(
                #     "loss_per_set_iteration", self.agent.loss_per_set_iteration, self.agent.iteration
                # )
                # self.agent.loss_per_set_iteration = 0

            # Exit the training loop if the maximum number of steps is reached
            # Note: If self.max_total_steps is set to 0, the training will continue indefinitely
            if self.agent.iteration >= self.max_total_iteration:
                self.agent.save_training_state()  # we save the model if we want latter performe more trainning
                self.agent.save_model()
                # // self.logger.log_loss(
                # //     "loss per epoch", self.loss_per_epoch / self.agent.iter_per_epoch, self.agent.epoch + 1
                # // )  # log the loss of last epoch
                exit()

    def run(self):
        set_seed()
        try:
            # start the training loop
            print("\n Start Training")
            self.train_loop()
        except KeyboardInterrupt:
            self.agent.save_training_state()
            self.agent.save_model()  # we save the model if we want to evaluate the current trained models


if __name__ == "__main__":
    TrainBCQ4e6Shuffle(csv_dir_path="transition_csv_DoubleDQN_OnRL", max_total_iteration=HYPER_PARAMS.nb_total_iteration).run()
