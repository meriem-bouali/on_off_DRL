from dqn import Agents, HYPER_PARAMS, network_config  # network_config_SumoGui
from rl_env.env_tools import CustomEnvWrapper, make_env
from rl_env.custom_env import SumoEnv

import os
import time
import itertools
from datetime import timedelta

from policy.logger import Logger


"""
This module implements a training loop for a Deep Q-Network (DQN) agent
The main execution is triggered when the script is run directly.

The agent's state representation is derived directly from SUMO's graphical interface, using screenshots captured from the SUMO-GUI.
"""


class TrainOnRL:
    def __init__(
        self,
        algo=HYPER_PARAMS.algo,
        max_total_steps=HYPER_PARAMS.max_total_steps,
        n_env=HYPER_PARAMS.n_env,
    ):
        # Set the environment variables for CUDA device configuration
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = HYPER_PARAMS.gpu
        self.policy = "OnRL"
        self.mode = "train"

        # Initialize the environment
        self.env = make_env(env=CustomEnvWrapper(custom_env=SumoEnv(mode=self.mode, policy=self.policy, gui=False)), n_env=n_env)

        # Create a Logger instance for storing metrics and agent transitions in HDF5/CSV files.
        self.logger = Logger(algo[:-5], self.mode, self.policy)

        # Initialize the agent's hyperparameters and network configuration
        self.agent: Agents.Agent = getattr(Agents, algo)(
            policy=self.policy,
            algo=algo,
            nn_conf_func=network_config,
            input_dim=self.env.observation_space,
            output_dim=self.env.action_space.n,  # action space is type of gym.space.Discrete
        )

        # Load the pre-trained model to resume training if it was interrupted previously
        self.agent.resume_training()

        self.max_total_steps = max_total_steps

        # Print training setup
        print(" \n TRAIN \n")
        print("‣ Algo", algo)
        print(self.agent)
        print("  • max_total_steps: ", max_total_steps)
        print("  • n_env: ", n_env)

    def init_replay_memory_buffer(self):
        print("\n Initialize Replay Memory Buffer")

        # Reset the environment and get the initial observations
        obses = self.env.reset()

        # number of steps required to fill the replay buffer the minimum size
        step_fill = self.agent.min_buffer_size // self.agent.n_env

        # Fill the replay memory buffer up to the minimum buffer size
        for t in range(step_fill):
            """
             The model is saved periodically, but the replay buffer is not. 
             On resuming training, the buffer starts empty and is refilled using the pretrained model.
            """
            # if the agent model/policy was trained for more than step_fill steps
            if t >= step_fill - self.agent.resume_step:
                # Choose actions based on the current policy with epsilon-greedy method
                actions = self.agent.choose_actions(obses)
            else:
                # Choose random action
                actions = [self.env.action_space.sample() for _ in range(self.agent.n_env)]

            # Execute the chosen actions in the environment
            new_obses, rews, dones, _ = self.env.step(actions)

            # Store the resulting transitions (current observations, actions, rewards, done flags, and next observations) in the replay buffer
            self.agent.store_transitions(obses, actions, rews, dones, new_obses, None)

            # Store the transitions in csv file
            self.logger.store_trans_csv(obses, actions, rews, dones, new_obses)

            # Update the current observations to the new observations
            obses = new_obses

            # Print progress every 10,000 steps
            if (t + 1) % (10000 // self.agent.n_env) == 0:
                print(str((t + 1) * self.agent.n_env) + " / " + str(self.agent.min_buffer_size))
                print("---", str(timedelta(seconds=round((time.time() - self.agent.start_time), 0))), "---")
                print(time.strftime("%Y-%m-%d %H:%M", time.localtime()))

    def train_loop(self):
        # Reset the environment and get the initial observations
        obses = self.env.reset()
        for step in itertools.count(start=self.agent.resume_step):
            # Update the current step count
            self.agent.step = step

            # Choose actions based on the current policy with epsilon-greed6y method
            actions = self.agent.choose_actions(obses)

            # Take a step in the environment with the chosen actions
            new_obses, rews, dones, infos = self.env.step(actions)

            # Store the transitions in the replay buffer
            self.agent.store_transitions(obses, actions, rews, dones, new_obses, infos)

            # Store the transitions in csv file
            self.logger.store_trans_csv(obses, actions, rews, dones, new_obses)

            # log episode info
            # // self.logger.log_episode_metrics(infos, dones)

            # Update the current observations
            obses = new_obses

            # Apply a single learning step
            self.agent.learn()

            # log the loss per iteration
            # // self.logger.log_loss("loss per iteration", loss, step)  # log the loss per iteration

            # Update the target network periodically
            self.agent.update_target_network()

            # Log the training progress periodically
            self.agent.log()

            # Save the model periodically.
            if self.agent.step % self.agent.save_freq == 0 and self.agent.step > self.agent.resume_step:
                if self.agent.buffer_location == "disk":
                    # If the replay buffer is stored on disk, save the full training state (model + buffer).
                    self.agent.save_training_state()
                else:
                    # If the buffer is stored in RAM, saving it frequently is too time-consuming,
                    # so we only save the model.
                    self.agent.save_model()
                    # self.agent.save_training_state()

            # Exit the training loop if the maximum number of steps is reached
            # Note: If self.max_total_steps is set to 0, the training will continue indefinitely
            if bool(self.max_total_steps) and (step * self.agent.n_env) >= self.max_total_steps:
                self.agent.save_model()
                self.agent.save_training_state()
                exit()

    def run(self):
        try:
            if self.agent.replay_memory_buffer.len() < self.agent.min_buffer_size:  # Check if the replay buffer is empty
                # fills up the buffer with initial experiences
                self.init_replay_memory_buffer()
            else:
                # The replay buffer is already filled/initialized because load_training_state() is called inside agent.__init__()
                print("Loading training state...")

            # start the training loop
            print("\n Start Training")
            self.train_loop()
        except KeyboardInterrupt:
            import psutil
            import os
            import signal

            # Terminate the SUMO Process
            for proc in psutil.process_iter(["pid", "name"]):
                if "sumo" in proc.info["name"] or "sumo-gui" in proc.info["name"]:
                    os.kill(proc.info["pid"], signal.SIGTERM)  # Or SIGKILL for forceful termination
                    print(f"Killed process {proc.info['name']} with PID {proc.info['pid']}")
            self.agent.save_training_state()
            self.agent.save_model()


if __name__ == "__main__":
    TrainOnRL().run()
