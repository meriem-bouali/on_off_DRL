import sys
import os


from dqn import HYPER_PARAMS, network_config, BCNetwork  # network_config_SumoGui
from rl_env.env_tools import CustomEnvWrapper, make_env
import os
from rl_env.custom_env import SumoEnv
from torch import device, cuda
from sumo_sim.sim_config import SimulationConfig as SC


from policy.logger import Logger
from datetime import datetime
import numpy as np
from policy.utils import set_seed


"""
This module implements a testing loop for evaluating the performance of a trained DQN model.
The main execution is triggered when the script is run directly.

The agent's state representation is derived directly from SUMO's graphical interface, using screenshots captured from the SUMO-GUI.
"""


class TestBCQ4e6Cleanshuffle:
    def __init__(self, model_name: str, seed: list, max_episodes: int, seed_category:str, gui: bool = False):
        # self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "save", model_name)

        self.model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "save", model_name
        )

        self.gui = gui
        self.ego_id = SC.ego_veh_id
        self.seed = seed

        # Set the environment variables for CUDA device configuration
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = HYPER_PARAMS.gpu

        self.policy = "BCQ4e6CleanShuffle"
        self.mode = "test"

        # Initialize the environment
        self.env = make_env(env=CustomEnvWrapper(custom_env=SumoEnv(mode=self.mode, policy=self.policy, gui=self.gui)))

        # Extract model identifier from the model path
        model_id = model_name.split("/")[-1].split(self.policy + "_model.pack")[0]  # model_id=agent+_+lr

        # Get Agent class name and learning rate from the model_id
        agent, _, lr, *_ = model_id.split("_")

        # Instantiate the corresponding network class based on the agent type
        self.network = BCNetwork(
            device(("cuda:" + HYPER_PARAMS.gpu) if cuda.is_available() else "cpu"),
            float(lr),
            network_config,
            self.env.observation_space,
            self.env.action_space.n,
        )

        # Create a Logger instance for storing metrics and agent transitions in HDF5/CSV files.
        self.logger = Logger(agent, self.mode, self.policy, store_trs=True,extra=seed_category)

        # Load the model and # Load its parameters into the networkparameters into the network

        self.network.load(self.model_path)

        self.env.setSeed(self.seed[0])
        # Reset the environment
        self.obs = self.env.reset()

        # Initialize variables
        self.action = 0
        self.ep = 0

        # Print Hyperparameters
        print("\nTesting \n")
        # for field_name, value in HYPER_PARAMS.__dict__.items():
        #     if not field_name.startswith("__"):
        #         print(f"  • {field_name}: {value}")

        self.max_episodes = max_episodes

    def loop(self):
        # Select action
        self.action = self.network.actions(np.expand_dims(self.obs, axis=0))[0]

        # Execute the action
        new_obs, rw, done, self.info = self.env.step(self.action)
        # print("rw=",rw, "  action=",self.action)
        self.logger.store_trans_test(episode_count=self.ep, obse=self.obs, action=self.action, reward=rw, done=done, new_obse=new_obs)
        self.obs = new_obs

        # If the episode ends (done is True)
        if done:
            self.ep += 1
            self.info["episode_num"] = self.ep
            self.info["wall_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            # Log  episode info in csv
            self.logger.log_info_test(self.info)

            # Print episode count and information
            print("\nEpisode :", self.ep)
            [print(k, ":", self.info[k]) for k in self.info]

            # Exit if the maximum number of episodes is reached.
            if self.ep >= self.max_episodes:
                self.env.close()
                return False

            else:
                self.env.setSeed(self.seed[self.ep])
                # Reset the environment
                self.obs = self.env.reset()
        return True

    def run(self):
        set_seed()
        try:
            while True:
                if not self.loop():
                    break
        except KeyboardInterrupt:
            # Print the last episode info
            print(self.info)


if __name__ == "__main__":
    TestBCQ4e6Cleanshuffle(
        model_name="BCQ_lr_0.0001_BCQ4e6CleanShuffle_model.pack", seed=SC.seed_test, max_episodes=len(SC.seed_test),seed_category="seed_test"
    ).run()

    TestBCQ4e6Cleanshuffle(
        model_name="BCQ_lr_0.0001_BCQ4e6CleanShuffle_model.pack", seed=SC.seed_train, max_episodes=len(SC.seed_train),seed_category="seed_train"
    ).run()
