import sys
import os

# sys.path.insert(0, os.path.expanduser("~/Documents/rl-state-study"))

from dqn import HYPER_PARAMS, network_config, Networks  # network_config_SumoGui
from rl_env.env_tools import CustomEnvWrapper, make_env
import os
from rl_env.custom_env import SumoEnv
from torch import device, cuda
from sumo_sim.sim_config import SimulationConfig as SC


from policy.logger import Logger
from datetime import datetime
import numpy as np

"""
This module implements a testing loop for evaluating the performance of a trained DQN model.
The main execution is triggered when the script is run directly.

The agent's state representation is derived directly from SUMO's graphical interface, using screenshots captured from the SUMO-GUI.
"""

# Dictionary maps different agent types to their corresponding neural network classes.
network_name = {
    "DQN": "DeepQNetwork",
    "DoubleDQN": "DeepQNetwork",
    "DuelingDoubleDQN": "DuelingDeepQNetwork",
    "PerDuelingDoubleDQN": "DuelingDeepQNetwork",
}


class TestOffRL4e6:
    def __init__(self, model_name: str, seed:list,max_episodes: int, gui: str = False):
        # self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "save", model_name)


        self.model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "save", "training_state_disk", model_name
            )


        self.gui = gui
        self.ego_id = SC.ego_veh_id
        self.seed=seed

        # Set the environment variables for CUDA device configuration
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = HYPER_PARAMS.gpu

        self.policy = "OffRL4e6"
        self.mode = "test"

        # Initialize the environment
        self.env = make_env(env=CustomEnvWrapper(custom_env=SumoEnv(mode=self.mode, policy=self.policy, gui=self.gui)))

        # Extract model identifier from the model path
        model_id = model_name.split("/")[-1].split(self.policy + "_model.pack")[0]  # model_id=agent+_+lr

        # Get Agent class name and learning rate from the model_id
        agent, _, lr, *_ = model_id.split("_")

        # Instantiate the corresponding network class based on the agent type
        self.network: Networks.Network = getattr(
            Networks,
            network_name[agent],  # Get the neural network class name
        )(
            device(("cuda:" + HYPER_PARAMS.gpu) if cuda.is_available() else "cpu"),
            float(lr),
            network_config,
            self.env.observation_space,
            self.env.action_space.n,
        )

        # Create a Logger instance for storing metrics and agent transitions in HDF5/CSV files.
        self.logger = Logger(agent, self.mode, self.policy, store_trs=False)

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
                exit()

            else:
                self.env.setSeed(self.seed[self.ep])
                # Reset the environment
                self.obs = self.env.reset()
                

    def run(self):
        try:
            while True:
                self.loop()
        except KeyboardInterrupt:
            # Print the last episode info
            print(self.info)


if __name__ == "__main__":
    TestOffRL4e6(model_name="DoubleDQN_lr_0.0001_OffRL4e6_model.pack",seed=SC.seed_test, max_episodes=len(SC.seed_test)).run()
