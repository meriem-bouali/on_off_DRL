import gymnasium as gym
from gymnasium import spaces
import numpy as np


"""Wraps the customized environment in OpenAi Gym"""


class CustomEnvWrapper(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, custom_env): 
        super(CustomEnvWrapper, self).__init__()

        self.custom_env = custom_env

        self.mode = self.custom_env.mode
        # self.player = self.custom_env.player

        self.steps = 0
        self.total_reward = 0

        action_space_n = self.custom_env.action_space_n
        observation_space_n = (
            (self.custom_env.observation_space_n,)
            if isinstance(self.custom_env.observation_space_n, int)
            else self.custom_env.observation_space_n
        )

        self.action_space = spaces.Discrete(action_space_n)
        self.observation_space = spaces.Box(low=-2.0, high=1.0, shape=observation_space_n, dtype=np.float32)
        # ? from romain code: self.observation_space = spaces.Box(low=0., high=1., shape=observation_space_n, dtype=np.float32)


    def reset(self):
        self.steps = 0
        self.total_reward = 0.0

        obs = self.custom_env.reset()
        return obs

    def close(self):
        self.custom_env.stop()

    def setSeed(self, seed):
        self.custom_env.seed=seed

    def step(self, action):
        obs, rwd, done, info = self.custom_env.step(action)

        self.steps += 1
        self.total_reward += rwd 

        return obs, rwd, done, info

  
