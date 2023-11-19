from typing import SupportsFloat
import gymnasium as gym
from gymnasium import spaces
import cv2
import numpy as np



class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work.
        Expects inputs to be of shape height x width x num_channels
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 80
        self.height = 60
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]
    

class PyTorchFrame(gym.ObservationWrapper):
    """Image shape to num_channels x height x width"""

    def __init__(self, env):
        super(PyTorchFrame, self).__init__(env)
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(shape[-1], shape[0], shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return np.rollaxis(observation, 2)
    

class BetterReward(gym.RewardWrapper):
    '''Reward on absolute distance to goal'''
    def __init__(self, env):
        super(BetterReward, self).__init__(env)
        self.env = env
        
    def reward(self, reward):
        agent = self.env.agent
        box = self.env.box
        bonus = 5000*reward if self.near(self.box) else 0
        return -np.linalg.norm(box.pos - agent.pos) + bonus