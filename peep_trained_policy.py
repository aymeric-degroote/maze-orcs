import numpy as np
import gymnasium as gym
import miniworld
from matplotlib import pyplot as plt
from PIL import Image
import random

from copy import copy
from itertools import count
from collections import deque
from utilities.wrappers import WarpFrame, PyTorchFrame
from utilities.agent import DQNAgent
from utilities.memory import ReplayBuffer
from tqdm import tqdm

import torch

env = gym.make("MiniWorld-Maze-v0", 
               num_rows=3, 
               num_cols=3, 
               room_size=2, 
               render_mode='human',
               view='top')


env = PyTorchFrame(env)
replay_buffer = ReplayBuffer(5000)
agent = DQNAgent(
    env.observation_space,
    env.action_space,
    replay_buffer,
    lr=1e-4,
    batch_size=32,
    gamma=0.99,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

agent.policy_network.load_state_dict(torch.load(r"C:\Users\plarotta\software\maze-orcs\checkpoint2.pth"))
state, _ = env.reset()
for _ in range(50):
    input('step?\n')
    next_state, _, _, _ ,_ = env.step(agent.act)
    env.render()
    state=next_state