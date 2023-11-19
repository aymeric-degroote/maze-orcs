import numpy as np
import gymnasium as gym
import miniworld
from matplotlib import pyplot as plt
from PIL import Image
import random
import time

from copy import copy
from itertools import count
from collections import deque
from utilities.wrappers import WarpFrame, PyTorchFrame
from utilities.agent import DQNAgent
from utilities.memory import ReplayBuffer
from tqdm import tqdm
from utilities.larotta_maze import PedroMaze

import torch
import numpy as np

# env = gym.make("MiniWorld-Maze-v0", 
#                num_rows=3, 
#                num_cols=3, 
#                room_size=2, 
#                render_mode='human',
#                max_episode_steps = 400,
#                view='top')
env = PedroMaze(num_rows=4, 
               num_cols=4, 
               room_size=3, 
               render_mode='human',
               max_episode_steps = 400,
               view='top',
               rng = np.random.default_rng(seed = 1))

env = WarpFrame(env)
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

agent.policy_network.load_state_dict(torch.load(r"C:\Users\plarotta\software\maze-orcs\checkpoint3.pth"))
state, _ = env.reset(seed=8)
for _ in range(300):
    # input('step?\n')
    next_state, _, done, truncated ,_ = env.step(agent.act(state))
    env.render()
    state=next_state
    if done or truncated:
        break
    time.sleep(.1)