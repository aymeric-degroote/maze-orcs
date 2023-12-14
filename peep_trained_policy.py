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
from dqn_utilities.wrappers import WarpFrame, PyTorchFrame, MaxAndSkipEnv, FrameStack, BetterReward
from dqn_utilities.agent import DQNAgent
from dqn_utilities.memory import ReplayBuffer
from tqdm import tqdm
from dqn_utilities.larotta_maze import PedroMaze
from stable_baselines3 import DQN
from stable_baselines3 import A2C
import copy
import torch
import numpy as np

env = PedroMaze(num_rows=3, 
               num_cols=3, 
               room_size=2, 
               render_mode='human',
               max_episode_steps = 400,
               view='top',
               rng = np.random.default_rng(seed = 1))
# env = gym.make("MiniWorld-Maze-v0", 
#                num_rows=3, 
#                num_cols=3, 
#                room_size=2, 
#                render_mode='human',
#                max_episode_steps = 500,
#                view='top')
# env = WarpFrame(env)
# env = PyTorchFrame(env)
# # env = MaxAndSkipEnv(env, skip=4)
env = WarpFrame(env)

# # env = ClipRewardEnv(env)

# env = BetterReward(env)
env = FrameStack(env, 8)
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

agent.policy_network.load_state_dict(torch.load(r"C:\Users\plarotta\software\maze-orcs\dqn_results\models\12-2\model50000.pth"))
state, info = env.reset()


# model = DQN.load("dqn_miniworld")
# model = DQN.load("dqn_miniworld")
for _ in range(300):
    # input('step?\n')
    # action, _states = model.predict(state, deterministic=False)
    action = agent.act(state)
    next_state, _, done, truncated ,_ = env.step(action)
    env.render()
    state=copy.deepcopy(next_state)
    if done or truncated:
        break
    time.sleep(.05)