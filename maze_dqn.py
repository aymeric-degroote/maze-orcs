'''Here I will try to implement a basic DQN to solve the maze nav task'''

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
               render_mode='top',
               view='top')

env = PyTorchFrame(env)
print("initialized environment...")

replay_buffer = ReplayBuffer(5000)
print("initialized replay buffer...")


agent = DQNAgent(
    env.observation_space,
    env.action_space,
    replay_buffer,
    lr=1e-4,
    batch_size=32,
    gamma=0.99,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
print("initialized dqn agent...")


eps_timesteps = 0.1* \
    int(500000)
episode_rewards = [0.0]

state, _ = env.reset()


for t in tqdm(range(500000)):
    if t % 250 == 0:
        print(f'training episode step {t}, rewards: {episode_rewards}')
    fraction = min(1.0, float(t) / eps_timesteps)
    eps_threshold = 1 + fraction * \
        (0.01 - 1)
    sample = random.random()

    if(sample > eps_threshold):
        # Exploit
        action = agent.act(state)
    else:
        # Explore
        action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    agent.memory.add(state, action, reward, next_state, float(done))
    state = next_state

    episode_rewards[-1] += reward
    if done:
        state,_ = env.reset()
        episode_rewards.append(0.0)

    if t > 1000 and t % 1 == 0:
        agent.optimise_td_loss()

    if t > 1000 and t % 100 == 0:
        agent.update_target_network()

    num_episodes = len(episode_rewards)

torch.save(agent.policy_network.state_dict(), f'checkpoint2.pth')
np.savetxt('rewards_per_episode2.csv', episode_rewards,
            delimiter=',', fmt='%1.3f')



















