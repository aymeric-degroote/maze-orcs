'''Here I will try to implement a basic DQN to solve the maze nav task'''

import numpy as np
import gymnasium as gym
import miniworld
from matplotlib import pyplot as plt
from PIL import Image



env = gym.make("MiniWorld-Maze-v0", 
               num_rows=10, 
               num_cols=10, 
               room_size=5, 
               render_mode='human',
               view='top')
env.reset()

initial_action = env.action_space.sample()
print(f'action space: {env.action_space}')
print(f'obs space: {env.observation_space}')
print(f'random act: {initial_action}')

obs, _, _, _, _ = env.step(initial_action)
print(obs.shape)

# obs[0:256, 0:256] = [255, 0, 0] # red patch in upper left
img = Image.fromarray(obs, 'RGB')
img.save('my.png')
img.show()




