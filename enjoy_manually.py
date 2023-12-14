import numpy as np
import gymnasium as gym
import miniworld
import pyglet
from pyglet.window import key
from pyglet import clock
import sys
from gymnasium import envs
from dqn_utilities.manual_control import ManualControl
from dqn_utilities.wrappers import WarpFrame, PyTorchFrame, MaxAndSkipEnv, FrameStack, BetterReward

from dqn_utilities.larotta_maze import PedroMaze

# ENVIRONMENT_NAME = "MiniWorld-Maze-v0" #change to whatever miniworld environment you want to manually play with
# env = gym.make(ENVIRONMENT_NAME, 
#                num_rows=3, 
#                num_cols=3, 
#                room_size=2, 
#                render_mode='human',
#                view='top')
env = PedroMaze(num_rows=3, 
               num_cols=3, 
               room_size=2, 
               render_mode='human',
               max_episode_steps = 300,
               view='agent',
               rng = np.random.default_rng(seed = 1))
# env = MaxAndSkipEnv(env, skip=4)
env = WarpFrame(env)
env = PyTorchFrame(env)
env = FrameStack(env, 4)

env.reset()

manual_control = ManualControl(env, no_time_limit=False, domain_rand=False)
manual_control.run()






