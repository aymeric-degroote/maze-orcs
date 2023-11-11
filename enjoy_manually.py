import numpy as np
import gymnasium as gym
import miniworld
import pyglet
from pyglet.window import key
from pyglet import clock
import sys
from gymnasium import envs
from utilities.manual_control import ManualControl

ENVIRONMENT_NAME = "MiniWorld-Maze-v0" #change to whatever miniworld environment you want to manually play with
env = gym.make(ENVIRONMENT_NAME, 
               num_rows=3, 
               num_cols=3, 
               room_size=2, 
               render_mode='human',
               view='top')
env.reset()

manual_control = ManualControl(env, no_time_limit=False, domain_rand=False)
manual_control.run()






