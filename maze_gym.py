"""
Used this minigrid tutorial:
https://minigrid.farama.org/content/create_env_tutorial/

And integrated mazelib to generate the maze
"""

from __future__ import annotations
# import gym-minigrid

import numpy as np
import matplotlib.pyplot as plt

# minigrid: RL environment (+ UI)
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

# mazelib: generates mazes
import mazelib
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.generate.BacktrackingGenerator import BacktrackingGenerator


from environment import MazeEnv
from policymaker import MazeGrid, Agent

from plot_functions import showPNG

def main():
    env = MazeEnv(render_mode="human")
    
    num_episodes = 5
    max_num_step = 1000
    for ep_id in range(num_episodes):
        observation, info = env.reset(seed=ep_id)

        agent = Agent(direction=env.agent_start_dir,
                      size=env.size)

        for i in range(max_num_step):
            #print(observation.get('image')[:,:,0])
            #input()
            action = agent.policy(observation)  # User-defined policy function
            observation, reward, terminated, truncated, info = env.step(action)
            
            # A reward of ‘1 - 0.9 * (step_count / max_steps)’ is given for success
            # and ‘0’ for failure.
            agent.rewards.append(reward)

            if terminated or truncated:
                break
        
        # enable manual control for testing
        #manual_control = ManualControl(env, seed=42)
        #manual_control.start()

        #env.close()

    
if __name__ == "__main__":
    main()
