"""
Used this minigrid tutorial:
https://minigrid.farama.org/content/create_env_tutorial/

And integrated mazelib to generate the maze
"""

from __future__ import annotations
# import gym-minigrid

import numpy as np
import matplotlib.pyplot as plt

from environment import MazeEnv
from policymaker import MazeGrid, Agent

from plot_functions import showPNG   # TODO: what is it for? can we remove it?


def main():
    
    num_episodes = 2
    max_num_step = 1000
    
    env = MazeEnv(render_mode="human", 
                  maze_type="dungeon" #"prims"
                 )
    
    observation, info = env.reset(seed=0)
    observation = observation.get('image')[:,:,0]
    
    obs_space_dims = observation.shape
    action_space_dims = 3
    
    print('obs dims')
    print(obs_space_dims)
    
    
    for ep_id in range(num_episodes):
        observation, info = env.reset(seed=ep_id)
        observation = observation.get('image')[:,:,0]
        
        agent = Agent(obs_space_dims, action_space_dims,
                      size=env.size, load_maze=True
                      #policy="greedy"
                     )
        
        for i in range(max_num_step):
            action = agent.policy(observation)
            
            observation, reward, terminated, truncated, info = env.step(action)
            observation = observation.get('image')[:,:,0]
            agent.rewards.append(reward)
            
            # Reward system:
            # success: ‘1 - 0.9 * (step_count / max_steps)’
            # failure: ‘0’
            
            # TODO: use this tutorial for RL implementation using reward and NN
            # https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/#sphx-glr-tutorials-training-agents-reinforce-invpend-gym-v26-py
            
            if terminated or truncated:
                break
        
        # TODO: delete the following comments if not needed
        ## enable manual control for testing
        #manual_control = ManualControl(env, seed=42)
        #manual_control.start()

    env.close()

    
if __name__ == "__main__":
    main()
