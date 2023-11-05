"""
Used this minigrid tutorial:
https://minigrid.farama.org/content/create_env_tutorial/

And integrated mazelib to generate the maze

Used this tutorial for RL agent:
https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/

"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import seaborn as sns

from environment import MazeEnv
from policymaker import MazeGrid, Agent
import gymnasium as gym


def main():
    
    num_episodes = 1000
    max_num_step = 50
    step_print = 100
    
    env = MazeEnv(#render_mode="human", 
                  maze_type="prims" # "dungeon"
                 )
    
    total_num_episodes = num_episodes
    rewards_over_seeds = []
    
    observation, info = env.reset(seed=0)
    observation = observation.get('image')[:,:,0]
    
    obs_space_dims = observation.shape
    action_space_dims = 3
    
    for seed in range(1):  # TODO: will need more seeds later
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        agent = Agent(obs_space_dims, action_space_dims, 
                      size=env.size, load_maze=False,
                      training=True
                          #policy="network"
                         )
        reward_over_episodes = []
        
        for ep_id in range(num_episodes):
            observation, info = env.reset(seed=ep_id)
            observation = observation.get('image')[:,:,0]
            
            ep_reward = 0
            
            for i in range(max_num_step):
                action = agent.policy(observation)

                observation, reward, terminated, truncated, info = env.step(action)
                observation = observation.get('image')[:,:,0]
                
                agent.rewards.append(reward)
                ep_reward += reward

                if terminated or truncated:
                    break
            
            reward_over_episodes.append(ep_reward)
            agent.update()
            
            if (ep_id+1)%step_print == 0:
                print(f"Episode {str(ep_id+1).rjust(3)}: Average Reward {np.mean(reward_over_episodes[-step_print:])}")

        plt.plot(reward_over_episodes)
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.show()


if __name__ == "__main__":
    main()
