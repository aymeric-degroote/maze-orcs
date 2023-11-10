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
import os

from environment import MazeEnv
from policymaker import MazeGrid, Agent
import gymnasium as gym
import pickle

def main():
    
    num_episodes = 1000
    max_num_step = 100
    
    size = 13
    weights_fn = "model_weights.pth"
    load_weights = weights_fn
    param_file = "param_model1.pkl"
    
    obs_space_dims = 49
    hidden_space_dims = [16,16]
    action_space_dims = 3
    model_dims = [obs_space_dims]+hidden_space_dims+[action_space_dims]
    
    with open(param_file, 'rb') as fp:
        params = pickle.load(fp)
    print("Loaded params dict from", param_file)
    
    env = MazeEnv(render_mode="human", 
                  size=size, 
                  maze_type="prims", # "dungeon"
                 )
    
    torch.manual_seed(0)
    np.random.seed(0)

    agent = Agent(obs_space_dims, action_space_dims, 
                  size=env.size,
                  load_maze=False,
                  training=False,
                  load_weights=load_weights,
                    #policy="network"
                    )
    
    reward_over_episodes = []
    nb_cells_seen_over_episodes = []

    for ep_id in range(params["episode"], params["episode"]+num_episodes):
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
        
        _, agent_pos_seen = env.get_stats()
        nb_cells_seen = len(agent_pos_seen)
        nb_cells_seen_over_episodes.append(nb_cells_seen)


    print(f"Average Reward {np.mean(reward_over_episodes):.4f} "
          f"Cells seen {np.mean(nb_cells_seen_over_episodes)} ")
     
    
    plt.plot(reward_over_episodes)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()

    w = 100
    plt.plot(np.convolve(reward_over_episodes, np.ones(w), 'valid') / w)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward over 100 episodes")
    plt.show()


if __name__ == "__main__":
    main()
