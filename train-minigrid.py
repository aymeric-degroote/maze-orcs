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
    step_print = 100
    size = 13
    weights_fn = "model_weights.pth"
    load_weights = weights_fn
    training = True
    learning_rate = 1e-4
    param_file = "param_model1.pkl"
    reward_new_cell = 0 #.01
    
    obs_space_dims = 49
    hidden_space_dims = [16,16]
    action_space_dims = 3
    model_dims = [obs_space_dims]+hidden_space_dims+[action_space_dims]
    
    params = {
        "episode": 0,
        "reward_over_episodes": [],
        "model_dims": model_dims,
        "max_num_step": max_num_step,
        "size": size,
    }
    
    if not os.path.isfile(param_file):
        with open(param_file, 'wb') as fp:
            pickle.dump(params, fp)
        print("Saving params dict in", param_file)
    else:
        with open(param_file, 'rb') as fp:
            params = pickle.load(fp)
        print("Loaded params dict from", param_file)
    
    env = MazeEnv(#render_mode="human", 
                  size=size, 
                  maze_type="prims", # "dungeon"
                  reward_new_cell=reward_new_cell,
                 )
    
    torch.manual_seed(0)
    np.random.seed(0)

    agent = Agent(obs_space_dims, action_space_dims, 
                  size=env.size,
                  load_maze=False,
                  training=training,
                  load_weights=load_weights,
                  learning_rate = learning_rate,
                    #policy="network"
                    )
    
    nb_cells_seen_over_episodes = []

    for ep_id in range(params["episode"], params["episode"]+num_episodes):
        observation, info = env.reset(seed=ep_id)
        observation = observation.get('image')[:,:,0]
        
        #env.agent_pos = env.agent_start_pos
        

        ep_reward = 0
        for i in range(max_num_step):
            action = agent.policy(observation)

            observation, reward, terminated, truncated, info = env.step(action)
            observation = observation.get('image')[:,:,0]

            agent.rewards.append(reward)
            ep_reward += reward

            if terminated or truncated:
                break

        params["reward_over_episodes"].append(ep_reward)
        agent.update()
        
        _, agent_pos_seen = env.get_stats()
        nb_cells_seen = len(agent_pos_seen)
        nb_cells_seen_over_episodes.append(nb_cells_seen)


        if (ep_id+1)%step_print == 0:
            print(f"Episode {str(ep_id+1).rjust(3)}: "
                  f"Average Reward {np.mean(params['reward_over_episodes'][-step_print:]):.4f} "
                  f"Cells seen {np.mean(nb_cells_seen_over_episodes[-step_print:])} ")
            
            fn = agent.save_weights(weights_fn)
            print("Saved model weights in", fn)
            with open(param_file, 'wb') as fp:
                pickle.dump(params, fp)
                print("Saving params dict in", param_file)
            #print("positions:", agent_pos_seen)
            #print("dist:", agent.network_policy(observation, get_dist=True))
            
        
        params["episode"] += 1
    
    
    plt.plot(params["reward_over_episodes"])
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()

    w = step_print
    plt.plot(np.convolve(params["reward_over_episodes"], np.ones(w), 'valid') / w)
    plt.xlabel(f"Average Reward over {w} episodes")
    plt.ylabel("Reward")
    plt.show()


if __name__ == "__main__":
    main()
