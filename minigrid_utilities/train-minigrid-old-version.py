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
import os
from minigrid_utilities.environment import MazeEnv
from minigrid_utilities.policymaker import Agent
import pickle

from minigrid_utilities.training import run_episode


def main():
    
    num_episodes = 1000
    max_num_step = 100
    step_print = 100
    size = 13
    training = True
    learning_rate = 1e-4
    param_file = "param_model2.pkl"
    reward_new_cell = 0 #.01

    load_weights_fn = f"model_weights_seed-0.pth"
    maze_seed = 2
    save_weights_fn = f"model_weights_seed-{maze_seed}.pth"

    obs_space_dims = 49
    hidden_space_dims = [16,16]
    action_space_dims = 3
    model_dims = [obs_space_dims]+hidden_space_dims+[action_space_dims]

    # TODO: I really don't like having this params dictionary. I'd rather have a "save" method
    #  to call for each class.
    params = {
        "episode": 0,
        "reward_over_episodes": [],
        "model_dims": model_dims,
        "max_num_step": max_num_step,
        "size": size,
    }

    # TODO: pickle is weird for this, we should save the runs in a csv file instead I think
    if not os.path.isfile(param_file):
        with open(param_file, 'wb') as fp:
            pickle.dump(params, fp)
        print("Saving params dict in", param_file)
    else:
        with open(param_file, 'rb') as fp:
            params = pickle.load(fp)
        print("Loaded params dict from", param_file)

    env = MazeEnv(render_mode="human",
                  size=size, 
                  maze_type="prims", # "dungeon"
                  reward_new_cell=reward_new_cell,
                  maze_seed=maze_seed,
                 )

    agent = Agent(obs_space_dims, action_space_dims,
                  size=env.size,
                  load_maze=False,
                  training=training,
                  load_weights_fn=load_weights_fn,
                  learning_rate = learning_rate,
                  #policy="network"
                  )
    
    nb_cells_seen_over_episodes = []
    nb_actions_over_episodes = []

    change_maze_at_each_episode = True

    for ep_id in range(params["episode"], params["episode"]+num_episodes):
        if change_maze_at_each_episode:
            env.reset(maze_seed=ep_id)
        else:
            env.reset()

        ep_reward = run_episode(agent, env, max_num_step)

        params["reward_over_episodes"].append(ep_reward)
        agent.update()
        
        _, agent_pos_seen, nb_actions = env.get_stats()
        nb_cells_seen = len(agent_pos_seen)
        nb_cells_seen_over_episodes.append(nb_cells_seen)
        nb_actions_over_episodes.append(nb_actions)

        if (ep_id+1)%step_print == 0:
            print(f"Episode {str(ep_id+1).rjust(3)} "
                  f"| Average Reward {np.mean(params['reward_over_episodes'][-step_print:]):.4f} "
                  f"| Cells seen {np.mean(nb_cells_seen_over_episodes[-step_print:])} "
                  f"| Number of steps {np.mean(nb_actions_over_episodes[-step_print:])}")
            
            fn = agent.save_weights(save_weights_fn)
            print("Saved model weights in", fn)
            with open(param_file, 'wb') as fp:
                pickle.dump(params, fp)
                #print("Saving params dict in", param_file)
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
