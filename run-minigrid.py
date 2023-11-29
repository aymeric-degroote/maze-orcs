"""
Used this minigrid tutorial:
https://minigrid.farama.org/content/create_env_tutorial/

And integrated mazelib to generate the maze

Used this tutorial for RL agent:
https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/

"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt

from minigrid_utilities.environment import MazeEnv
from minigrid_utilities.policymaker import Agent

from minigrid_utilities.training import run_agent, initialize_training


def main(render_mode=None):
    num_episodes = 1000
    max_num_step = 100

    size = 13
    window_size = 100

    maze_seed = 3
    agnostic_method = "batch"
    run_id = 42

    weights_fn = f"model_weights_method-{agnostic_method}_run-{run_id}_seed-{maze_seed}.pth"

    #weights_fn = "model_weights_seed-0.pth"
    load_weights = weights_fn

    change_maze_at_each_episode = (maze_seed is None)

    obs_space_dims = 49
    action_space_dims = 3

    env = MazeEnv(render_mode=render_mode,
                  size=size,
                  maze_type="prims",  # "dungeon"
                  maze_seed=maze_seed,
                  )

    agent = Agent(obs_space_dims, action_space_dims,
                  size=env.size,
                  load_maze=False,
                  training=False,
                  load_weights_fn=load_weights,
                  )

    stats = run_agent(agent, env, num_episodes, max_num_step,
                      change_maze_at_each_episode=change_maze_at_each_episode,
                      training=False)

    reward_over_episodes = stats["reward_over_episodes"]
    nb_cells_seen_over_episodes = stats["nb_cells_seen"]
    nb_actions_over_episodes = stats["nb_actions"]

    print(f"Average Reward {np.mean(reward_over_episodes):.4f}")
    print(f"Cells seen {np.mean(nb_cells_seen_over_episodes):.2f}")
    print(f"Actions taken {np.mean(nb_actions_over_episodes):.2f}")

    plt.plot(reward_over_episodes)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()

    w = window_size
    plt.plot(np.convolve(reward_over_episodes, np.ones(w), 'valid') / w)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward over 100 episodes")
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        render_mode = sys.argv[1]
        if render_mode == "human":
            main(render_mode)
    else:
        main()
