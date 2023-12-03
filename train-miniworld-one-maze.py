"""

"""

from __future__ import annotations

import sys

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import miniworld

from utilities.wrappers import WarpFrame, PyTorchFrame
from utilities.agent import DQNAgent
from utilities.memory import ReplayBuffer

import torch

from minigrid_utilities.training import initialize_training, fine_tune_agent, run_agent


def main(render_mode=None):

    maze_env = "minigrid"
    obs_space_dims = (7,7)
    maze_env = "miniworld"
    obs_space_dims = (60, 80, 3)

    num_episodes_fine_tune = 500

    max_num_step = 1000

    step_print = 100
    window_plot = 10 #step_print
    size = 13
    learning_rate = 1e-4
    discount_factor = 0.9
    reward_new_cell = 0.01
    reward_closer_point = 0.01

    buffer_size = 1
    memory = False

    maze_seed = 3
    agnostic_method = "scratch"  #"batch"  # "maml"
    #run_id = 52; nn_id = "384"
    run_id = 56; nn_id = "3072"
    run_id = 72; nn_id = "384"; buffer_size=5
    run_id = 69; nn_id = "lstm"; buffer_size = 5
    run_id = 77; nn_id = "lstm"; buffer_size = 1; memory = True

    torch.autograd.set_detect_anomaly(True)

    if buffer_size is None:
        ft_save_weights_fn = f"model_weights_method-{agnostic_method}_run-{run_id}_seed-{maze_seed}.pth"
    else:
        ft_save_weights_fn = f"model_weights_method-{agnostic_method}_run-{run_id}_seed-{maze_seed}-buffer-{buffer_size}.pth"
    
    # Init from scratch
    load_weights_fn = None

    # Agnostic
    # load_weights_fn = f"model_weights_method-{agnostic_method}_run-{run_id}.pth"

    # Reinforce
    #load_weights_fn = ft_save_weights_fn

    hidden_space_dims = [16, 16]
    action_space_dims = 3
    # TODO: use model_dims to define the NN for the agent policy. or not. idk

    agent, env = initialize_training(obs_space_dims,
                                     action_space_dims,
                                     maze_env=maze_env,
                                     size=size,
                                     load_weights_fn=load_weights_fn,
                                     learning_rate=learning_rate,
                                     render_mode=render_mode,
                                     buffer_size=buffer_size,
                                     reward_new_cell=reward_new_cell,
                                     reward_closer_point=reward_closer_point,
                                     discount_factor=discount_factor,
                                     nn_id=nn_id,
                                     memory=memory)

    print("-- Training model --")
    rewards_per_maze = np.zeros((1, num_episodes_fine_tune))

    print(f"-- Maze seed {maze_seed} --")
    env.reset(maze_seed=maze_seed)

    stats = run_agent(agent, env,
                      num_episodes=num_episodes_fine_tune,
                      max_num_step=max_num_step,
                      change_maze_at_each_episode=False,
                      training=True,
                      step_print=step_print,
                      save_weights_fn=ft_save_weights_fn,
                      retain_graph=None or memory)

    rewards_per_maze[0] = stats["reward_over_episodes"]

    # Plot to see how fast it converges with one agnostic method or another
    plt.plot(rewards_per_maze[0], c='gray', alpha=0.5)
    # plt.legend()
    plt.title(f"Reward per episode")
    # plt.savefig(f"runs_minigrid/plots/fine-tuned-{agnostic_method}-agents-rewards-seed-{maze_seed}.png")
    plt.show()

    w = window_plot
    plt.plot(np.convolve(rewards_per_maze[0], np.ones(w), 'valid') / w, c='gray', alpha=0.5)
    # plt.plot(np.convolve(rewards_per_maze.mean(axis=0), np.ones(w), 'valid') / w, c='red', label="Average")
    # plt.legend()
    plt.xlabel(f"Average Reward over {w} episodes")
    # plt.savefig(f"runs_minigrid/plots/fine-tuned-{agnostic_method}-agent-average-rewards-seed-{maze_seed}.png")
    plt.show()


if __name__ == "__main__":
    # TODO: add all possible arguments
    # -seed seed
    # -render human
    # -num_episodes num_episodes
    # -method maml

    if len(sys.argv) > 1:
        render_mode = sys.argv[1]
        if render_mode == "human":
            main(render_mode)
    else:
        main()
