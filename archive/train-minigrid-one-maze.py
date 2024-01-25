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

from reinforce_utilities.training import initialize_training, fine_tune_agent, run_agent


def main(render_mode=None):

    num_episodes = 10000
    num_episodes_per_maze = 100 # for MAML method
    batch_size = 5
    # num_batches = num_episodes // (num_episodes_per_maze * batch_size)
    num_episodes_fine_tune = 1000

    max_num_step = 100

    step_print = 100
    window_plot = 10 #step_print
    size = 13
    learning_rate = 1e-4
    reward_new_cell = 0.01
    discount_factor = 0.95

    buffer_size = 15

    maze_seed = 3
    agnostic_method = "scratch"  #"batch"  # "maml"
    run_id = 50

    if buffer_size is None:
        ft_save_weights_fn = f"model_weights_method-{agnostic_method}_run-{run_id}_seed-{maze_seed}.pth"
    else:
        ft_save_weights_fn = f"model_weights_method-{agnostic_method}_run-{run_id}_seed-{maze_seed}-buffer-{buffer_size}.pth"

    # Init from scratch
    #load_weights_fn = None

    # Agnostic
    # load_weights_fn = f"model_weights_method-{agnostic_method}_run-{run_id}.pth"

    # Reinforce
    load_weights_fn = ft_save_weights_fn


    obs_space_dims = (7,7)
    hidden_space_dims = [16, 16]
    action_space_dims = 3
    # TODO: use model_dims to define the NN for the agent policy. or not. idk
    model_dims = [obs_space_dims] + hidden_space_dims + [action_space_dims]

    agent, env = initialize_training(obs_space_dims,
                                     action_space_dims,
                                     size=size,
                                     load_weights_fn=load_weights_fn,
                                     learning_rate=learning_rate,
                                     render_mode=render_mode,
                                     reward_new_position=reward_new_cell,
                                     buffer_size=buffer_size)

    agent.gamma = discount_factor

    print("-- Training fine-tuned model --")
    rewards_per_maze = np.zeros((1, num_episodes_fine_tune))

    print(f"-- Maze seed {maze_seed} --")
    env.reset(maze_seed=maze_seed)

    stats = run_agent(agent, env, num_episodes=num_episodes_fine_tune,
                      max_num_step = max_num_step,
                      change_maze_at_each_episode=False,
                      training=True,
                      step_print=step_print,
                      save_weights_fn=ft_save_weights_fn)

    rewards_per_maze[0] = stats["reward_over_episodes"]

    # Plot to see how fast it converges with one agnostic method or another
    plt.plot(rewards_per_maze[0], c='gray', alpha=0.5)
    #plt.legend()
    plt.title(f"Reward per episode")
    # plt.savefig(f"runs_minigrid/plots/fine-tuned-{agnostic_method}-agents-rewards-seed-{maze_seed}.png")
    plt.show()

    w = window_plot
    plt.plot(np.convolve(rewards_per_maze[0], np.ones(w), 'valid') / w, c='gray', alpha=0.5)
    # plt.plot(np.convolve(rewards_per_maze.mean(axis=0), np.ones(w), 'valid') / w, c='red', label="Average")
    #plt.legend()
    plt.xlabel(f"Average Reward over {w} episodes")
    #plt.savefig(f"runs_minigrid/plots/fine-tuned-{agnostic_method}-agent-average-rewards-seed-{maze_seed}.png")
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
