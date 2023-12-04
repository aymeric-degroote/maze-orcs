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

from reinforce_utilities.training import train_agnostic_agent, initialize_training, fine_tune_agent


def main(render_mode=None):
    num_episodes = 10000
    num_episodes_per_maze = 100 # for MAML method
    batch_size = 5
    # num_batches = num_episodes // (num_episodes_per_maze * batch_size)
    num_episodes_fine_tune = 1000

    max_num_step = 100
    num_mazes = 10 # for testing

    step_print = 100
    window_plot = 10 #step_print
    size = 13
    learning_rate = 1e-4
    reward_new_cell = 0.01

    #agnostic_method = "batch"
    agnostic_method = "maml"

    run_id = 50
    save_agnostic_weights_fn = f"model_weights_method-{agnostic_method}_run-{run_id}.pth"
    load_weights_fn = save_agnostic_weights_fn   # None
    # TODO. save the config of the run in a file

    obs_space_dims = 49
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
                                     reward_new_cell=reward_new_cell)

    print("-- Training agnostic model --")
    stats = train_agnostic_agent(agent, env,
                                 method=agnostic_method,
                                 num_episodes=num_episodes,
                                 max_num_step=max_num_step,
                                 num_episodes_per_maze=num_episodes_per_maze,
                                 step_print=step_print,
                                 save_weights_fn=save_agnostic_weights_fn,
                                 batch_size=batch_size)

    plt.plot(stats["reward_over_episodes"])
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig(f"runs_minigrid/plots/agnostic-{agnostic_method}-agent-rewards.png")
    plt.show()

    w = step_print
    plt.plot(np.convolve(stats["reward_over_episodes"], np.ones(w), 'valid') / w)
    plt.xlabel(f"Average Reward over {w} episodes")
    plt.ylabel("Reward")
    plt.savefig(f"runs_minigrid/plots/agnostic-{agnostic_method}-agent-average-rewards.png")
    plt.show()

    agnostic_weights = agent.get_weights(copy=True)

    # TODO: the next part is actually testing of the agnostic model. Could be in a separate file

    print("-- Training fine-tuned models --")
    rewards_per_maze = np.zeros((num_mazes, num_episodes_fine_tune))

    for maze_seed in range(num_mazes):
        print(f"-- Maze seed {maze_seed} --")

        # Reset the weights to those of the agnostic model
        agent.set_weights(agnostic_weights)

        ft_save_weights_fn = f"model_weights_method-{agnostic_method}_run-{run_id}_seed-{maze_seed}.pth"

        stats = fine_tune_agent(agent, env, maze_seed=maze_seed,
                                num_episodes=num_episodes_fine_tune,
                                max_num_step=max_num_step,
                                step_print=step_print,
                                save_weights_fn=ft_save_weights_fn)
        rewards_per_maze[maze_seed] = stats["reward_over_episodes"]

    # Plot to see how fast it converges with one agnostic method or another
    for maze_seed in range(num_mazes):
        plt.plot(rewards_per_maze[maze_seed], c='gray', alpha=0.5)
    plt.plot(rewards_per_maze.mean(axis=0), c='red', label="Average")
    plt.legend()
    plt.title(f"Reward per episode")
    plt.savefig(f"runs_minigrid/plots/fine-tuned-{agnostic_method}-agents-rewards.png")
    plt.show()

    w = window_plot
    for maze_seed in range(num_mazes):
        plt.plot(np.convolve(rewards_per_maze[maze_seed], np.ones(w), 'valid') / w, c='gray', alpha=0.5)
    plt.plot(np.convolve(rewards_per_maze.mean(axis=0), np.ones(w), 'valid') / w, c='red', label="Average")
    plt.legend()
    plt.xlabel(f"Average Reward over {w} episodes")
    plt.savefig(f"runs_minigrid/plots/fine-tuned-{agnostic_method}-agent-average-rewards.png")
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
