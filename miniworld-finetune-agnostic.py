"""

"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt

from reinforce_utilities.training import initialize_training, fine_tune_agent, run_agent, train_agnostic_agent

import wandb
wandb.login()


def main(render_mode=None):

    # TODO: save all the parameters for one run in a txt/pickle/json file
    #  so that we don't need to write everything each time

    # number of episodes for each maze
    num_episodes_fine_tune = 1000

    max_num_step = 1000
    num_mazes = 10

    step_print = 100
    window_plot = 100 #step_print
    learning_rate = 1e-4
    discount_factor = 0.95
    reward_new_cell = 0.0 #01
    reward_closer_point = 0.0 #01

    buffer_size = 1
    memory = False

    # run_id = 101; agnostic_method = "maml"; nn_id = "384"
    run_id = 103; agnostic_method = "maml"; nn_id = "lstm"; memory = True
    run_id = 201; agnostic_method = "scratch"; nn_id = "384"
    run_id = 203; agnostic_method = "scratch"; nn_id = "lstm"; memory = True
    run_id = 205; nn_id = "3072"

    run_info = f"method-{agnostic_method}_run-{run_id}"

    run = wandb.init(project="maze-orcs", name=f"{run_info}-finetune")
    load_weights_fn = f"model_weights_{run_info}.pth"

    agent, env = initialize_training(obs_space_dims=(60, 80, 3),
                                     action_space_dims=3,
                                     maze_env="miniworld",
                                     load_weights_fn=load_weights_fn,
                                     learning_rate=learning_rate,
                                     render_mode=render_mode,
                                     buffer_size=buffer_size,
                                     reward_new_position=reward_new_cell,
                                     reward_closer_point=reward_closer_point,
                                     discount_factor=discount_factor,
                                     nn_id=nn_id,
                                     memory=memory)

    agnostic_weights = agent.get_weights(copy=True)

    print("-- Training fine-tuned models --")
    print(f"Starting finetuning evaluation of run {run_info}")

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

    run.finish()

    # Plot to see how fast it converges with one agnostic method or another
    for maze_seed in range(num_mazes):
        plt.plot(rewards_per_maze[maze_seed], c='gray', alpha=0.5)
    plt.plot(rewards_per_maze.mean(axis=0), c='red', label="Average")
    plt.legend()
    plt.title(f"Reward per episode")
    plt.savefig(f"runs_miniworld/plots/{run_info}-finetuned-rewards.png")
    plt.show()

    w = window_plot
    for maze_seed in range(num_mazes):
        plt.plot(np.convolve(rewards_per_maze[maze_seed], np.ones(w), 'valid') / w, c='gray', alpha=0.5)
    plt.plot(np.convolve(rewards_per_maze.mean(axis=0), np.ones(w), 'valid') / w, c='red', label="Average")
    plt.legend()
    plt.xlabel(f"Average Reward over {w} episodes")
    plt.savefig(f"runs_miniworld/plots/{run_info}-finetuned-average-rewards.png")
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
