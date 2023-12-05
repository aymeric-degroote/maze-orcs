"""

"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt

from reinforce_utilities.training import initialize_training, train_agnostic_agent, run_agent

import wandb
wandb.login()

def main(render_mode=None):

    num_episodes = 5000
    max_num_step = 1000

    step_print = 100
    window_plot = 100 #step_print
    learning_rate = 1e-4
    discount_factor = 0.95
    reward_new_position = 0.001
    reward_closer_point = 0.0 #01

    buffer_size = 1
    memory = False

    agnostic_method = "scratch"
    run_id = 201; nn_id = "384"
    #run_id = 203; nn_id = "lstm"; memory = True
    run_id = 205; nn_id = "3072"

    run_info = f"method-{agnostic_method}_run-{run_id}"

    wandb.init(project="maze-orcs", name=f"{run_info}-agnostic")

    save_agnostic_weights_fn = f"model_weights_{run_info}.pth"

    # Init from scratch
    load_weights_fn = None

    # Reinforce the model weights
    #load_weights_fn = save_agnostic_weights_fn

    agent, env = initialize_training(obs_space_dims=(60, 80, 3),
                                     action_space_dims=3,
                                     maze_env="miniworld",
                                     load_weights_fn=load_weights_fn,
                                     learning_rate=learning_rate,
                                     render_mode=render_mode,
                                     buffer_size=buffer_size,
                                     reward_new_position=reward_new_position,
                                     reward_closer_point=reward_closer_point,
                                     discount_factor=discount_factor,
                                     nn_id=nn_id,
                                     memory=memory)

    print("-- Training model --")
    print(f"Starting scratch training of run {run_info}")

    stats = run_agent(agent, env,
                      num_episodes=num_episodes,
                      max_num_step=max_num_step,
                      change_maze_at_each_episode=True,
                      training=True,
                      step_print=step_print,
                      save_weights_fn=save_agnostic_weights_fn)

    if wandb.run is not None:
        wandb.run.finish()

    # Plot to see how fast it converges with one agnostic method or another
    plt.plot(stats["reward_over_episodes"], c='gray', alpha=0.5)
    plt.title(f"Reward per episode")
    plt.savefig(f"runs_miniworld/plots/{run_info}-agnostic-rewards.png")
    plt.show()

    w = window_plot
    plt.plot(np.convolve(stats["reward_over_episodes"], np.ones(w), 'valid') / w, c='gray', alpha=0.5)
    # plt.plot(np.convolve(rewards_per_maze.mean(axis=0), np.ones(w), 'valid') / w, c='red', label="Average")
    # plt.legend()
    plt.xlabel(f"Average Reward over {w} episodes")
    plt.savefig(f"runs_miniworld/plots/{run_info}-agnostic-average-rewards.png")
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
