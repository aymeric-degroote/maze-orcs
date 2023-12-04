"""

"""

from __future__ import annotations

import sys

import numpy as np
import matplotlib.pyplot as plt

from reinforce_utilities.training import initialize_training, train_agnostic_agent


def main(render_mode=None):

    num_episodes = 1000
    num_episodes_per_maze = 10  # for MAML method
    batch_size = 5
    # num_batches = num_episodes // (num_episodes_per_maze * batch_size)

    max_num_step = 100

    step_print = 1
    window_plot = 100 #step_print
    learning_rate = 1e-4
    discount_factor = 0.95
    reward_new_cell = 0.01
    reward_closer_point = 0.01

    buffer_size = 1
    memory = False

    agnostic_method = "maml"
    run_id = 101; nn_id = "384"
    # run_id = 103; nn_id = "lstm"; memory = True

    run_info = f"method-{agnostic_method}_run-{run_id}"
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
                                     reward_new_cell=reward_new_cell,
                                     reward_closer_point=reward_closer_point,
                                     discount_factor=discount_factor,
                                     nn_id=nn_id,
                                     memory=memory)

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
    plt.savefig(f"runs_minigrid/plots/{run_info}-agnostic-rewards.png")
    plt.show()

    w = window_plot
    plt.plot(np.convolve(stats["reward_over_episodes"], np.ones(w), 'valid') / w)
    plt.xlabel(f"Average Reward over {w} episodes")
    plt.ylabel("Reward")
    plt.savefig(f"runs_minigrid/plots/{run_info}-agnostic-average-rewards.png")
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
