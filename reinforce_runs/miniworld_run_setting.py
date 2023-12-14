"""

"""

from __future__ import annotations

import sys

import numpy as np

from plot_functions import plot_agnostic_training_curve, plot_finetune_training_curve
from reinforce_utilities.training import initialize_training, train_agnostic_agent, run_agent, fine_tune_agent

import wandb

wandb.login()


def run_train_agnostic(run_id,
                       nn_id,
                       agnostic_method,

                       num_episodes_per_maze=None,
                       batch_size=None,

                       num_episodes=5000,
                       max_num_step=1000,

                       step_print=100,
                       window_plot=100,
                       learning_rate=1e-4,
                       discount_factor=0.99,
                       reward_new_position=0.001,
                       reward_closer_point=0.0,

                       buffer_size=1,
                       memory=False,

                       render_mode=None,

                       use_wandb=True):

    run_info = f"method-{agnostic_method}_run-{run_id}"

    if use_wandb:
        wandb.init(project="maze-orcs", name=f"{run_info}-agnostic")

    save_agnostic_weights_fn = f"model_weights_{run_info}.pth"

    # Init from scratch
    load_weights_fn = None

    # Reinforce the model weights
    # load_weights_fn = save_agnostic_weights_fn

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
    print(f"Starting {agnostic_method} training of run {run_info}")

    stats = train_agnostic_agent(agent, env,
                                 method=agnostic_method,
                                 num_episodes=num_episodes,
                                 max_num_step=max_num_step,
                                 num_episodes_per_maze=num_episodes_per_maze,
                                 step_print=step_print,
                                 save_weights_fn=save_agnostic_weights_fn,
                                 batch_size=batch_size)

    if wandb.run is not None:
        wandb.run.finish()

    plot_agnostic_training_curve(run_info, stats,
                                 window_plot=window_plot,
                                 maze_env="miniworld")


def run_finetune_agnostic(run_id,
                          nn_id,
                          agnostic_method,

                          num_episodes_fine_tune=1000,

                          max_num_step=1000,
                          num_mazes=10,

                          step_print=100,
                          window_plot=100,
                          learning_rate=1e-4,
                          discount_factor=0.99,
                          reward_new_cell=0.0,
                          reward_closer_point=0.0,

                          buffer_size=1,
                          memory=False,

                          render_mode=None,

                          use_wandb=True):

    run_info = f"method-{agnostic_method}_run-{run_id}"

    if use_wandb:
        wandb.init(project="maze-orcs", name=f"{run_info}-finetune")
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

    if wandb.run is not None:
        wandb.run.finish()

    plot_finetune_training_curve(run_info, rewards_per_maze,
                                 window_plot=window_plot,
                                 maze_env="miniworld")
