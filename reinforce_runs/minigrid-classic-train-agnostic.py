"""

"""

from __future__ import annotations

import sys

from minigrid_run_setting import run_train_agnostic

import wandb

wandb.login()


def main(render_mode=None):
    use_wandb = False  # True

    num_episodes = 5000
    max_num_step = 100

    learning_rate = 1e-3
    discount_factor = 0.99
    reward_new_cell = 0.01

    buffer_size = 1
    memory = False

    agnostic_method = "classic"
    run_id = 401; nn_id = None
    # run_id = 403; nn_id = "lstm-minigrid"; memory = True

    # reload_weights = True
    reload_weights = False

    run_train_agnostic(run_id,
                       nn_id,
                       agnostic_method,
                       num_episodes=num_episodes,
                       max_num_step=max_num_step,
                       learning_rate=learning_rate,
                       discount_factor=discount_factor,
                       reward_new_cell=reward_new_cell,
                       buffer_size=buffer_size,
                       memory=memory,
                       render_mode=render_mode,
                       use_wandb=use_wandb,
                       reload_weights=reload_weights)


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
