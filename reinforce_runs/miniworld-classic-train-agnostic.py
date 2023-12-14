"""

"""

from __future__ import annotations

import sys

from miniworld_run_setting import run_train_agnostic

import wandb

wandb.login()


def main(render_mode=None):
    use_wandb = False

    num_episodes = 10000
    max_num_step = 1000

    learning_rate = 1e-4
    discount_factor = 0.99
    reward_new_position = 0.01
    reward_closer_point = 0.0  # 01

    buffer_size = 1
    memory = False

    # TODO: save run params to some file
    agnostic_method = "classic"
    run_id = 201; nn_id = "384"
    # run_id = 203; nn_id = "lstm"; memory = True
    # run_id = 205; nn_id = "3072"
    # run_id = 207; nn_id = "3072"; discount_factor = 1.

    run_train_agnostic(run_id,
                       nn_id,
                       agnostic_method,
                       num_episodes=num_episodes,
                       max_num_step=max_num_step,
                       learning_rate=learning_rate,
                       discount_factor=discount_factor,
                       reward_new_position=reward_new_position,
                       reward_closer_point=reward_closer_point,
                       buffer_size=buffer_size,
                       memory=memory,
                       render_mode=render_mode,
                       use_wandb=use_wandb)


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
