"""

"""

from __future__ import annotations

import sys

from minigrid_run_setting import run_finetune_agnostic

import wandb
wandb.login()


def main(render_mode=None):

    use_wandb = False

    # number of episodes for each maze
    num_episodes_fine_tune = 1000

    max_num_step = 300
    num_mazes = 10

    learning_rate = 1e-3
    discount_factor = 0.99
    reward_new_cell = 0.0 #1

    buffer_size = 1
    memory = False

    #run_id = 403; agnostic_method = "classic"; nn_id = "lstm-minigrid"; memory = True
    #run_id = 401; agnostic_method = "classic"; nn_id = None

    run_id = 501; agnostic_method = "maml"; nn_id = None

    run_finetune_agnostic(run_id,
                          nn_id,
                          agnostic_method,
                          num_episodes_fine_tune=num_episodes_fine_tune,
                          max_num_step=max_num_step,
                          num_mazes=num_mazes,
                          learning_rate=learning_rate,
                          discount_factor=discount_factor,
                          reward_new_cell=reward_new_cell,
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
