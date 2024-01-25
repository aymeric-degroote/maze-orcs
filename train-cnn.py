"""

"""

from __future__ import annotations

import sys

from run_setting import run_train_agnostic


def main(render_mode=None):

    num_episodes = 10000
    max_num_step = 1000

    learning_rate = 1e-4
    discount_factor = 0.99
    reward_new_position = 0.01
    reward_closer_point = 0.0  # 01

    buffer_size = 1
    memory = True

    # TODO: save run params to some file
    agnostic_method = "classic"
    run_id = "train-cnn"; nn_id = "384"
    run_id = "train-lstm"; nn_id = "lstm"

    reload_weights = True
    load_cnn = True

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
                       reload_weights=reload_weights,
                       load_cnn=load_cnn)


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
