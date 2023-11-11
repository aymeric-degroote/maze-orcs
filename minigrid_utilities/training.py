import numpy as np

from minigrid_utilities.environment import MazeEnv
from minigrid_utilities.policymaker import Agent


def initialize_training(obs_space_dims,
                        action_space_dims,
                        size,
                        load_weights_fn=None,
                        render_mode=None,
                        learning_rate=None,
                        reward_new_cell=None,
                        maze_type=None,
                        **kwargs):
    # TODO: not sure if it is good practice to play around with kwargs
    if learning_rate is not None:
        kwargs["learning_rate"] = learning_rate

    agent = Agent(obs_space_dims, action_space_dims,
                  size=size,
                  load_maze=False,
                  training=True,
                  load_weights_fn=load_weights_fn,
                  **kwargs
                  )
    kwargs = dict()
    if reward_new_cell is not None:
        kwargs["reward_new_cell"] = reward_new_cell
    if maze_type is not None:
        kwargs["maze_type"] = maze_type

    env = MazeEnv(render_mode=render_mode,
                  size=size,
                  **kwargs)

    return agent, env


def train_agnostic_agent(agent, env, method,
                         num_episodes,
                         max_num_step,
                         step_print=None,
                         save_weights_fn=None,
                         batch_size=None
                         ):
    """
    Methods:
        - classic: each episode is a different maze and agent is updated at each episode
        - batch: each episode is a different maze and agent is updated every batch_size episodes (could be parallelized)
        - MAML: Model-Agnostic Meta-Learning algorithm
    """

    if method == "classic":
        return run_agent(agent, env, num_episodes, max_num_step,
                         change_maze_at_each_episode=True,
                         training=True,
                         step_print=step_print,
                         save_weights_fn=save_weights_fn)
    elif method == "batch":
        return run_agent(agent, env, num_episodes, max_num_step,
                         change_maze_at_each_episode=True,
                         training=True,
                         step_print=step_print,
                         save_weights_fn=save_weights_fn,
                         batch_size=batch_size)
    elif method == "MAML":
        raise NotImplementedError
    else:
        raise NameError(f"method '{method}' unknown")


def fine_tune_agent(agent, env, maze_seed,
                    num_episodes,
                    max_num_step,
                    step_print=None,
                    save_weights_fn=None
                    ):
    env.reset(maze_seed=maze_seed)

    return run_agent(agent, env, num_episodes, max_num_step,
                     change_maze_at_each_episode=False,
                     training=True,
                     step_print=step_print,
                     save_weights_fn=save_weights_fn)


def run_episode(agent, env, max_num_step):
    observation = env.gen_obs().get('image')[:, :, 0]

    ep_reward = 0
    for i in range(max_num_step):
        action = agent.policy(observation)

        observation, reward, terminated, truncated, info = env.step(action)
        observation = observation.get('image')[:, :, 0]

        agent.rewards.append(reward)
        ep_reward += reward

        if terminated or truncated:
            break

    return ep_reward


def run_agent(agent, env, num_episodes, max_num_step, change_maze_at_each_episode,
              training=True, step_print=None, save_weights_fn=None,
              batch_size=None):
    assert training == agent.training, f"Agent is in {'training' if agent.training else 'eval'} mode " \
                                       f"but you are running in {'training' if training else 'eval'} mode"

    stats = {
        "reward_over_episodes": [],
        "nb_cells_seen_over_episodes": [],
        "nb_actions_over_episodes": [],
    }

    for ep_id in range(num_episodes):
        if change_maze_at_each_episode:
            env.reset(maze_seed=ep_id)
        else:
            env.reset()

        ep_reward = run_episode(agent, env, max_num_step)

        stats["reward_over_episodes"].append(ep_reward)
        if training:
            if batch_size is None or (ep_id + 1) % batch_size == 0:
                agent.update()

        _, agent_pos_seen, nb_actions = env.get_stats()
        nb_cells_seen = len(agent_pos_seen)
        stats["nb_cells_seen_over_episodes"].append(nb_cells_seen)
        stats["nb_actions_over_episodes"].append(nb_actions)

        if step_print and (ep_id + 1) % step_print == 0:
            print(f"Episode {str(ep_id + 1).rjust(3)} "
                  f"| Average Reward {np.mean(stats['reward_over_episodes'][-step_print:]):.4f} "
                  f"| Cells seen {np.mean(stats['nb_cells_seen_over_episodes'][-step_print:])} "
                  f"| Number of steps {np.mean(stats['nb_actions_over_episodes'][-step_print:])}")

            if training and save_weights_fn:
                fn = agent.save_weights(save_weights_fn)
                print("saved model weights in", fn)

    return stats
