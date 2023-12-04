
from copy import deepcopy

import numpy as np
import torch

from reinforce_utilities.environment import MiniGridMazeEnv, MiniWorldMazeEnv
from reinforce_utilities.policymaker import Agent, MiniGridAgent, MiniWorldAgent

import gymnasium as gym
import miniworld

from utilities.wrappers import WarpFrame, PyTorchFrame


def initialize_training(obs_space_dims,
                        action_space_dims,
                        size=13,
                        maze_env=None,
                        load_weights_fn=None,
                        render_mode=None,
                        learning_rate=None,
                        reward_new_cell=None,
                        reward_closer_point=None,
                        maze_gen_algo=None,
                        buffer_size=None,
                        discount_factor=None,
                        **agent_kwargs):
    if maze_env is None:
        maze_env = "minigrid"

    # TODO: not sure if it is good practice to play around with kwargs
    if learning_rate is not None:
        agent_kwargs["learning_rate"] = learning_rate
    if buffer_size is not None:
        agent_kwargs["buffer_size"] = buffer_size
    if discount_factor is not None:
        agent_kwargs["discount_factor"] = discount_factor

    env_kwargs = dict()
    if reward_new_cell is not None:
        env_kwargs["reward_new_cell"] = reward_new_cell
    if maze_gen_algo is not None:
        env_kwargs["maze_gen_algo"] = maze_gen_algo

    if maze_env.lower() == "minigrid":
        agent = MiniGridAgent(obs_space_dims, action_space_dims,
                  size=size,
                  maze_env=maze_env,
                  load_maze=False,
                  training=True,
                  load_weights_fn=load_weights_fn,
                  **agent_kwargs
                  )
        env = MiniGridMazeEnv(render_mode=render_mode,
                              size=size,
                              **env_kwargs)
    elif maze_env.lower() == "miniworld":
        if reward_closer_point is not None:
            env_kwargs["reward_closer_point"] = reward_closer_point
        agent = MiniWorldAgent(obs_space_dims, action_space_dims,
                               maze_env=maze_env,
                               load_maze=False,
                               training=True,
                               load_weights_fn=load_weights_fn,
                               **agent_kwargs
                               )

        env = MiniWorldMazeEnv(render_mode=render_mode,
                               size=size,
                               **env_kwargs)
    else:
        raise NameError

    return agent, env


def train_agnostic_agent(agent, env, method,
                         num_episodes,
                         max_num_step,
                         num_episodes_per_maze=None,
                         step_print=None,
                         save_weights_fn=None,
                         batch_size=None,
                         ):
    """
    Methods:
        - classic: each episode is a different maze and agent is updated at each episode
        - batch: each episode is a different maze and agent is updated every batch_size episodes (could be parallelized)
        - MAML: Model-Agnostic Meta-Learning algorithm
    """

    if method == "scratch":
        # no training at all, so the model is undoubtedly agnostic
        return init_stats()
    elif method == "classic":
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
    elif method.lower() == "maml":
        assert num_episodes_per_maze is not None, "arg 'num_episodes_per_maze' missing"
        return run_maml_agent(agent, env, num_episodes, max_num_step,
                              num_episodes_per_maze,
                              batch_size,
                              # alpha_lr=None, beta_lr=None,
                              save_weights_fn=save_weights_fn,
                              )
    else:
        raise NameError(f"method '{method}' unknown")


def fine_tune_agent(agent, env, maze_seed,
                    num_episodes,
                    max_num_step,
                    step_print=None,
                    save_weights_fn=None,
                    ):
    env.reset(maze_seed=maze_seed)

    return run_agent(agent, env, num_episodes, max_num_step,
                     change_maze_at_each_episode=False,
                     training=True,
                     step_print=step_print,
                     save_weights_fn=save_weights_fn)


def run_episode(agent, env, max_num_step):
    observation = env.gen_obs()

    ep_reward = 0
    for i in range(max_num_step):
        action = agent.policy(observation)

        observation, reward, terminated, truncated, info = env.step(action)

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

    stats = init_stats()

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

        env_stats = env.get_stats()
        agent_pos_seen = env_stats.get("agent_pos_seen")
        nb_actions = env_stats.get("nb_actions")

        if agent_pos_seen:
            nb_cells_seen = len(agent_pos_seen)
        else:
            nb_cells_seen = 0
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


def run_maml_agent(agent, env, num_episodes, max_num_step, num_episodes_per_maze,
                   batch_size, alpha_lr=None, beta_lr=0.01,
                   step_print=None, save_weights_fn=None,
                   ):
    """
    Implementation of Algorithm 3 from: https://arxiv.org/abs/1703.03400
    H: max_num_step
    K: num_episodes_fine_tune

    """
    assert agent.training, f"Agent is in eval mode"

    num_batches = num_episodes // (num_episodes_per_maze * batch_size)
    maze_seeds = [-1]

    stats = init_stats()

    for batch_id in range(num_batches):
        print(f"-- Batch {batch_id} --")
        agnostic_weights = agent.get_weights(copy=True)

        batch_rewards = []

        # TODO: do we want to have new mazes for each batch? I assumed so
        # TODO: use iter function to generate the seeds
        maze_seeds = list(range(maze_seeds[-1]+1, maze_seeds[-1]+1+batch_size))

        ft_weights_grad_list = {}

        for maze_seed in maze_seeds:
            print(f"-- Maze seed #{maze_seed} --", end="\r")
            env.reset(maze_seed=maze_seed)

            agent.set_weights(agnostic_weights)
            maze_reward = 0
            for ep_id in range(num_episodes_per_maze//2):
                ep_reward = run_episode(agent, env, max_num_step)
                maze_reward += ep_reward

                stats["reward_over_episodes"].append(ep_reward)

            # update agent after K episodes
            # technically summing the loss whereas the paper take the average...
            agent.update()  # TODO: use alpha_lr

            for ep_id in range(num_episodes_per_maze//2):
                ep_reward = run_episode(agent, env, max_num_step)
                maze_reward += ep_reward

                stats["reward_over_episodes"].append(ep_reward)

            batch_rewards.append(maze_reward/num_episodes_per_maze)
            #batch_rewards.append(ep_reward)

            loss = agent.compute_loss()
            loss.backward()

            # TODO: the following should be in the Agent class
            # state_dict doesn't return the grad tensors
            for weight_name, tensor in enumerate(agent.net.parameters()):
                if weight_name not in ft_weights_grad_list:
                    ft_weights_grad_list[weight_name] = []

                ft_weights_grad = deepcopy(tensor.grad)
                ft_weights_grad_list[weight_name].append(ft_weights_grad)
                tensor.grad.zero_()

            agent.log_probs = []
            agent.rewards = []

        agent.set_weights(agnostic_weights)
        for weight_name, param in enumerate(agent.net.parameters()):
            # Important that it's not a copy here
            grad_tensor = sum(ft_weights_grad_list[weight_name])
            with torch.no_grad():
                #print('beta', beta_lr, '      ')
                #print('grad', grad_tensor)
                param -= beta_lr * grad_tensor

        print(f"Batch #{str(batch_id).rjust(3)} "
              f"| Average Reward {np.mean(batch_rewards):.4f}")

        if save_weights_fn:
            fn = agent.save_weights(save_weights_fn)
            print("saved model weights in", fn)

    return stats


def init_stats():
    return {
        "reward_over_episodes": [],
        "nb_cells_seen_over_episodes": [],
        "nb_actions_over_episodes": [],
    }
