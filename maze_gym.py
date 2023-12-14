"""
Used this minigrid tutorial:
https://minigrid.farama.org/content/create_env_tutorial/

And integrated mazelib to generate the maze

TODO: rename this file as minigreedy.py
"""

from __future__ import annotations

from reinforce_utilities.environment import MiniGridMazeEnv
from reinforce_utilities.policymaker import MiniGridAgent


def main():

    num_episodes = 3
    max_num_step = 1000

    env = MiniGridMazeEnv(render_mode="human",
                          maze_gen_algo="dungeon"  #"prims"
                          )

    observation, info = env.reset(seed=0)

    obs_space_dims = observation.shape
    action_space_dims = 3

    for ep_id in range(num_episodes):
        observation, info = env.reset(seed=ep_id)

        agent = MiniGridAgent(obs_space_dims, action_space_dims,
                              size=env.size, load_maze=True,
                              policy="greedy",
                              network=False
                     )

        for i in range(max_num_step):
            action = agent.policy(observation)

            observation, reward, terminated, truncated, info = env.step(action)
            agent.rewards.append(reward)

            # Reward system:
            # success: ‘1 - 0.9 * (step_count / max_steps)’
            # failure: ‘0’

            if terminated or truncated:
                break

    env.close()

    
if __name__ == "__main__":
    main()
