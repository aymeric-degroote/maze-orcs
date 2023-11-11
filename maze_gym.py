"""
Used this minigrid tutorial:
https://minigrid.farama.org/content/create_env_tutorial/

And integrated mazelib to generate the maze

TODO: remove this file that is pretty useless
"""

from __future__ import annotations

from minigrid_utilities.environment import MazeEnv
from minigrid_utilities.policymaker import Agent


def main():

    num_episodes = 2
    max_num_step = 1000

    env = MazeEnv(render_mode="human",
                  maze_type="dungeon" #"prims"
                 )

    observation, info = env.reset(seed=0)
    observation = observation.get('image')[:,:,0]
    # TODO: either integrate [:,:,0] in reset function or use the whole image
    # Doesn't really matter since we will switch to MiniWorld at some point

    obs_space_dims = observation.shape
    action_space_dims = 3

    for ep_id in range(num_episodes):
        observation, info = env.reset(seed=ep_id)
        observation = observation.get('image')[:,:,0]

        agent = Agent(obs_space_dims, action_space_dims,
                      size=env.size, load_maze=True,
                      policy="greedy"
                     )

        for i in range(max_num_step):
            action = agent.policy(observation)

            observation, reward, terminated, truncated, info = env.step(action)
            observation = observation.get('image')[:,:,0]
            agent.rewards.append(reward)

            # Reward system:
            # success: ‘1 - 0.9 * (step_count / max_steps)’
            # failure: ‘0’

            if terminated or truncated:
                break

    env.close()

    
if __name__ == "__main__":
    main()
