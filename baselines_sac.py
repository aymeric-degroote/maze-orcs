from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import numpy as np
from dqn_utilities.larotta_maze import PedroMaze, MiniWorldMazeEnv

from dqn_utilities.wrappers import WarpFrame, PyTorchFrame, BetterReward, MaxAndSkipEnv, FrameStack, ClipRewardEnv
from stable_baselines3 import SAC


# Parallel environments
# vec_env = make_vec_env("MiniWorld-Maze-v0", n_envs=4)
# env = MiniWorldMazeEnv(reward_closer_point=1,)
env = gym.make("MiniWorld-Maze-v0", 
               num_rows=3, 
               num_cols=3, 
               room_size=2, 
               render_mode='human',
               max_episode_steps = 500,
               view='top')
env = WarpFrame(env)
# env = BetterReward(env)
env = FrameStack(env, k=4)
env = ClipRewardEnv(env)
# vec_env = make_vec_env("MiniWorld-Maze-v0", n_envs=4)

model = SAC("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("sac_miniworld")

# del model # remove to demonstrate saving and loading

# model = A2C.load("a2c_cartpole")
print('DONE')

obs,_ = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones,_, info = env.step(action)
    env.render("human")