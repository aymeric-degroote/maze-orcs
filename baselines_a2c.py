from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import numpy as np
from dqn_utilities.larotta_maze import PedroMaze, MiniWorldMazeEnv

from dqn_utilities.wrappers import WarpFrame, PyTorchFrame, BetterReward, MaxAndSkipEnv, FrameStack, ClipRewardEnv



# Parallel environments
# vec_env = make_vec_env("MiniWorld-Maze-v0", n_envs=4)
# env = MiniWorldMazeEnv(reward_closer_point=1,)
# env = gym.make("MiniWorld-Maze-v0", 
#                num_rows=4, 
#                num_cols=4, 
#                room_size=3, 
#                render_mode='human',
#                max_episode_steps = 500,
#                view='top')
env = PedroMaze(num_rows=3, 
               num_cols=3, 
               room_size=2, 
               render_mode='human',
               max_episode_steps = 750,
               view='top',
               rng = np.random.default_rng(seed = 1))
env = WarpFrame(env)
# env = BetterReward(env)
env = FrameStack(env, k=8)
# env = ClipRewardEnv(env)
# vec_env = make_vec_env("MiniWorld-Maze-v0", n_envs=4)

model = A2C("CnnPolicy", env, verbose=1, learning_rate=0.0001)
model.learn(total_timesteps=1000000)
model.save("a2c_miniworld")

# del model # remove to demonstrate saving and loading

# model = A2C.load("a2c_cartpole")
print('DONE')

obs,_ = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones,_, info = env.step(action)
    env.render()