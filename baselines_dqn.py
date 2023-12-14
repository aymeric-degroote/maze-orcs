import gymnasium as gym
import numpy as np
from dqn_utilities.larotta_maze import PedroMaze, MiniWorldMazeEnv

from dqn_utilities.wrappers import WarpFrame, PyTorchFrame, BetterReward, MaxAndSkipEnv, FrameStack, ClipRewardEnv

from stable_baselines3 import DQN

# env = PedroMaze(num_rows=4, 
#                num_cols=4, 
#                room_size=3, 
#                render_mode='human',
#                max_episode_steps = 300,
#                view='top',rng = np.random.default_rng(seed = 1))

env = gym.make("MiniWorld-Maze-v0", 
               num_rows=3, 
               num_cols=3, 
               room_size=2, 
               render_mode='human',
               max_episode_steps = 500,
               view='top')
# env = MiniWorldMazeEnv()

# env = PyTorchFrame(env)
# env = BetterReward(env)
print(env.observation_space.shape)
env = WarpFrame(env)
print(env.observation_space.shape)
env = FrameStack(env, k=10)
obs,_ = env.reset()
print(env.observation_space.shape)
# print(obs)
# print(obs.shape)
# input()
# env = BetterReward(env)
env = ClipRewardEnv(env)
env = PyTorchFrame(env)

model = DQN("CnnPolicy", env, verbose=1,
            buffer_size=500000,
            learning_starts=50000,
            max_grad_norm=5,
            learning_rate=2e-4)
model.learn(total_timesteps=1000000, 
            log_interval=4,
            )
model.save("dqn_miniworld")

# del model # remove to demonstrate saving and loading

# model = DQN.load("dqn_miniworld")
print('DONE')
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
        print('done')
        break