from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv
from dqn_utilities.larotta_maze import PedroMaze
from dqn_utilities.wrappers import WarpFrame, PyTorchFrame, BetterReward, MaxAndSkipEnv, FrameStack, ClipRewardEnv
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env


# env = PedroMaze(num_rows=4, 
#                num_cols=4, 
#                room_size=3, 
#                render_mode='human',
#                max_episode_steps = 300,
#                view='top')
# env = gym.make("MiniWorld-Maze-v0", 
#                num_rows=3, 
#                num_cols=3, 
#                room_size=2, 
#                render_mode='human',
#                max_episode_steps = 400,
#                view='top')
# env = BetterReward(env)

env = make_vec_env("MiniWorld-Maze-v0", n_envs=8)

# Initialize the model
model = PPO("CnnPolicy", env, verbose=1,n_steps=128, n_epochs=4, batch_size=256,vf_coef= 0.5 ,ent_coef= 0.01)

# Train the model
model.learn(1000)

model.save("ppo_miniworld")
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
# model = model_class.load("./her_bit_env", env=env)
print("DONE")
obs = env.reset()
for _ in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render('human')
    
