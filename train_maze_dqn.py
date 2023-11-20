'''Here I will try to implement a basic DQN to solve the maze nav task'''

import numpy as np

from dqn_utilities.wrappers import WarpFrame, PyTorchFrame, BetterReward, MaxAndSkipEnv, FrameStack
from dqn_utilities.agent import DQNAgent
from dqn_utilities.memory import ReplayBuffer
from tqdm import tqdm
from dqn_utilities.larotta_maze import PedroMaze

import torch
import wandb
wandb.login()
run=wandb.init()

env = PedroMaze(num_rows=4, 
               num_cols=4, 
               room_size=3, 
               render_mode='human',
               max_episode_steps = 500,
               view='top',
               rng = np.random.default_rng(seed = 1))


# env = WarpFrame(env)
# env = PyTorchFrame(env)
# env = BetterReward(env)

env = MaxAndSkipEnv(env, skip=4)
env = WarpFrame(env)
env = PyTorchFrame(env)
# env = ClipRewardEnv(env)
env = FrameStack(env, 4)
print("initialized environment...")
print(f'OBS SPACE: {env.observation_space}')


replay_buffer = ReplayBuffer(5000)
print("initialized replay buffer...")


agent = DQNAgent(
    env.observation_space,
    env.action_space,
    replay_buffer,
    lr=1e-4,
    batch_size=32,
    gamma=0.99,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
print("initialized dqn agent...")



episode_rewards = [0.0]

state, _ = env.reset()
movingAverage = 0

maze_eps = 0
steps_in_maze = 0
num_episodes = 0
epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.9999


for t in tqdm(range(500000)):
    
    if t % 1000 == 0:
        print(f'training episode step {t+1}, moving avg reward: {movingAverage}, episode # {num_episodes}, episodes in current maze: {maze_eps}, epsilon: {epsilon}')


    if np.random.choice([True, False], p=[epsilon,1-epsilon]):
        
        # Explore
        action = env.action_space.sample()
    else:
        # Exploit
        action = agent.act(state)

    next_state, reward, done, truncated, info = env.step(action)

    agent.memory.add(state, action, reward, next_state, float(done))
    state = next_state

    episode_rewards[-1] += reward
    if done or truncated:
        maze_eps += 1
        if num_episodes >= 110:
            movingAverage=np.mean(episode_rewards[len(episode_rewards)-100:len(episode_rewards)-1])
        if maze_eps >= 500:
            # change maze
            env.room_rng = np.random.default_rng(seed = t)

            #reset replay buffer for maze
            # agent.memory = ReplayBuffer(5000)
            steps_in_maze = 0
            maze_eps = 0
            epsilon = 1.0

            
        wandb.log({ "episode reward" : episode_rewards[-1], "moving avg": movingAverage, 'episode num': num_episodes})
        state,_ = env.reset()
        episode_rewards.append(0.0)
        

    if steps_in_maze > 1000 and t % 1 == 0:
        agent.optimise_td_loss()

    if steps_in_maze > 1000 and t % 100 == 0:
        agent.update_target_network()

    num_episodes = len(episode_rewards)
    steps_in_maze +=1

    new_eps = epsilon * decay_rate
    epsilon = new_eps if new_eps > min_epsilon else min_epsilon

torch.save(agent.policy_network.state_dict(), f'dqn_results/models/checkpoint1.pth')
np.savetxt('dqn_results/reward_logs/rewards_per_episode1.csv', episode_rewards,
            delimiter=',', fmt='%1.3f')
run.finish()



















