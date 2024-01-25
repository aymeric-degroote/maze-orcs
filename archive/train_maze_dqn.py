'''Here I will try to implement a basic DQN to solve the maze nav task'''

import numpy as np

from archive.dqn_utilities import WarpFrame, PyTorchFrame, FrameStack
from archive.dqn_utilities import DQNAgent
# from dqn_utilities.memory import ReplayBuffer
# from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm
from archive.dqn_utilities import PedroMaze
from archive.dqn_utilities import ReplayBuffer

import torch
import wandb
wandb.login()
run=wandb.init()

env = PedroMaze(num_rows=3, 
               num_cols=3, 
               room_size=2, 
               render_mode='human',
               max_episode_steps = 400,
               view='top',
               rng = np.random.default_rng(seed = 1))


# env = WarpFrame(env)
# env = PyTorchFrame(env)
# env = BetterReward(env)

# env = MaxAndSkipEnv(env, skip=4)
env = WarpFrame(env)
# env = PyTorchFrame(env)
# env = ClipRewardEnv(env)

# env = BetterReward(env)
env = FrameStack(env, 8)
env = PyTorchFrame(env)
print("initialized environment...")
print(f'OBS SPACE: {env.observation_space}')


replay_buffer = ReplayBuffer(5000)
print("initialized replay buffer...")


agent = DQNAgent(
    env.observation_space,
    env.action_space,
    replay_buffer,
    lr=.001,
    batch_size=32,
    gamma=0.99,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
print("initialized dqn agent...")



episode_rewards = [0.0]

state, _ = env.reset()
print(7777)
movingAverage = 0

maze_eps = 0
steps_in_maze = 0
num_episodes = 0
epsilon = 1
min_epsilon = 0.01
decay_rate = 0.9999


for t in tqdm(range(100000)):
    
    if t % 1000 == 0:
        wandb.log({'training episode step':t+1, 'moving avg reward': movingAverage, 'episode #': num_episodes, 'episodes_in_current_maze': maze_eps, 'epsilon': epsilon})


    if np.random.choice([True, False], p=[epsilon,1-epsilon]):
        
        # Explore
        action = env.action_space.sample()
    else:
        # Exploit
        action = agent.act(state)

    next_state, reward, done, truncated, info = env.step(action)

    # to convert to new gym standard
    done = np.logical_or(done, truncated)

    agent.memory.add(state, action, reward, next_state, float(done))
    state = next_state

    episode_rewards[-1] += reward
    if done:
        maze_eps += 1
        if num_episodes >= 110:
            movingAverage=np.mean(episode_rewards[len(episode_rewards)-100:len(episode_rewards)-1])
        # if maze_eps >= 1100:
        #     # change maze
        #     env.room_rng = np.random.default_rng(seed = t)

        # #     #reset replay buffer for maze
        #     # agent.memory = ReplayBuffer(5000)
        #     steps_in_maze = 0
        #     maze_eps = 0
            # epsilon = 1.0

            
        wandb.log({ "episode reward" : episode_rewards[-1], "moving avg": movingAverage, 'episode num': num_episodes})
        state,_ = env.reset()
        episode_rewards.append(0.0)
        

    if t > 1000 and t % 1 == 0:
        agent.optimise_td_loss()
        torch.nn.utils.clip_grad_norm_(list(agent.policy_network.parameters()),5)
        # agent.scheduler.step(movingAverage)

    if t > 1000 and t % 100 == 0:
        agent.update_target_network()

    num_episodes = len(episode_rewards)
    steps_in_maze +=1

    new_eps = epsilon * decay_rate
    epsilon = new_eps if new_eps > min_epsilon else min_epsilon

    if t % 50000 == 0:
        torch.save(agent.policy_network.state_dict(), f'dqn_results/models/12-2/model{t}.pth')

    
    

np.savetxt('dqn_results/reward_logs/moonshot/model_rewards2.csv', episode_rewards,
            delimiter=',', fmt='%1.3f')
run.finish()



















