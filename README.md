# maze-orcs

Multi-agent RL for Efficient Maze Navigation


ORCS 4529 - Reinforcement Learning
Project - Fall 2023

Team members:  
Aymeric Degroote  
Pedro Leandro La Rotta  
Kunal Kundu

Instructor: Shipra Agrawal

To train an agnostic model using REINFORCE on MiniWorld:
```
python3 miniworld-maml-train-agnostic.py
```
in the terminal. 
To fine tune the agnostic model:
```
python3 miniworld-maml-finetune.py
```
in the terminal.



To train REINFORCE on MiniGrid:
```
python3 train-minigrid.py
```
in the terminal.

To evaluate REINFORCE on MiniGrid:
```
python3 run-minigrid.py
```
in the terminal.

To display REINFORCE on MiniGrid:
```
python3 run-minigrid.py human
```
in the terminal.



Way to go is by running:
```
python3 maze_gym.py
```
in the terminal.

Module for training a maze navigator via DQN. In this task the agent is rewarded 
when it successfully navigates to the red box in the map. Maps are randomly generated 
mazes, and you can run training via:
'''
python3 maze_dqn.py
'''
after installing dependencies




