# maze-orcs

Deep Reinforcement Learning for End-to-End Maze Navigation

ORCS E4529 - Reinforcement Learning
Project - Fall 2023

Team members:  
Aymeric Degroote  
Pedro Leandro La Rotta  
Kunal Kundu

Instructor: Shipra Agrawal


To train an agnostic model using REINFORCE on MiniWorld:
```
python3 reinforce_runs/miniworld-classic-train-agnostic.py
```
or
```
python3 reinforce_runs/miniworld-maml-train-agnostic.py
```
in the terminal. 

To fine tune an agnostic model on several mazes and assess performance:
```
python3 reinforce_runs/miniworld-finetune-agnostic.py
```
in the terminal.

Replace 'miniworld' by 'minigrid' for the equivalent in MiniGrid.



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

To see how REINFORCE is doing on MiniGrid:
```
python3 run-minigrid.py human
```
in the terminal.



Way to go is by running:
```
python3 maze_gym.py
```
in the terminal.



In the training of REINFORCE using a Value function approach, we edited a few files in the miniworld library 
directly that we did not find relevant to add to the repo. However, those two files are:
- manual_control.py
- miniworld_control.py

