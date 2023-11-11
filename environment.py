"""
Used this minigrid tutorial:
https://minigrid.farama.org/content/create_env_tutorial/

And integrated mazelib to generate the maze
"""

from __future__ import annotations

import numpy as np
import random
import torch

# minigrid: RL environment (+ UI)
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

# mazelib: generates mazes
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.generate.DungeonRooms import DungeonRooms


class MazeEnv(MiniGridEnv):
    """
    Reward system:
    - get to the goal: 1 - 0.9 * (step_count / max_steps)
    - move to a cell never seen: 0.01
    """
    def __init__(
        self,
        size=31,
        max_steps: int | None = None,
        maze_seed=None,
        maze_type="dungeon",
        reward_new_cell=0.0,
        **kwargs,
    ):
        self.size = size
        self.maze_type = maze_type
        self.maze_seed = maze_seed

        self._gen_positions()

        self.total_reward = 0
        self.nb_actions = 0
        self.reward_new_cell = reward_new_cell
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "We are not using this function obviously"
    
    def _gen_positions(self):
        if self.maze_seed is not None:
            np.random.seed(self.maze_seed)

        self.agent_start_pos = (1 + 2*np.random.randint(self.size//2),
                                1 + 2*np.random.randint(self.size//2))
        self.goal_pos = (np.random.randint(2,self.size-2),
                         np.random.randint(2,self.size-2))
        self.agent_pos_seen = {self.agent_start_pos}
        self.agent_start_dir = 0
        
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        
        m = Maze()

        if self.maze_seed is not None:
            random.seed(self.maze_seed)

        if self.maze_type.lower() == "prims":
            m.generator = Prims(self.size//2, self.size//2)
        elif self.maze_type.lower() in ["dungeon", "dungeonrooms"]:
            m.generator = DungeonRooms(self.size//2,self.size//2)
        else:
            print("Unknown maze type, generating using DungeonRooms")
            m.generator = DungeonRooms(self.size//2,self.size//2)
        
        m.start = self.agent_start_pos
        m.end = self.goal_pos
        m.generate()
        gridded = m.grid

        for row in range(height):
            for col in range(width):
                if gridded[row, col] == 1:
                    self.grid.set(row,col, Wall())
        
        self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        self.mission = "Maze ORCS"
    
    def step(self, *args, **kwargs):
        observation, reward, terminated, truncated, info = super().step(*args, **kwargs) 
        
        if self.agent_pos not in self.agent_pos_seen:
            reward += self.reward_new_cell
            self.agent_pos_seen.add(self.agent_pos)

        self.nb_actions += 1

        self.total_reward += reward
        return observation, reward, terminated, truncated, info
    
    def reset(self, maze_seed=None, *args, **kwargs):
        if maze_seed is not None:
            self.maze_seed = maze_seed

        output = super().reset(*args, **kwargs)

        self._gen_positions()
        
        self.total_reward = 0
        self.nb_actions = 0
        
        return output
    
    def get_stats(self):
        
        return self.total_reward, self.agent_pos_seen, self.nb_actions
        
        
        