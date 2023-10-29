"""
Used this minigrid tutorial:
https://minigrid.farama.org/content/create_env_tutorial/

And integrated mazelib to generate the maze
"""

from __future__ import annotations
# import gym-minigrid

import numpy as np
import matplotlib.pyplot as plt

# minigrid: RL environment (+ UI)
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

# mazelib: generates mazes
import mazelib
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.generate.BacktrackingGenerator import BacktrackingGenerator


class MazeEnv(MiniGridEnv):
    def __init__(
        self,
        size=31,
        agent_start_pos=None,
        agent_start_dir=0,
        goal_pos=None,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.size = size
        if agent_start_pos is None:
            # odd coordinates are never walls so agent will not spawn inside a wall
            agent_start_pos = (1 + 2*np.random.randint(self.size//2), 
                               1 + 2*np.random.randint(self.size//2))
        if goal_pos is None:
            goal_pos = (np.random.randint(2,self.size-2), 
                        np.random.randint(2,self.size-2))
        
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = goal_pos

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
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        print('grid size:', width,height)
        
        # 15x15 for a 31x31 grid?
        # 31 = 15 empty cells + 16 walls, that's why!
        m = Maze()
        m.generator = Prims(15, 15)
        m.start = (self.agent_start_pos[0], self.agent_start_pos[1])
        m.end = (self.goal_pos[0], self.goal_pos[1])
        m.generate()
        gridded = m.grid
        print(f'gridded: {gridded.shape}')
        #print(f'walls: {np.asarray(np.where(gridded == 1)).T[:10]}\n'
        #      f'other: {np.asarray(np.where(gridded != 1)).T[:10]}')

        for row in range(height):
            for col in range(width):
                if gridded[row, col] == 1:
                    self.grid.set(row,col, Wall())

        self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])
        
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            # never run
            print("Placing agent...")
            self.place_agent()

        self.mission = "grand mission"
