
from __future__ import annotations
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
import mazelib
from mazelib import Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt



class MazeEnv(MiniGridEnv):
    def __init__(
        self,
        size=31,
        agent_start_pos=(np.random.randint(2,28), np.random.randint(2,28)),
        agent_start_dir=0,
        goal_pos=(np.random.randint(2,28), np.random.randint(2,28)),
        max_steps: int | None = None,
        **kwargs,
    ):
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
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        m = Maze()
        
        
        m.generator = Prims(15, 15)
        m.start = (self.agent_start_pos[0], self.agent_start_pos[1])
        m.end = (self.goal_pos[0], self.goal_pos[1])
        m.generate()
        gridded = m.grid


        for row in range(height):
            for col in range(width):
                if gridded[row, col] == 1:
                    self.grid.set(row,col, Wall())

        self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

        # TODO: fix error caused by agent spawning inside wall
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"
def showPNG(grid):
    """Generate a simple image of the maze."""
    plt.figure(figsize=(10, 5))
    plt.imshow(grid, cmap=plt.cm.binary, interpolation='nearest')
    plt.xticks([]), plt.yticks([])
    plt.show()

def main():
    env = MazeEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()
