import numpy as np
import matplotlib.pyplot as plt


class MazeGrid():
    UNKNOWN = 0
    EMPTY = 1
    WALL = 2
    GOAL = 8
    
    def __init__(self, size):
        self.size = size
        self.grid = np.full((size,size), self.UNKNOWN)
        self.center = (size//2, size//2)
        self.grid[self.center] = self.EMPTY
    
    def update(self, observation):
        """
        observation is a 7x7 grid. Each point has 3 coordinates:
        - object_idx:
            0: unseen
            1: empty
            2: wall
            8: goal
        - color_idx: color of object
        - state: no use here, indicates if door is open/closed
        
        all values can be find at:
        https://github.com/mit-acl/gym-minigrid/blob/master/gym_minigrid/minigrid.py
        """
        observed_image = observation.get('image')[:,:,0]
        #print("obs")
        #print(observed_image)
        window = self.grid[self.center[0]-6:self.center[0]+1, 
                           self.center[1]-3:self.center[1]+4] * 1  # copy
        new_window = np.flip(np.rot90(observed_image, 3), 1)
        
        self.grid[self.center[0]-6:self.center[0]+1, 
                  self.center[1]-3:self.center[1]+4] = window + new_window*(window==0)
    
    def can_move_forward(self):
        return self.grid[self.center[0]-1, self.center[1]] == self.EMPTY
    
    def forward(self):
        if self.can_move_forward():
            self.grid = np.roll(self.grid, 1, axis=0)
    
    def left(self):
        self.grid = np.rot90(self.grid, 3)
    
    def right(self):
        self.grid = np.rot90(self.grid, 1)


class Agent():
    action_space = {0: "left"  ,   1: "right"   ,  2: "forward"}
    def __init__(self, size, pos=None, direction=0):
        if pos is None:
            pos = np.array([size//2, size//2])
        
        self.pos = pos # never used
        self.direction = direction # never used
        
        self.maze = MazeGrid(size)
        self.maze.grid[self.pos[0], self.pos[1]] = self.maze.EMPTY
        
        self.rewards = []
        self.actions_to_do = []
    
    def policy(self, observation):
        """
        0: left     1: right     2: forward
        """
        self.maze.update(observation)
        
        #print(self.maze.grid)
        if self.actions_to_do:
            action = self.actions_to_do.pop()
        else:
            action = self.greedy_bfs_policy()
            #action = self.input_policy()
            #action = self.random_policy()
        
        if action == 0:
            self.maze.left()
        elif action == 1:
            self.maze.right()
        else:
            self.maze.forward()
        
        print(self.action_space[action])
        return action
    
    def random_policy(self):
        if self.maze.can_move_forward():
            return np.random.randint(3)
        else:
            return np.random.randint(2)
    
    def input_policy(self):
        a = input("Action? ")
        if a in "012":
            return int(a)
        else:
            print("Wrong input. Options are: 0, 1 or 2")
            return self.input_policy()
    
    def neighbors(self, cell):
        neighs = []
        cells = [(cell[0]-1, cell[1]),
                  (cell[0]+1, cell[1]),
                  (cell[0], cell[1]-1),
                  (cell[0], cell[1]+1)]
        actions = [[2], # forward
                   [2,0,0], # left left forward
                   [2,0], # right forward
                   [2,1]  # left forward
                   ]
        # actions are written backwards 'cause we pop the list later
        for c,a in zip(cells, actions):
            if 0 <= c[0] < self.maze.size and \
               0 <= c[1] < self.maze.size:
                if self.maze.grid[c] != self.maze.WALL:
                    neighs.append((c,a))
        return neighs
    
    def greedy_bfs_policy(self):
        """
        Not efficient at all because we run the algorithm every time
        """
        to_explore = [self.maze.center]
        previous_cell = {self.maze.center:(None,[])}
        target_found = False
        target = None
        while not target_found and to_explore:
            cell = to_explore.pop(0)
            for c,a in self.neighbors(cell):
                if c not in previous_cell:
                    previous_cell[c] = (cell, a)
                    to_explore.append(c)
                if self.maze.grid[c] == self.maze.GOAL or \
                    self.maze.grid[c] == self.maze.UNKNOWN:
                    target = c
                    target_found = True
                    break
        
        if target is None:
            print("BFS couldn't find goal or unknown cell")
            print(self.maze.grid)
            print(previous_cell)
            return 42 #np.random.randint(3)
        
        intermediate_cell, actions = previous_cell[target]
        while intermediate_cell != self.maze.center:
            target = intermediate_cell
            intermediate_cell, actions = previous_cell[target]
        
        self.actions_to_do += actions
        
        return self.actions_to_do.pop()
        
        
        