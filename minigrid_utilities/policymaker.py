from __future__ import annotations

import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


class MazeGrid:
    UNKNOWN = 0
    EMPTY = 1
    WALL = 2
    GOAL = 8

    def __init__(self, size):
        self.size = size
        self.grid = np.full((size, size), self.UNKNOWN)
        self.center = (size // 2, size // 2)
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
        observed_image = observation  # .get('image')[:,:,0]
        window = self.grid[self.center[0] - 6:self.center[0] + 1,
                           self.center[1] - 3:self.center[1] + 4] * 1  # copy
        new_window = np.flip(np.rot90(observed_image, 3), 1)

        self.grid[self.center[0] - 6:self.center[0] + 1,
                  self.center[1] - 3:self.center[1] + 4] = window + new_window * (window == 0)

    def can_move_forward(self):
        return self.grid[self.center[0] - 1, self.center[1]] == self.EMPTY

    def forward(self):
        if self.can_move_forward():
            self.grid = np.roll(self.grid, 1, axis=0)

    def left(self):
        self.grid = np.rot90(self.grid, 3)

    def right(self):
        self.grid = np.rot90(self.grid, 1)


class PolicyNetwork(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims, action_space_dims: int):
        """Initializes a neural network that estimates the distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 16
        hidden_space2 = 16

        # Create Network
        # TODO: make a real NN, not this thing
        self.net = nn.Sequential(
            # nn.Conv2d(1,20,5),
            # nn.ReLU(),
            # nn.Conv2d(20,64,5),
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(obs_space_dims * 4, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
            nn.Linear(hidden_space2, action_space_dims),
            nn.Softmax(dim=1)
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Conditioned on the observation, returns the distribution from which an action is sampled from.

        Args:
            observation: Observation from the environment

        Returns:
            action_distribution: predicted distribution of action to take
        """
        action_distribution = self.net(observation.float())  # TODO: remove x.float?

        return action_distribution


class Agent:
    action_space = {0: "left",
                    1: "right",
                    2: "forward"}

    def __init__(self, obs_space_dims, action_space_dims, size,
                 pos=None, direction=0, policy=None,
                 verbose=False, load_maze=False, training=False,
                 load_weights_fn=None, network=True, path='.',
                 learning_rate=1e-4, **kwargs):

        if kwargs:
            print("Agent init: unused kwargs:", kwargs)

        self.path = path

        if pos is None:
            pos = np.array([size // 2, size // 2])

        self.pos = pos  # never used
        self.direction = direction  # never used

        self.policies = {"random": self.random_policy}

        self.maze = MazeGrid(size)

        if load_maze:
            self.policies["input"] = self.input_policy
            self.policies["greedy"] = self.greedy_bfs_policy
            self.policies["left"] = self.keep_on_left

        self.training = training
        if network:
            # Hyperparameters
            self.learning_rate = learning_rate  # Learning rate for policy optimization
            self.gamma = 0.95  # Discount factor
            self.eps = 1e-6  # small number for mathematical stability

            self.log_probs = []  # Stores probability values of the sampled action
            self.rewards = []  # Stores the corresponding rewards

            self.net = PolicyNetwork(obs_space_dims, action_space_dims)

            if load_weights_fn:
                self.load_weights(load_weights_fn)

                if self.training:
                    self.net.train()
                else:
                    # TODO: print a warning?
                    print("Warning: Agent network is in eval mode")
                    self.net.eval()
            self.optimizer = torch.optim.AdamW(self.net.parameters(),
                                               lr=self.learning_rate)

            self.policies["network"] = self.network_policy

        if policy is not None and policy not in self.policies.keys():
            policy = None
            print(f"policy invalid, using another one instead")

        self.default_policy = policy

        self.verbose = verbose

        self.rewards = []
        self.actions_to_do = []

    def save_weights(self, weights_fn):
        fn = os.path.join(self.path, "runs_minigrid/weights", weights_fn)
        torch.save(self.net.state_dict(), fn)
        return fn

    def load_weights(self, load_weights_fn):
        fn = os.path.join(self.path, "runs_minigrid/weights", load_weights_fn)
        self.net.load_state_dict(torch.load(fn))
        print("Loaded model weights from", fn)

    def get_weights(self, copy=False):
        """
        Returns the weights of the model. It does NOT return a
        """
        if copy:
            return deepcopy(self.net.state_dict())
        else:
            return self.net.state_dict()

    def set_weights(self, weights):
        self.net.load_state_dict(weights)

    def policy(self, observation):
        """
        0: left     1: right     2: forward
        """
        self.maze.update(observation)

        if self.actions_to_do:
            action = self.actions_to_do.pop()
        else:
            if self.default_policy is not None:
                action = self.policies[self.default_policy](observation)
            else:
                action = self.network_policy(observation)
                # action = self.greedy_bfs_policy()
                # action = self.keep_on_left()
                # action = self.input_policy()
                # action = self.random_policy()

        if action == 0:
            self.maze.left()
        elif action == 1:
            self.maze.right()
        elif action == 2:
            self.maze.forward()
        else:
            print(f"Unknown action: {action}")

        if self.verbose:
            print(self.action_space[action])
        return action

    def network_policy(self, state: np.ndarray, get_dist=False) -> int:
        """Returns an action, conditioned on the policy and observation.
        
        Args:
            state: Observation from the environment
            get_dist: if True, returns the distribution instead of sampling from it

        Returns:
            action: Action to be performed
        """
        objects = [self.maze.UNKNOWN, self.maze.EMPTY,
                   self.maze.WALL, self.maze.GOAL]

        image = np.array([[state == i for i in objects]], dtype=float)

        image = torch.tensor(image)

        # returns probability of taking each action
        distrib = self.net(image).squeeze()
        # use squeeze() only if batch of 1
        # TODO: do we always have a batch of 1?

        if get_dist:
            return distrib.detach().numpy()

        action = distrib.multinomial(num_samples=1)
        action = action.numpy()

        log_prob = torch.log(distrib[action])
        self.log_probs.append(log_prob)

        return int(action)

    def compute_loss(self):
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = torch.tensor(0.)
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.log_probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        return loss

    def update(self, loss=None, do_backward=True):
        """Updates the policy network's weights."""
        if loss is None:
            loss = self.compute_loss()

        # Update the policy network
        if do_backward:
            self.optimizer.zero_grad()
            loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.log_probs = []
        self.rewards = []

    def random_policy(self, observation=None):
        if self.maze.can_move_forward():
            return np.random.randint(3)
        else:
            return np.random.randint(2)

    def input_policy(self, observation=None):
        a = input("Action? ")
        if a in "012":
            return int(a)
        else:
            print("Wrong input. Options are: 0, 1 or 2")
            return self.input_policy()

    def greedy_bfs_policy(self, observation=None):
        """
        Not efficient at all because we run the algorithm every time
        """

        def neighbors(cell):
            neighs = []
            cells = [((cell[0] - 1) % self.maze.size, cell[1]),
                     ((cell[0] + 1) % self.maze.size, cell[1]),
                     (cell[0], (cell[1] - 1) % self.maze.size),
                     (cell[0], (cell[1] + 1) % self.maze.size)]
            actions = [[2],  # forward
                       [2, 0, 0],  # left left forward
                       [2, 0],  # left forward
                       [2, 1]  # right forward
                       ]
            # actions are written backwards because we use list.pop() later
            for c, a in zip(cells, actions):
                if 0 <= c[0] < self.maze.size and \
                        0 <= c[1] < self.maze.size:
                    if self.maze.grid[c] != self.maze.WALL:
                        neighs.append((c, a))
            return neighs

        to_explore = [self.maze.center]
        previous_cell = {self.maze.center: (None, [])}
        target_found = False
        target = None
        while not target_found and to_explore:
            cell = to_explore.pop(0)
            for c, a in neighbors(cell):
                if c not in previous_cell:
                    previous_cell[c] = (cell, a)
                    to_explore.append(c)
                if self.maze.grid[c] == self.maze.GOAL or \
                        self.maze.grid[c] == self.maze.UNKNOWN:
                    target = c
                    target_found = True
                    break

        if target is None:
            # TODO: I haven't been able to reproduce the bug but it happened before
            print("BFS couldn't find goal or unknown cell")
            print(self.maze.grid)
            print(previous_cell)
            return 42
            # return np.random.randint(3)

        intermediate_cell, actions = previous_cell[target]
        while intermediate_cell != self.maze.center:
            target = intermediate_cell
            intermediate_cell, actions = previous_cell[target]

        self.actions_to_do += actions

        return self.actions_to_do.pop()

    def keep_on_left(self, observation=None):

        cell = self.maze.center
        neighs = []
        cells = [(cell[0], cell[1] - 1),  # left
                 (cell[0] - 1, cell[1]),  # forward
                 (cell[0], cell[1] + 1),  # right
                 (cell[0] + 1, cell[1]),  # behind
                 ]
        actions = [[2, 0],  # left forward
                   [2],  # forward
                   [2, 1],  # right forward
                   [2, 1, 1],  # left left forward
                   ]
        # actions are written backwards because we use list.pop() later
        for c, a in zip(cells, actions):
            if 0 <= c[0] < self.maze.size and \
                    0 <= c[1] < self.maze.size:
                if self.maze.grid[c] == self.maze.GOAL:
                    neighs = [(c, a)]
                    break
                elif self.maze.grid[c] != self.maze.WALL:
                    neighs.append((c, a))

        self.actions_to_do += neighs[0][1]

        return self.actions_to_do.pop()
