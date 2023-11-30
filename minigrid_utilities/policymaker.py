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

    def __init__(self, obs_space_dims, action_space_dims: int,
                 maze_env, hidden_space_dims=None):
        """Initializes a neural network that estimates the distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        print("Model input size", obs_space_dims)

        if maze_env == "minigrid":
            # TODO: this is a very basic NN
            #  check in the literature what kind of model architecture they use

            hidden_space_dims = [64, 32]
            model_dims = [np.prod(obs_space_dims)] + hidden_space_dims + [action_space_dims]

            print("model dims", model_dims)
            layers = [nn.Flatten()]

            for i in range(len(model_dims)-1):
                layers.append(nn.Linear(model_dims[i], model_dims[i+1]))
                layers.append(nn.Tanh())

            layers.append(nn.Softmax(dim=1))
            self.net = nn.Sequential(*layers)
        else:
            # obs_space_dims is (60,80, 3)
            # action_space_dims is (3)

            # TODO: the hyperparameters are arbitrary

            self.net = nn.Sequential(
                # size (3, 60, 80)
                nn.Conv2d(in_channels=3, out_channels=16,
                          kernel_size=(5,5), stride=(5,5)),
                nn.ReLU(),
                # size (16, 12, 16)
                nn.Conv2d(in_channels=16, out_channels=32,
                          kernel_size=(4,4), stride=(4,4)),
                nn.ReLU(),
                # size (32, 3, 4)
                nn.Flatten(start_dim=0),
                # size (32*3*4) = (384)
                nn.Linear(384, 32),
                nn.Tanh(),
                nn.Linear(32, 16),
                nn.Tanh(),
                nn.Linear(16, action_space_dims),
                nn.Softmax(dim=0)
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

    def __init__(self, obs_space_dims, action_space_dims,
                 maze_env="minigrid", policy=None,
                 verbose=False, training=False,
                 load_weights_fn=None, network=True, path='.',
                 learning_rate=1e-4, discount_factor=0.95, buffer_size=None,
                 **kwargs):
        if kwargs:
            print("Agent init: unused kwargs:", kwargs)

        self.path = path

        self.maze_env = maze_env

        self.obs_space_dims = obs_space_dims

        if isinstance(self.obs_space_dims, int):
            self.obs_space_dims = (self.obs_space_dims,)

        if buffer_size is None:
            buffer_size = 1

        self.buffer = torch.zeros((1, buffer_size, *self.obs_space_dims))

        self.policies = {"random": self.random_policy,
                         "input": self.input_policy}

        self.training = training
        if network:
            # Hyperparameters
            self.learning_rate = learning_rate  # Learning rate for policy optimization
            self.gamma = discount_factor  # Discount factor
            self.eps = 1e-6  # small number for mathematical stability

            self.log_probs = []  # Stores probability values of the sampled action
            self.rewards = []  # Stores the corresponding rewards

            self.net = PolicyNetwork(self.buffer.shape, action_space_dims,
                                     maze_env=maze_env)

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
        fn = os.path.join(self.path, f"runs_{self.maze_env}/weights", weights_fn)
        torch.save(self.net.state_dict(), fn)
        return fn

    def load_weights(self, load_weights_fn):
        fn = os.path.join(self.path, f"runs_{self.maze_env}/weights", load_weights_fn)
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
        assert tuple(observation.shape) == tuple(self.obs_space_dims), f"observation shape {observation.shape}, " \
                                                         f"expected {self.obs_space_dims}"

        self.buffer = np.roll(self.buffer, 1, axis=1)
        self.buffer[0, 0] = observation

        state = self.buffer

        if self.actions_to_do:
            action = self.actions_to_do.pop()
        else:
            if self.default_policy is not None:
                policy = self.policies[self.default_policy]
            else:
                policy = self.network_policy
            action = policy(state)

        if action not in [0, 1, 2]:
            print(f"Unknown action: {action}")

        if self.verbose:
            print(self.action_space[action])
        return action

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
        return np.random.randint(3)

    def network_policy(self, state: np.ndarray, get_dist=False) -> int:
        """Returns an action, conditioned on the policy and observation.

                Args:
                    state: Observation from the environment
                    get_dist: if True, returns the distribution instead of sampling from it

                Returns:
                    action: Action to be performed
                """
        state = torch.tensor(state).squeeze().permute(2, 0, 1)

        # returns probability of taking each action
        distrib = self.net(state).squeeze()
        # use squeeze() only if batch of 1
        # TODO: do we always have a batch of 1?

        if get_dist:
            return distrib.detach().numpy()

        action = distrib.multinomial(num_samples=1)
        action = action.numpy()

        log_prob = torch.log(distrib[action])
        self.log_probs.append(log_prob)

        return int(action)

    def input_policy(self, observation=None):
        a = input("Action? ")
        if a in "012":
            return int(a)
        else:
            print("Wrong input. Options are: 0, 1 or 2")
            return self.input_policy()


class MiniGridAgent(Agent):
    def __init__(self, obs_space_dims, action_space_dims, size,
                 pos=None, direction=0,
                 load_maze=None,
                 *args, **kwargs):

        super().__init__(obs_space_dims*4, action_space_dims,
                         *args, **kwargs)

        if pos is None:
            pos = np.array([size // 2, size // 2])

        self.pos = pos  # never used
        self.direction = direction  # never used

        self.maze = MazeGrid(size)

        if load_maze:
            self.policies["greedy"] = self.greedy_bfs_policy
            self.policies["left"] = self.keep_on_left

    # override
    def random_policy(self, observation=None):
        if self.maze.can_move_forward():
            return np.random.randint(3)
        else:
            return np.random.randint(2)

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

    # override
    def policy(self, observation):
        objects = [self.maze.UNKNOWN, self.maze.EMPTY,
                   self.maze.WALL, self.maze.GOAL]

        image = np.array([[observation == i for i in objects]], dtype=float)

        action = super().policy(image.flatten())

        if action == 0:
            self.maze.left()
        elif action == 1:
            self.maze.right()
        elif action == 2:
            self.maze.forward()

        return action

class MiniWorldAgent(Agent):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


