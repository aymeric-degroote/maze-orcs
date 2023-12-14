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

    def reset(self):
        self.grid *= self.UNKNOWN  # = 0
        self.center = (self.size // 2, self.size // 2)
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
                 maze_env,
                 hidden_space_dims=None,  # For Dense NN
                 nn_id=None,
                 hidden_state_dims=None,  # For LSTM
                 buffer_size=1,
                 memory=False):
        """Initializes a neural network that estimates the distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        print("Model input size", obs_space_dims)

        self.maze_env = maze_env
        self.obs_state_dims = obs_space_dims[0]
        self.hidden_state_dims = hidden_state_dims
        self.memory = memory

        if self.maze_env == "minigrid":
            # this is a very basic NN, check in the literature what kind of model architecture
            # they use if you want to improve it
            self.nn_id = nn_id

            if nn_id == "lstm-minigrid":

                # TODO: should define hidden_state_dims in function arguments
                self.hidden_state_dims = 100
                self.lstm_num_Layers = 1

                self.lstm = nn.LSTM(input_size=self.obs_state_dims,
                                    hidden_size=self.hidden_state_dims,
                                    num_layers=self.lstm_num_Layers,
                                    )

                # TODO: what about concatenating hidden_state and embedded_state as input for action_net?
                # TODO: I think we need a Deep Dense NN here. 3 layers size 64. Can't hurt.
                dense_layer_dims = 16
                self.action_net = nn.Sequential(
                    nn.Linear(self.hidden_state_dims, dense_layer_dims),
                    nn.Tanh(),
                    nn.Linear(dense_layer_dims, dense_layer_dims),
                    nn.Tanh(),
                    nn.Linear(dense_layer_dims, action_space_dims),
                    nn.Softmax(dim=1)   # TODO: check dim
                )

                if self.memory:
                    # no batch size = 1
                    self.hidden_state = torch.zeros(self.lstm_num_Layers, 1, self.hidden_state_dims)
                    self.cell_output = torch.zeros(self.lstm_num_Layers, 1, self.hidden_state_dims)

            else:
                if hidden_space_dims is None:
                    hidden_space_dims = [64, 32]
                model_dims = [np.prod(obs_space_dims) * 4] + hidden_space_dims + [action_space_dims]

                print("model dims", model_dims)
                layers = [nn.Flatten()]

                for i in range(len(model_dims) - 1):
                    layers.append(nn.Linear(model_dims[i], model_dims[i + 1]))
                    layers.append(nn.Tanh())

                layers.append(nn.Softmax(dim=1))
                self.net = nn.Sequential(*layers)
        else:
            # obs_space_dims is (60,80, 3)
            # action_space_dims is (3)

            if nn_id is None:
                nn_id = "384"

            self.nn_id = nn_id

            # TODO: all the hyperparameters are arbitrary for now, needs some fine tuning

            if self.nn_id == "lstm":

                # TODO: should define hidden_state_dims in function arguments
                self.hidden_state_dims = 200
                self.embedded_state_dims = 32
                self.lstm_num_Layers = 3

                # TODO: Should we use a max_pool layer at first?
                # TODO: Can we train embedding net separately? Use autoencoder?
                self.embedding_net = nn.Sequential(
                    # size (3, 60, 80)
                    nn.Conv2d(in_channels=3, out_channels=16,
                              kernel_size=(6, 6), stride=(2, 2)),
                    nn.ReLU(),
                    # size (16, 28, 38)
                    nn.Conv2d(in_channels=16, out_channels=32,
                              kernel_size=(4, 4), stride=(2, 2)),
                    nn.ReLU(),
                    # size (64, 13, 18)
                    nn.Conv2d(in_channels=32, out_channels=64,
                              kernel_size=(3, 4), stride=(2, 2)),
                    nn.ReLU(),
                    # size (64, 6, 8)
                    nn.Flatten(start_dim=0),
                    # size (64*6*8) = (3072)
                    nn.Linear(3072, 128),
                    nn.Tanh(),
                    nn.Linear(128, self.embedded_state_dims)
                )

                self.lstm = nn.LSTM(input_size=self.embedded_state_dims,
                                    hidden_size=self.hidden_state_dims,
                                    num_layers=self.lstm_num_Layers,
                                    )

                # TODO: what about concatenating hidden_state and embedded_state as input for action_net?
                # TODO: I think we need a Deep Dense NN here. 3 layers size 64. Can't hurt.
                dense_layer_dims = 64
                self.action_net = nn.Sequential(
                    nn.Linear(self.hidden_state_dims, dense_layer_dims),
                    nn.Tanh(),
                    nn.Linear(dense_layer_dims, dense_layer_dims),
                    nn.Tanh(),
                    nn.Linear(dense_layer_dims, action_space_dims),
                    nn.Softmax(dim=0)  # TODO: check dim
                )

                # used to count number of parameters of each model
                # def count_parameters(model):
                #     return sum(p.numel() for p in model.parameters() if p.requires_grad)

                if self.memory:
                    # no batch size = 1
                    self.hidden_state = torch.zeros(self.lstm_num_Layers, 1, self.hidden_state_dims)
                    self.cell_output = torch.zeros(self.lstm_num_Layers, 1, self.hidden_state_dims)

            elif self.nn_id == "3072":
                # h' = (h-kernel)/stride + 1
                self.net = nn.Sequential(
                    # size (3, 60, 80)
                    nn.Conv2d(in_channels=3 * buffer_size, out_channels=16,
                              kernel_size=(6, 6), stride=(2, 2)),
                    nn.ReLU(),
                    # size (16, 28, 38)
                    nn.Conv2d(in_channels=16, out_channels=32,
                              kernel_size=(4, 4), stride=(2, 2)),
                    nn.ReLU(),
                    # size (64, 13, 18)
                    nn.Conv2d(in_channels=32, out_channels=64,
                              kernel_size=(3, 4), stride=(2, 2)),
                    nn.ReLU(),
                    # size (64, 6, 8)
                    nn.Flatten(start_dim=0),
                    # size (64*6*8) = (3072)
                    nn.Linear(3072, 128),
                    nn.Tanh(),
                    nn.Linear(128, 16),
                    nn.Tanh(),
                    nn.Linear(16, action_space_dims),
                    nn.Softmax(dim=0)
                )
            elif self.nn_id == "384":
                # h' = (h-kernel)/stride + 1
                self.net = nn.Sequential(
                    # size (3, 60, 80)
                    nn.Conv2d(in_channels=3 * buffer_size, out_channels=16,
                              kernel_size=(5, 5), stride=(5, 5)),
                    nn.ReLU(),
                    # size (16, 12, 16)
                    nn.Conv2d(in_channels=16, out_channels=32,
                              kernel_size=(4, 4), stride=(4, 4)),
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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Conditioned on the observation, returns the distribution from which an action is sampled from.

        Args:
            state: Observation from the environment

        Returns:
            action_distribution: predicted distribution of action to take
        """
        if self.maze_env == "minigrid":
            if self.nn_id == "lstm-minigrid":
                batch_size, buffer_size, num_channels, height, width = state.size()

                assert self.memory, NotImplementedError()

                observation = state[:, 0].float()
                x = observation.reshape(batch_size, -1, self.obs_state_dims)

                # found this line on a forum. don't know why it works, but it works!
                # I think it's making a copy basically
                hidden = self.hidden_state.data
                cell_input = self.cell_output.data

                output, (new_hidden_state, new_cell_output) = self.lstm(x, (hidden, cell_input))

                self.hidden_state = new_hidden_state
                self.cell_output = new_cell_output

                action_distribution = self.action_net(output.float())

            else:
                action_distribution = self.net(state.float())
            return action_distribution

        elif self.maze_env == "miniworld":
            # (1, buffer_size, H, W, channels)
            state = state.permute(0, 1, 4, 2, 3)
            # (1, buffer_size, channels, H, W)

            if self.nn_id in ["3072", "384"]:
                state = state.reshape(1, -1, *state.shape[-2:])
                # (1, buffer_size * channels, H, W)

                action_distribution = self.net(state.float())

            elif self.nn_id == "lstm":
                # state is a frame sequence (batch_size, num_channels, seq_len, height, width)

                # Get the dimensions
                # (1, buffer_size, channels, H, W)
                batch_size, buffer_size, num_channels, height, width = state.size()

                if self.memory:
                    observation = state[:, 0]
                    embedded_obs = self.embedding_net(observation)
                    embedded_obs = embedded_obs.reshape(batch_size, -1, self.embedded_state_dims)

                    # found this line on a forum. don't know why it works, but it works!
                    # I think it's making a copy basically
                    hidden = self.hidden_state.data
                    cell_input = self.cell_output.data

                    output, (new_hidden_state, new_cell_output) = self.lstm(embedded_obs, (hidden, cell_input))

                    self.hidden_state = new_hidden_state
                    self.cell_output = new_cell_output

                    # self.hidden_state = [new_hidden_state]

                else:
                    # Initialize Hidden State (short-term memory & long-term memory)
                    hidden_state = (torch.zeros(batch_size, 1, self.hidden_state_dims),
                                    torch.zeros(batch_size, 1, self.hidden_state_dims))

                    output = torch.zeros(batch_size, 1, self.hidden_state_dims)

                    for frame in range(buffer_size):
                        observation = state[:, frame]
                        embedded_obs = self.embedding_net(observation)
                        embedded_obs = embedded_obs.reshape(batch_size, -1, self.embedded_state_dims)
                        output, hidden_state = self.lstm(embedded_obs, hidden_state)

                action_distribution = self.action_net(output)
            else:
                raise NameError("unknown nn_id")

            return action_distribution

        else:
            raise NameError("environment name missing")


class Agent:
    action_space = {0: "left",
                    1: "right",
                    2: "forward"}

    def __init__(self, obs_space_dims, action_space_dims,
                 maze_env="minigrid", policy=None,
                 verbose=False, training=False,
                 load_weights_fn=None, network=True,
                 nn_id=None, path='.',
                 learning_rate=1e-4, discount_factor=0.95, buffer_size=None,
                 memory=False,
                 **kwargs):
        if kwargs:
            print("Agent init: unused kwargs:", kwargs)

        self.path = path

        self.maze_env = maze_env

        self.obs_space_dims = obs_space_dims

        if isinstance(self.obs_space_dims, int):
            self.obs_space_dims = (self.obs_space_dims,)

        if buffer_size is not None:
            self.buffer = torch.zeros((1, buffer_size, *self.obs_space_dims))
            self.state_shape = self.buffer.shape
        else:
            self.state_shape = self.obs_space_dims
        if not hasattr(self, "policies"):
            self.policies = {}

        self.policies["random"] = self.random_policy
        self.policies["input"] = self.input_policy

        self.training = training
        if network:
            # Hyperparameters
            self.learning_rate = learning_rate  # Learning rate for policy optimization
            self.gamma = discount_factor  # Discount factor
            self.eps = 1e-6  # small number for mathematical stability

            self.log_probs = []  # Stores probability values of the sampled action
            self.rewards = []  # Stores the corresponding rewards

            print('nn_id', nn_id)
            # self.net = PolicyNetwork(self.state_shape, action_space_dims,
            #                         maze_env=maze_env, nn_id=nn_id,
            #                         buffer_size=buffer_size or 1)
            self.retain_graph = None or memory
            self.net = PolicyNetwork(self.obs_space_dims, action_space_dims,
                                     maze_env=maze_env, nn_id=nn_id,
                                     buffer_size=buffer_size or 1,
                                     memory=memory)

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

        print('policy', policy)
        print("keys", self.policies.keys())

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
        assert tuple(observation.shape) == tuple(self.obs_space_dims), \
            f"observation shape {observation.shape}, expected {self.obs_space_dims}"

        if hasattr(self, "buffer"):
            self.buffer = np.roll(self.buffer, 1, axis=1)
            self.buffer[0, 0] = observation

            state = self.buffer
        else:
            state = observation

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
            # TODO: do we need retain_graph? I added it for debugging LSTM, but not sure it matters
            # try to run LSTM with memory before removing it
            loss.backward(retain_graph=self.retain_graph)
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
        state = torch.tensor(state)

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

    def input_policy(self, *args, **kwargs):
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

        if load_maze:
            self.policies = {
                "greedy": self.greedy_bfs_policy,
                "left": self.keep_on_left
            }

        super().__init__(obs_space_dims, action_space_dims,
                         *args, **kwargs)

        if pos is None:
            pos = np.array([size // 2, size // 2])

        self.pos = pos  # never used
        self.direction = direction  # never used

        self.maze = MazeGrid(size)

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
        self.maze.update(observation)

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
            self.maze.reset()
            return np.random.randint(3)

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
        observation = observation.flatten()
        action = super().policy(observation)

        if action == 0:
            self.maze.left()
        elif action == 1:
            self.maze.right()
        elif action == 2:
            self.maze.forward()

        return action

    # override
    def network_policy(self, state, *args, **kwargs):
        objects = [self.maze.UNKNOWN, self.maze.EMPTY,
                   self.maze.WALL, self.maze.GOAL]

        state = np.array([[state == i for i in objects]], dtype=float)

        return super().network_policy(state, *args, **kwargs)


class MiniWorldAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override
    def network_policy(self, state, *args, **kwargs):
        # nothing to do actually, but it's here if needed

        return super().network_policy(state, *args, **kwargs)
