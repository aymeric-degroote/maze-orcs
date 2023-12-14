from gymnasium import spaces, utils

from miniworld.entity import Box
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS
from miniworld.envs.maze import Maze as MiniWorldMaze

class PedroMaze(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Maze environment in which the agent has to reach a red box.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    +(1 - 0.2 * (step_count / max_episode_steps)) when red box reached and zero otherwise.

    ## Arguments

    ```python
    env = gym.make("MiniWorld-Maze-v0")
    # or
    env = gym.make("MiniWorld-MazeS2-v0")
    # or
    env = gym.make("MiniWorld-MazeS3-v0")
    # or
    env = gym.make("MiniWorld-MazeS3Fast-v0")
    ```

    """

    def __init__(
        self, num_rows=8, num_cols=8, room_size=3, max_episode_steps=None, rng = None, **kwargs
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.room_size = room_size
        self.gap_size = 0.25
        self.room_rng = rng

        MiniWorldEnv.__init__(
            self,
            max_episode_steps=max_episode_steps or num_rows * num_cols * 24,
            **kwargs,
        )
        utils.EzPickle.__init__(
            self,
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        if not self.room_rng:
            self.room_rng = self.np_random
        rows = []

        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):
                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex="brick_wall",
                    # floor_tex='asphalt'
                )
                row.append(room)

            rows.append(row)

        visited = set()

        def visit(i, j):
            """
            Recursive backtracking maze construction algorithm
            https://stackoverflow.com/questions/38502
            """

            room = rows[j][i]

            visited.add(room)

            # Reorder the neighbors to visit in a random order
            orders = [(0, 1), (0, -1), (-1, 0), (1, 0)]
            assert 4 <= len(orders)
            neighbors = []

            while len(neighbors) < 4:
                elem = orders[self.room_rng.choice(len(orders))]
                orders.remove(elem)
                neighbors.append(elem)

            # For each possible neighbor
            for dj, di in neighbors:
                ni = i + di
                nj = j + dj

                if nj < 0 or nj >= self.num_rows:
                    continue
                if ni < 0 or ni >= self.num_cols:
                    continue

                neighbor = rows[nj][ni]

                if neighbor in visited:
                    continue

                if di == 0:
                    self.connect_rooms(
                        room, neighbor, min_x=room.min_x, max_x=room.max_x
                    )
                elif dj == 0:
                    self.connect_rooms(
                        room, neighbor, min_z=room.min_z, max_z=room.max_z
                    )

                visit(ni, nj)

        # Generate the maze starting from the top-left corner
        visit(0, 0)

        self.box = self.place_entity(Box(color="red"))

        self.place_agent()

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info

class MiniWorldMazeEnv(MiniWorldMaze):
    def __init__(
            self,
            size=31,
            num_rows=3,
            num_cols=3,
            room_size=2,
            render_mode='human',  # 'top'
            view='top',
            max_steps: int | None = None,
            maze_seed=None,
            maze_type="dungeon",
            reward_new_cell=0.0,
            reward_closer_point=0.0,
            **kwargs,
    ):
        #self.size = size
        self.maze_type = maze_type
        self.maze_seed = maze_seed

        #self._gen_positions()

        self.total_reward = 0
        self.nb_actions = 0
        self.agent_pos_seen = set()
        self.reward_new_cell = reward_new_cell
        self.best_dist = 1e8
        self.reward_closer_point = reward_closer_point

        super().__init__(#"MiniWorld-Maze-v0",
                         num_rows=3,
                         num_cols=3,
                         room_size=2,
                         render_mode='human',  # 'top'
                         view='top')

    def step(self, *args, **kwargs):
        observation, reward, terminated, truncated, info = super().step(*args, **kwargs)
        # TODO: separate reward and custom_reward so we can track progress more easily

        custom_reward = 0
        if tuple(self.agent.pos) not in self.agent_pos_seen:
            custom_reward += self.reward_new_cell
            self.agent_pos_seen.add(tuple(self.agent.pos))

        if self.reward_closer_point > 0:
            dist = np.linalg.norm(self.agent.pos - self.box.pos)
            if dist < self.best_dist:
                self.best_dist = dist
                custom_reward += self.reward_closer_point

        reward += custom_reward

        self.nb_actions += 1
        self.total_reward += reward

        return observation, reward, terminated, truncated, info

    def reset(self, maze_seed=None, *args, **kwargs):
        if maze_seed is not None:
            self.maze_seed = maze_seed

        output = super().reset(*args, **kwargs)

        # TODO: the seed has no effect here

        #self._gen_positions()

        self.total_reward = 0
        self.nb_actions = 0

        return output

    def get_stats(self):
        return {
            "total_reward": self.total_reward,
            "nb_actions": self.nb_actions
        }

    def reset_to_seed(self):
        if self.maze_seed is not None:
            random.seed(self.maze_seed)
            np.random.seed(self.maze_seed)

    def gen_obs(self, *args, **kwargs):
        obs = super().render_obs(*args, **kwargs)

        return obs #.get('image')[:, :, 0]