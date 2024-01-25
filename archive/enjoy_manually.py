import numpy as np
from archive.dqn_utilities import ManualControl
from archive.dqn_utilities import WarpFrame, PyTorchFrame, FrameStack

from archive.dqn_utilities import PedroMaze

# ENVIRONMENT_NAME = "MiniWorld-Maze-v0" #change to whatever miniworld environment you want to manually play with
# env = gym.make(ENVIRONMENT_NAME, 
#                num_rows=3, 
#                num_cols=3, 
#                room_size=2, 
#                render_mode='human',
#                view='top')
env = PedroMaze(num_rows=3, 
               num_cols=3, 
               room_size=2, 
               render_mode='human',
               max_episode_steps = 300,
               view='agent',
               rng = np.random.default_rng(seed = 1))
# env = MaxAndSkipEnv(env, skip=4)
env = WarpFrame(env)
env = PyTorchFrame(env)
env = FrameStack(env, 4)

env.reset()

manual_control = ManualControl(env, no_time_limit=False, domain_rand=False)
manual_control.run()






