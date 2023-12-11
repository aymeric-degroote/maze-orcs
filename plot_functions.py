import numpy as np
import matplotlib.pyplot as plt


def showPNG(grid):
    """Generate a simple image of the maze."""
    plt.figure(figsize=(10, 5))
    plt.imshow(grid, cmap=plt.cm.binary, interpolation='nearest')
    plt.xticks([]), plt.yticks([])
    plt.show()


def plot_agnostic_training_curve(run_info, stats, window_plot=100, maze_env="miniworld"):
    plt.plot(stats["reward_over_episodes"])
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig(f"runs_{maze_env}/plots/{run_info}-agnostic-rewards.png")
    plt.show()

    w = window_plot
    plt.plot(np.convolve(stats["reward_over_episodes"], np.ones(w), 'valid') / w)
    plt.xlabel(f"Average Reward over {w} episodes")
    plt.ylabel("Reward")
    plt.savefig(f"runs_{maze_env}/plots/{run_info}-agnostic-average-rewards.png")
    plt.show()


def plot_finetune_training_curve(run_info, rewards_per_maze,
                                 window_plot=100, maze_env="miniworld"):
    num_mazes = rewards_per_maze.shape[0]

    # Plot to see how fast it converges with one agnostic method or another
    for maze_seed in range(num_mazes):
        plt.plot(rewards_per_maze[maze_seed], c='gray', alpha=0.5)
    plt.plot(rewards_per_maze.mean(axis=0), c='red', label="Average")
    plt.legend()
    plt.title(f"Reward per episode")
    plt.savefig(f"runs_{maze_env}/plots/{run_info}-finetuned-rewards.png")
    plt.show()

    w = window_plot
    for maze_seed in range(num_mazes):
        plt.plot(np.convolve(rewards_per_maze[maze_seed], np.ones(w), 'valid') / w, c='gray', alpha=0.5)
    plt.plot(np.convolve(rewards_per_maze.mean(axis=0), np.ones(w), 'valid') / w, c='red', label="Average")
    plt.legend()
    plt.xlabel(f"Average Reward over {w} episodes")
    plt.savefig(f"runs_{maze_env}/plots/{run_info}-finetuned-average-rewards.png")
    plt.show()