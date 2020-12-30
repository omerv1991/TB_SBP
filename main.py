import gym
from stable_baselines3 import DQN
from traj_replay_buffer import TrajReplayBuffer
import os
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3.common import results_plotter


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

env = gym.make('CartPole-v1')
#env = gym.make('FrozenLake-v0')


log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

model = DQN('MlpPolicy', env, verbose=1)#prioritized_replay=True

model.replay_buffer=TrajReplayBuffer(
            model.buffer_size,
            model.observation_space,
            model.action_space,
            model.device,
            optimize_memory_usage=model.optimize_memory_usage,
        )

model.learn(total_timesteps=int(1e5))
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
plot_results(log_dir)

print("done")