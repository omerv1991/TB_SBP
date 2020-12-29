"""import gym
import highway_env
import numpy as np

from stable_baselines3 import HER, SAC, DDPG , DQN
from stable_baselines3.common.noise import NormalActionNoise

#env = gym.make("parking-v0")
env = gym.make("CarRacing-v0")
# Create the action noise object that will be used for exploration
n_actions = env.action_space.shape[0]
noise_std = 0.2
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))

model = HER('MlpPolicy', env, DDPG, n_sampled_goal=4,
            goal_selection_strategy='future', online_sampling=True,
            verbose=1, buffer_size=int(1e6),
            learning_rate=1e-3, action_noise=action_noise,
            gamma=0.95, batch_size=256,
            policy_kwargs=dict(net_arch=[256, 256, 256]), max_episode_length=100)


# Train for 2e5 steps
model.learn(int(2e5))
# Save the trained agent
#model.save('her_ddpg_highway')
#model = HER.load('her_ddpg_highway', env=env)
obs = env.reset()

# Evaluate the agent
episode_reward = 0
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    if done or info[0].get('is_success', False):
        print("Reward:", episode_reward, "Success?", info[0].get('is_success', False))
        episode_reward = 0.0
        obs = env.reset()"""


import gym
from stable_baselines3 import DQN
from traj_replay_buffer import TrajReplayBuffer
from stable_baselines3.common import results_plotter
import os
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


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


model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    #env.render()
    if done:
      obs = env.reset()
env.close()

plot_results(log_dir)


