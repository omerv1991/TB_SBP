import gym
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
        obs = env.reset()

"""
import gym
from stable_baselines3 import DQN

env = gym.make('CartPole-v1')
#model = PPO('MlpPolicy', 'CartPole-v1').learn(10000)

model = DQN('MlpPolicy', env, verbose=1,model_class=DQN)#prioritized_replay=True
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
env.close()
"""