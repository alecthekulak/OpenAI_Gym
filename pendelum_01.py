# Alec Kulakowski
import gym
import numpy as np
import pandas as pd
from sklearn.linear_model import PassiveAggressiveRegressor

env = gym.make('Pendulum-v0')
model = PassiveAggressiveRegressor()
observation_history = pd.DataFrame(columns=['x0', 'x1', 'x2'])



def make_action(action_raw):
    if action_raw > 2:
        action_raw = 2
    elif action_raw < -2:
        action_raw = -2
    return np.array([action_raw], dtype=np.float32)


for trial in range(1):
    observation = env.reset()
    reward = -8
    action = 0
    for frame in range(100):
        previous_observation = observation
        previous_reward = reward

        ###
        env.render()
        # action = env.action_space.sample()
        action = -3
        print(make_action(action))
        observation, reward, done, info = env.step(make_action(action))
        if done:
            print(f'Finished after {frame} frames')
            break

print(f'observation: {observation}')
print(f'reward: {reward}')
print(f'done: {done}')
print(f'info: {info}')

print(env.action_space)
print(env.action_space.high)  # 2
print(env.action_space.low)  # -2
print(env.reward_range)  # -inf, inf
print(env.observation_space)
print(env.observation_space.high)  # 1, 1, 8
print(env.observation_space.low)  # -1, -1, -8
