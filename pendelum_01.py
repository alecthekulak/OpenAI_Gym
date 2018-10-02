# Alec Kulakowski
import gym
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor, Perceptron

env = gym.make('Pendulum-v0')
model = SGDRegressor(tol=None, max_iter=None)
# model = Perceptron()
observation_history = pd.DataFrame(columns=['x0', 'x1', 'x2'])
good_action_history = pd.DataFrame(columns=['y0'])


def make_action(action_raw):
    if action_raw > 2:
        action_raw = 2
    elif action_raw < -2:
        action_raw = -2
    return np.array([action_raw], dtype=np.float32)


while len(observation_history) < 50:
    print("Generating random training samples")
    observation = env.reset()
    reward = False  # reward = -8
    for frame in range(200):
        previous_reward = reward
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(make_action(action))
        observation = observation.flatten() * np.array([1, 1, 1 / 8])  # Normalize the input variables
        if previous_reward and reward > previous_reward:
            observation_history.loc[len(observation_history)] = observation
            good_action_history.loc[len(good_action_history)] = action
        if done:
            break

for trial in range(10):
    print(f'Starting trial #{trial+1}')
    observation = env.reset()
    reward = False  # reward = -8
    model.fit(observation_history, good_action_history.values.ravel())
    score = model.score(observation_history, good_action_history.values.ravel())
    print(f'Model score: {round(score*100, 2)}%, with {len(observation_history)} training samples.')
    for frame in range(200):
        previous_reward = reward
        ###
        env.render()
        action = model.predict(observation.reshape(1, -1))
        observation, reward, done, info = env.step(make_action(action))
        observation = observation.flatten() * np.array([1, 1, 1 / 8])  # Normalize the input variables
        if previous_reward and reward > previous_reward:
            observation_history.loc[len(observation_history)] = observation
            good_action_history.loc[len(good_action_history)] = action
        if done:
            print(f'Finished trial #{trial+1} after {frame} frames')
            break

# print(f'observation: {observation}')
# print(f'reward: {reward}')
# print(f'done: {done}')
# print(f'info: {info}')
#
# print(env.action_space)
# print(env.action_space.high)  # 2
# print(env.action_space.low)  # -2
# print(env.reward_range)  # -inf, inf
# print(env.observation_space)  # box 3
# print(env.observation_space.high)  # 1, 1, 8
# print(env.observation_space.low)  # -1, -1, -8
env.close()
