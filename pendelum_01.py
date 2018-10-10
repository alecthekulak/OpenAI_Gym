# Alec Kulakowski
import gym
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

env = gym.make('Pendulum-v0')
# model = SGDRegressor(tol=None, max_iter=None)  # 3.53%
# model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam')
model = SVR(kernel='linear')  # max = 79.18%
# model = SVR(kernel='rbf')  # max = 63.01%
observation_history = pd.DataFrame(columns=['x0', 'x1', 'x2'])
good_action_history = pd.DataFrame(columns=['y0'])


def make_action(action_raw):
    if action_raw > 2:
        action_raw = 2
    elif action_raw < -2:
        action_raw = -2
    return np.array([action_raw], dtype=np.float32)


print("Generating random training samples")
while len(observation_history) < 500: #500
    observation = env.reset()
    reward = False  # reward = -8
    for frame in range(200):
        previous_reward = reward
        env.render()
        action = make_action(env.action_space.sample())
        observation, reward, done, info = env.step(action)
        observation = observation.flatten() * np.array([1, 1, 1 / 8])  # Normalize the input variables
        if previous_reward and reward > previous_reward:
            observation_history.loc[len(observation_history)] = observation
            good_action_history.loc[len(good_action_history)] = action.item()
        if done:
            break

for trial in range(20):
    print(f'Starting trial #{trial+1}')
    observation = env.reset()
    reward = False  # reward = -8
    model.fit(observation_history, good_action_history.values.ravel().tolist())
    score = model.score(observation_history, good_action_history.values.ravel().tolist())
    print(f'Model score: {round(score*100, 2)}%, with {len(observation_history)} training samples.')
    for frame in range(201):
        previous_reward = reward
        ###
        env.render()
        action = make_action(model.predict(observation.reshape(1, -1)))
        observation, reward, done, info = env.step(action)
        observation = observation.flatten() * np.array([1, 1, 1 / 8])  # Normalize the input variables
        if previous_reward and reward > previous_reward:
            observation_history.loc[len(observation_history)] = observation
            good_action_history.loc[len(good_action_history)] = action.item()
        if done:
            if frame == 199:
                print(f'Finished trial #{trial+1} after {frame+1} frames')
            else:
                print(f'Model finished early in trial #{trial+1} after only {frame+1} frames')
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
