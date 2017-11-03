import numpy as np
import gym
from bokeh.plotting import figure, show

env = gym.make('CartPole-v0')

def episode(action, parameters):
	observation = env.reset()
	episode_reward = 0
	
	while True:
		observation, reward, done, info = env.step(action(observation, parameters))
		episode_reward += reward
		if done:
			break
	
	return episode_reward

def linear_action(observation, parameters):
	action = 0 if np.matmul(observation, parameters) < 0 else 1
	return action

def plot(episodes_to_solve):
	x = episodes_to_solve.nonzero()[0]
	y = episodes_to_solve[x]
	average = int(np.matmul(x, y)/np.sum(y))
	plot = figure(title = 'Episodes to solve, {} on average (red line)'.format(average), x_axis_label = 'Number of episodes', y_axis_label = 'Times occurred')
	plot.vbar(x = x, top = y, width = 2)
	plot.line(x = [average, average], y = [0, 1], color="red", line_width=2)
	show(plot)

def hill_climbing():
	episodes_to_solve = np.zeros(10000)
	for _ in range(400):
		best_reward = 0
		parameters = np.random.uniform(-1, 1, 4)
		for episode_index in range(1, 200):
			noise_intensity = 0.5
			noise = np.random.uniform(-1, 1, 4) * noise_intensity
			noised_parameters = parameters + noise
			reward = episode(linear_action, noised_parameters)
			if best_reward < reward:
				best_reward = reward
				parameters = noised_parameters
				if best_reward == 200:
					episodes_to_solve[episode_index] += 1
					print('solved at episode', episode_index)
					break
	return episodes_to_solve

episodes_to_solve = hill_climbing()
plot(episodes_to_solve)

