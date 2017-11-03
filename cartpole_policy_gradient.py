import numpy as np
import os
import tensorflow as tf
import gym
import random
import gym

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

S, H, A, R = 4, 10, 2, 1

def episode(action, parameters):
	observation = env.reset()
	episode_reward = 0
	while True:
		observation, reward, done, info = env.step(action(observation, parameters))
		episode_reward += reward
		if done:
			break
	return episode_reward
	
def policy_gradient():
	with tf.variable_scope("policy"):
		states = tf.placeholder("float", [None, S], name="states")
		actions = tf.placeholder("float", [None, A], name="actions")
		dfuture_rewards = tf.placeholder("float", [None, 1], name="dfuture_rewards")
		w1 = tf.get_variable("policy_w1", [S, A])
		action_probs = tf.nn.softmax(tf.matmul(states, w1))
		action_probs_s = tf.reduce_sum(tf.multiply(action_probs, actions), axis=1)
		loss = -tf.reduce_sum(tf.log(action_probs_s) * dfuture_rewards)
		loss_summary = tf.summary.scalar("loss", loss)
		optimizer = tf.train.AdamOptimizer(1e-2).minimize(loss)
		return states, action_probs, actions, dfuture_rewards, optimizer, loss, loss_summary

def value_gradient():
	with tf.variable_scope("value"):
		states = tf.placeholder("float", [None, S], name="states")
		rewards = tf.placeholder("float", [None, R], name="rewards")
		w1 = tf.get_variable("w1", [S, H])
		b1 = tf.get_variable("b1", [H])
		w2 = tf.get_variable("w2", [H, R])
		b2 = tf.get_variable("b2", [R])
		h1 = tf.nn.relu(tf.matmul(states, w1) + b1)
		rewards_pred = tf.matmul(h1, w2) + b2
		loss = tf.nn.l2_loss(rewards_pred - rewards)
		loss_summary = tf.summary.scalar("loss", loss)
		optimizer = tf.train.AdamOptimizer(1e-1).minimize(loss)
		return states, rewards_pred, rewards, optimizer, loss, loss_summary

def run_episode(env, policy_grad, value_grad, sess):
	policy_states, policy_action_probs, policy_actions, policy_dfuture_rewards, policy_optimizer, policy_loss, policy_loss_summary = policy_grad
	value_states, value_rewards_pred, value_rewards, value_optimizer, value_loss, value_loss_summary = value_grad
	
	# run episode
	observation = env.reset()
	episode_reward = 0
	states = []
	actions = []
	transitions = []
	future_rewards = []
	dfuture_rewards = []
	for _ in range(200):
		states.append(observation)
		action_prob = sess.run(policy_action_probs, feed_dict={policy_states: [observation]})[0][0]
		action = 0 if random.uniform(0, 1) < action_prob else 1
		one_hoc_action = np.zeros(2)
		one_hoc_action[action] = 1
		actions.append(one_hoc_action)
		old_observation = observation
		observation, reward, done, info = env.step(action)
		transitions.append((old_observation, action, reward))
		episode_reward += reward
		if done: break 
	
	# calculate the gradient of expected discounted reward
	for index, transition in enumerate(transitions):
		observation, action, reward = transition
		
		rest_transitions = np.array(transitions[index:])
		gamma = 0.97
		discounts = np.array([gamma ** index for index, _ in enumerate(rest_transitions)])
		future_reward = np.sum(np.multiply(rest_transitions[:,2], discounts))
		future_rewards.append(future_reward)
		
		future_reward_pred = sess.run(value_rewards_pred, feed_dict={value_states: [observation]})[0]
		dfuture_rewards.append(future_reward - future_reward_pred)
		
    # optimize state value
	_, value_loss_summary_val = sess.run([value_optimizer, value_loss_summary], feed_dict={value_states: np.vstack(states), value_rewards: np.vstack(future_rewards)})
	writer.add_summary(value_loss_summary_val, i)
    
    # optimize policy
	_, policy_loss_summary_val = sess.run([policy_optimizer, policy_loss_summary], feed_dict={policy_states: np.vstack(states), policy_actions: np.vstack(actions), policy_dfuture_rewards: np.vstack(dfuture_rewards)})
	writer.add_summary(policy_loss_summary_val, i)
	
	return episode_reward
		
env = gym.make('CartPole-v0')
policy_grad = policy_gradient()
value_grad = value_gradient()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("logs", sess.graph)

# train
for i in range(2000):
	reward = run_episode(env, policy_grad, value_grad, sess)
	if reward == 200:
		print('reward 200 at iteration', i)

# render
policy_states, policy_action_probs, policy_actions, policy_dfuture_rewards, policy_optimizer, policy_loss, policy_loss_summary = policy_grad
while True:
	done = False
	observation = env.reset()
	i = 0
	while not done:
		env.render()
		action_prob = sess.run(policy_action_probs, feed_dict={policy_states: [observation]})[0][1]
		action = 0 if action_prob < 0.5 else 1
		observation, reward, done, _ = env.step(action)
		if done: print('done at iteration', i)
		i += 1