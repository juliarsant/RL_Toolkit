"""
Julia Santaniello
06/25/23

For training a single policy. Input hyperparamters and saved policy name.
"""
from Algorithms.deepQNetwork import DeepQNetwork

def train(p, game, num_actions, obs_space, epochs, gamma, epsilon, batchSize, buffer, replay):
    policy = DeepQNetwork(p, game, num_actions, obs_space, epochs, gamma, epsilon, buffer, replay, batchSize)
    policy.train()
    policy.evaluate()