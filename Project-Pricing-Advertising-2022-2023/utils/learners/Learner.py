"""Defined by the number of arms it can pull, the current round and the list of
collected rewards. The learner interacts with the environment by selecting the arm to pull at each round and
observing the reward given by the environment"""

import numpy as np

class Learner:
    #Initializes the Learner class with the number of arms, and creates empty lists for rewards per arm and collected rewards.
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        #list of emty lists, one list for every arm
        self.rewards_per_arm = x = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])

    # input is pulled arm and environmental reward
    # update the list reward per arm and collect reward
    def update_observations(self, pulled_arm, reward):
        #append the reward value to the list related to the arm
        #pulled by the learn and passed as input
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
