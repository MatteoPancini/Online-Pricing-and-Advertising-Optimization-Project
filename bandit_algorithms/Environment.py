"""Environment Class is defined by:
Number of arms, probability distribution for each arm reward function
The environment interacts with the learner by returning a stochastic reward
depending on the pulled arm"""

import numpy as np
class Environment():
    def __init__(self, n_arms, probabilities):
        #Initializes the Environment class with the number of arms and probabilities of each arm.
        self.n_arms = n_arms
        self.probabilities = probabilities #in this example Bernoullian, 1 value per arm

    def round(self, pulled_arm):
        #Simulates a round of the multi-armed bandit game, where the player pulls a specific arm and receives a reward.
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward


