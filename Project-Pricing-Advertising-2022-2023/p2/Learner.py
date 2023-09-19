import numpy as np

class Learner:
    def __init__(self, n_arms):
        # Initializes the Learner class with the number of arms, and creates empty lists for rewards per arm and collected rewards.

        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = x = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        # append the reward value to the list related to the arm
        # pulled by the learn and passed as input
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
