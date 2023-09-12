import numpy as np

class OptimizerLearner:
    def __init__(self, bids_arms, prices_arms):
        self.bids_arms = bids_arms
        self.n_bids_arms = len(bids_arms)
        self.prices_arms = prices_arms
        self.n_prices_arms = len(prices_arms)
        self.t = 0
        self.collected_rewards = np.array([])

    def update_observations(self, reward):
        self.collected_rewards = np.append(self.collected_rewards, reward)

    def update_observations_bulk(self, rewards):
        self.update_observations(rewards)