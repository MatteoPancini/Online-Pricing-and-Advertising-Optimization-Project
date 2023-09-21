import numpy as np
from utils.tools import fun

class Bidding_Environment_3:
    def __init__(self, bids, clicks_sigma, cost_sigma, user_class, n_arms):
        self.bids = bids
        self.clicks_means = self.initialize_clicks(user_class=user_class, bids=bids)
        self.cost_means = self.initialize_cost(user_class=user_class, bids=bids)
        self.clicks_sigmas = np.ones(len(bids)) * clicks_sigma
        self.cost_sigmas = np.ones(len(bids)) * cost_sigma
        self.n_arms = n_arms
    
    def initialize_clicks(self, user_class, bids):
        means = np.zeros(len(bids))
        for j, b in enumerate(bids):
            means[j] = user_class.get_click_bids(b)
        return means

    def initialize_cost(self, user_class, bids):
        means = np.zeros(len(bids))
        for j, b in enumerate(bids):
            means[j] = user_class.get_cost_per_click(b)
        return means

    def round(self, pulled_arm):
        sample_clicks = self.clicks_means[pulled_arm] + np.random.normal(0, self.clicks_sigmas[pulled_arm]/10)
        sample_cost =self.cost_means[pulled_arm]+  np.random.normal(0, self.cost_sigmas[pulled_arm]/10)

        if sample_cost > self.bids[pulled_arm]:
            sample_cost = self.bids[pulled_arm]

        if int(sample_clicks) < 0:
            sample_clicks = self.clicks_means[pulled_arm]
        
        if sample_cost < 0:
            sample_cost = 0
            sample_clicks = 0

        return int(sample_clicks), sample_cost