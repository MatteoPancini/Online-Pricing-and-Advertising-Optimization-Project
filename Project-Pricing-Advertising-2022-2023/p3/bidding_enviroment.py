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

    #TODO: probabily remove
    def initialize_means(self, user_class, bids, price):
        means = np.zeros(len(bids))
        for i in range(len(means)):
                means[i] = fun(user_class, bids[i], price)
        return means
    
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
        sample_clicks = np.random.normal(self.clicks_means[pulled_arm], self.clicks_sigmas[pulled_arm])
        sample_cost = np.random.normal(self.cost_means[pulled_arm], self.cost_sigmas[pulled_arm])
        #Handle the exceptions
        if sample_cost > self.bids[pulled_arm]:
            sample_cost = self.bids[pulled_arm]
        if sample_cost < 0:
            sample_cost = 0
        if int(sample_clicks) < 0:
            sample_clicks = self.clicks_means[pulled_arm]

        return int(sample_clicks), sample_cost