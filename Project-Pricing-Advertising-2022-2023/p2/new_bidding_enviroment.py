import numpy as np
import utils.projectParameters as param
from utils.tools import calculate_margin


class Bidding_Environment_2:
    def __init__(self, bids, clicks_sigma, cost_sigma, user_class, n_arms):
        self.bids = bids
        self.user_id = user_class.get_user_index() + 1
        self.pricing_probabilities = param.pricing_probabilities_per_user[self.user_id]
        self.optimal_price_index = np.argmax(self.pricing_probabilities)
        self.conversion_rate_per_user = self.pricing_probabilities[
                           self.optimal_price_index]
        self.clicks_means = self.initialize_clicks(user_class=user_class, bids=bids)
        self.cost_means = self.initialize_cost(user_class=user_class, bids=bids)
        self.clicks_sigmas = np.ones(len(bids)) * clicks_sigma
        self.cost_sigmas = np.ones(len(bids)) * cost_sigma

        self.n_arms = n_arms
        self.n_clicks = np.round(param.n_clicks_per_bid_per_class[self.user_id](param.bids)).astype(np.int32)
        self.cum_costs = param.total_cost_per_bid_per_class[self.user_id](param.bids)


        self.optimal_bid = np.max(self.pricing_probabilities[self.optimal_price_index] * self.n_clicks * (
                    param.prices[self.optimal_price_index] - param.cost) - self.cum_costs)

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
        # Handle the exceptions
        if sample_cost > self.bids[pulled_arm]:
            sample_cost = self.bids[pulled_arm]
        if sample_cost < 0:
            sample_cost = 0
        if int(sample_clicks) < 0:
            sample_clicks = self.clicks_means[pulled_arm]

        daily_reward = self.conversion_rate_per_user * calculate_margin(
            param.prices[self.optimal_price_index]) - sample_cost * sample_clicks

        return daily_reward
