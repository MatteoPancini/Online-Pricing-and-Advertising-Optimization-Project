import utils.projectParameters as param
import numpy as np


class Bidding_Environment_2:
    def __init__(self, class_id):
        self.class_id = class_id
        self.probabilities = param.pricing_probabilities_per_user[self.class_id]
        self.optimal_price_idx = np.argmax(self.probabilities*(param.prices-param.cost))
        self.n_clicks = np.round(param.n_clicks_per_bid_per_class[self.class_id](param.bids)).astype(np.int32)
        self.cum_costs = param.total_cost_per_bid_per_class[self.class_id](param.bids)
        self.optimal_bid = np.max(self.probabilities[self.optimal_price_idx] * self.n_clicks * (param.prices[self.optimal_price_idx] - param.cost) - self.cum_costs)
        self.optimal_bid_idx = np.argmax(self.probabilities[self.optimal_price_idx] * self.n_clicks * (param.prices[self.optimal_price_idx] - param.cost) - self.cum_costs)
        self.n_arms = len(param.bids)

    def round(self, pulled_arm):
        bid = param.bids[pulled_arm]
        n_clicks = max(0, param.n_clicks_per_bid_per_class[self.class_id](bid) + np.random.normal(0,
                                                                                                 param.clicks_sigma))
        cum_cost = max(0, param.n_clicks_per_bid_per_class[self.class_id](bid) + np.random.normal(0,
                                                                                                 param.cost_sigma))
        reward = self.probabilities[self.optimal_price_idx] * n_clicks * (param.prices[self.optimal_price_idx] - param.cost) - cum_cost
        return n_clicks, cum_cost, reward
