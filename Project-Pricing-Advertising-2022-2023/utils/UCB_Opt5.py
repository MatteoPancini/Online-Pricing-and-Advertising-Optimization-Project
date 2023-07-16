import utils.projectParameters as param
import numpy as np


class UCB_BaseOptimizer_5:
    def __init__(self, learner_type, class_id, learner_parameters):
        self.learner = learner_type(*learner_parameters)
        self.n_clicks = np.round(param.n_clicks_per_bid_per_class[class_id](param.bids)).astype(np.int32)
        self.cum_costs = param.total_cost_per_bid_per_class[class_id](param.bids)
        self.collected_rewards = []

    def pull_arm(self):
        price_idx = self.learner.pull_arm()
        empirical_conversion_rate = self.learner.empirical_means + self.learner.confidence
        bid_idx = np.argmax(empirical_conversion_rate[price_idx] * self.n_clicks * (
                    param.prices[price_idx] - param.cost) - self.cum_costs)
        return price_idx, bid_idx

    def update(self, pulled_arm, reward):
        self.learner.update(pulled_arm, reward)
        self.collected_rewards.append(reward[2])
