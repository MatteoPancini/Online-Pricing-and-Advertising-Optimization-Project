import utils.projectParameters as param
import numpy as np


class UCB_BaseOptimizer_5:
    def __init__(self, learner_type, user_class, learner_parameters):
        self.learner = learner_type(*learner_parameters)
        self.user_class_id = user_class.get_user_index() + 1
        self.num_clicks = np.round(param.n_clicks_per_bid_per_class[self.user_class_id](param.bids)).astype(np.int32)
        self.total_costs = param.total_cost_per_bid_per_class[self.user_class_id](param.bids)
        self.collected_rewards = []

    def pull_arm(self):
        price_index = self.learner.pull_arm()

        empirical_conversion_rate = self.learner.empirical_means + self.learner.confidence

        bid_index = np.argmax(empirical_conversion_rate[price_index] * self.num_clicks * (
                    param.prices[price_index] - param.cost) - self.total_costs)

        return price_index, bid_index

    def update(self, pulled_arm, reward):
        self.learner.update(pulled_arm, reward)
        self.collected_rewards.append(reward[2])
