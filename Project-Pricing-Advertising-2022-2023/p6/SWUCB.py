from p6.CumSumUCB import *
import numpy as np
import utils.projectParameters as param


class SWUCBLearner(UCBLearner):
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.window_size = window_size
        self.pulled_arms = np.array([])
        self.n_samples_per_t = np.array([])
        self.n_conversion_per_t = np.array([])

    def pull_arm(self):
        upper_conf = (self.empirical_means + self.confidence) * (np.array(param.prices) * 0.7)
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward[2])
        self.pulled_arms = np.append(self.pulled_arms, pulled_arm)
        self.n_samples_per_t = np.append(self.n_samples_per_t, reward[0] + reward[1])
        self.n_conversion_per_t = np.append(self.n_conversion_per_t, reward[0])
        for arm in range(self.n_arms):
            n_samples = np.sum(self.n_samples_per_t[-self.window_size:] * (self.pulled_arms[-self.window_size:] == arm))
            n_conversions = np.sum(self.n_conversion_per_t[-self.window_size:] * (self.pulled_arms[-self.window_size:] == arm))

            self.empirical_means[arm] = n_conversions / n_samples if n_samples > 0 else 0
            self.confidence[arm] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf

class UCBOptimizer:
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