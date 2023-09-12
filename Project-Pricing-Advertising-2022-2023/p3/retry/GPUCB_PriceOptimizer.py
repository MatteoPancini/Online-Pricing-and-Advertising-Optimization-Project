import numpy as np
from p3.retry.Optimizer import *
from p3.retry.GPUCB_PriceOptimizer import *
from p3.retry.GPUCB_Learner import *
import utils.projectParameters as param
from p3.retry.TS_Learner import *


class GPUCB_PriceOptimizer(OptimizerLearner):
    def __init__(self, bids_arms, prices_arms):
        super().__init__(bids_arms, prices_arms)
        self.n_click_learner = GPUCBLearner(bids_arms)
        self.cum_cost_learner = GPUCBLearner(bids_arms)
        self.price_learner = TSLearner(len(prices_arms))

    def update(self, pulled_bids_arm, pulled_prices_arm, n_conversions, n_clicks, cum_cost, reward):
        self.update_observations(reward)
        self.n_click_learner.update(pulled_bids_arm, n_clicks)
        self.cum_cost_learner.update(pulled_bids_arm, cum_cost)
        self.price_learner.update(pulled_prices_arm, [n_conversions, n_clicks - n_conversions, reward])

    def update_bulk(self, pulled_bids_arms, pulled_prices_arms, n_conversions_per_arm, n_clicks_per_arm, cum_cost_per_arm, reward_per_arm):
        self.update_observations_bulk(reward_per_arm)
        self.n_click_learner.update_bulk(pulled_bids_arms, n_clicks_per_arm)
        self.cum_cost_learner.update_bulk(pulled_bids_arms, cum_cost_per_arm)
        self.price_learner.update_bulk(pulled_prices_arms, [n_conversions_per_arm, n_clicks_per_arm - n_conversions_per_arm, reward_per_arm])

    def pull_arms(self):
        sampled_price_idx = self.price_learner.pull_arm()
        n_clicks_upper_conf = self.n_click_learner.empirical_means + self.n_click_learner.confidence
        cum_cost_lower_conf = self.cum_cost_learner.empirical_means - self.cum_cost_learner.confidence
        sampled_conversion_rate = np.random.beta(self.price_learner.beta_parameters[sampled_price_idx, 0],
                                                 self.price_learner.beta_parameters[sampled_price_idx, 1])
        sampled_reward = sampled_conversion_rate * n_clicks_upper_conf * (
                param.prices[sampled_price_idx] - param.cost) - cum_cost_lower_conf
        return np.random.choice(np.where(sampled_reward == sampled_reward.max())[0]), sampled_price_idx