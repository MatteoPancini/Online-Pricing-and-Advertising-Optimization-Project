import numpy as np
from p4.TS_learner import TS_Learner
from p4.GPUCB_learner import *
from p4.UCB_learner import *


class Multi_UCB_Learner():
    def __init__(self, bids_arms, prices_arms):
        self.n_click_learner = GPUCBLearner(bids_arms)
        self.cum_cost_learner = GPUCBLearner(bids_arms)
        self.price_learner = TS_Learner(len(prices_arms))
        #other parameters
        self.bids_arms = bids_arms
        self.n_bids_arms = len(bids_arms)
        self.prices_arms = prices_arms
        self.n_prices_arms = len(prices_arms)
        self.t = 0
        self.collected_rewards = np.array([])
        self.prices = [50, 100, 150, 200, 250]

    def update_observations(self, reward):
        self.collected_rewards = np.append(self.collected_rewards, reward)

    def update_observations_bulk(self, rewards):
        self.update_observations(rewards)

    def update(self, pulled_bids_arm, pulled_prices_arm, n_conversions, n_clicks, cum_cost, reward):
        self.update_observations(reward)
        self.n_click_learner.update(pulled_bids_arm, n_clicks)
        self.cum_cost_learner.update(pulled_bids_arm, cum_cost)
        self.price_learner.update_simple(pulled_prices_arm, [n_conversions, n_clicks - n_conversions, reward])

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
        sampled_reward = sampled_conversion_rate * n_clicks_upper_conf * (self.prices[sampled_price_idx] - (self.prices[sampled_price_idx]/100)*30) - cum_cost_lower_conf
        return np.random.choice(np.where(sampled_reward == sampled_reward.max())[0]), sampled_price_idx