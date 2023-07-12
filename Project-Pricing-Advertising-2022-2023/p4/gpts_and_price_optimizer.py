import numpy as np
from GPTS_learner import *
from TS_learner import *

class Multi_TS_Learner():
    def __init__(self, bids_arms, prices_arms):
        self.n_click_learner = GPTS_Learner(bids_arms) #PRICING learner (cicks over bid function)
        self.cum_cost_learner = GPTS_Learner(bids_arms) #COST learner (costs over bid function)
        self.price_learner = TS_Learner(len(prices_arms)) #PRICING learner (actual price of the object)
        #other parameters
        self.bids_arms = bids_arms
        self.n_bids_arms = len(bids_arms)
        self.prices_arms = prices_arms
        self.n_prices_arms = len(prices_arms)
        self.t = 0
        self.collected_rewards = np.array([])
        self.prices = [50, 100, 150, 200, 250]

    def update_observations(self, reward): #appends the rewards
        self.collected_rewards = np.append(self.collected_rewards, reward)

    def update_observations_bulk(self, rewards): #just a for loop
        self.update_observations(rewards)

    def update(self, pulled_bids_arm, pulled_prices_arm, n_conversions, n_clicks, cum_cost, reward):
        self.update_observations(reward) #saves the reward
        self.n_click_learner.update(pulled_bids_arm, n_clicks) #update pricing
        self.cum_cost_learner.update(pulled_bids_arm, cum_cost) #update advertising
        self.price_learner.update(pulled_prices_arm, [n_conversions, n_clicks - n_conversions, reward]) #update basic price

    def update_bulk(self, pulled_bids_arms, pulled_prices_arms, n_conversions_per_arm, n_clicks_per_arm, cum_cost_per_arm, reward_per_arm): #just a for loop
        self.update_observations_bulk(reward_per_arm)
        self.n_click_learner.update_bulk(pulled_bids_arms, n_clicks_per_arm)
        self.cum_cost_learner.update_bulk(pulled_bids_arms, cum_cost_per_arm)
        self.price_learner.update_bulk(pulled_prices_arms, [n_conversions_per_arm, n_clicks_per_arm - n_conversions_per_arm, reward_per_arm])

    def pull_arms(self):
        sampled_price_index = self.price_learner.pull_arm()
        sampled_n_clicks = np.random.normal(self.n_click_learner.means, self.n_click_learner.sigmas) #uses click learner
        sampled_cum_cost = np.random.normal(self.cum_cost_learner.means, self.cum_cost_learner.sigmas) #uses cum_cost learner
        sampled_conversion_rate = np.random.beta(self.price_learner.beta_parameters[sampled_price_index, 0], #uses price learner
                                                 self.price_learner.beta_parameters[sampled_price_index, 1])
        margin = self.prices[sampled_price_index] - (self.prices[sampled_price_index]/100)*30
        sampled_reward = sampled_conversion_rate * sampled_n_clicks * (margin) - sampled_cum_cost
        return np.random.choice(np.where(sampled_reward == sampled_reward.max())[0]), sampled_price_index