import utils.projectParameters as param
import numpy as np
import math


class Environment6_1:
    def __init__(self, user_class_id, T):
        self.user_class_id = user_class_id
        self.pricing_prob_per_phase = param.pricing_probabilities_per_phase
        self.T = T
        self.n_phases = 3
        self.phase_size = math.ceil(T / self.n_phases)

        self.n_clicks = np.round(param.n_clicks_per_bid_per_class[self.user_class_id](param.bids)).astype(np.int32)
        self.tot_costs = param.total_cost_per_bid_per_class[self.user_class_id](param.bids)
        
        self.opt_price = {}
        self.opt_price_idx = {}
        self.opt_bid_idx = {}

        self.n_arms = len(param.prices)

        for phase in range(1, self.n_phases + 1):
            self.opt_price_idx[phase] = np.argmax(self.pricing_prob_per_phase[phase]*(param.prices-param.cost))
            self.opt_price[phase] = np.max(self.pricing_prob_per_phase[phase][self.opt_price_idx[phase]] * self.n_clicks * (
                        param.prices[self.opt_price_idx[phase]] - param.cost) - self.tot_costs)

            self.opt_bid_idx[phase] = np.argmax(self.pricing_prob_per_phase[phase][self.opt_price_idx[phase]] * self.n_clicks *
                                                (param.prices[self.opt_price_idx[phase]] - param.cost) - self.tot_costs)
            
    def round(self, pulled_arm, t): 
        phase = min(math.floor(t / self.phase_size) + 1, self.n_phases)
        result = np.random.binomial(1, self.pricing_prob_per_phase[phase][pulled_arm], self.n_clicks)
        reward = np.sum(result) * (param.prices[pulled_arm] - param.cost) - self.tot_costs 
        return np.sum(result), self.n_clicks - np.sum(result), reward, result 

    #def round(self, pulled_price_arm, pulled_bid_arm, t):
        phase = min(math.floor(t / self.phase_size) + 1, self.n_phases)

        result = np.random.binomial(1, self.pricing_prob_per_phase[phase][pulled_price_arm], self.n_clicks[pulled_bid_arm]) # Simulation of the clicks of the users

        reward = np.sum(result) * (param.prices[pulled_price_arm] - param.cost) - self.tot_costs[pulled_bid_arm] # Calculation of rewards

        return np.sum(result), self.n_clicks[pulled_bid_arm] - np.sum(result), reward, result   # Users who clicked, users who didn't click, reward, users who clicked (1) and users who didn't click (0)

    def get_optimal_price(self, t):
        phase = min(math.floor(t / self.phase_size) + 1, self.n_phases)
        return self.opt_price[phase]