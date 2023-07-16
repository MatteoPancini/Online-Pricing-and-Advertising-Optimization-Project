import math
import numpy as np

import utils.projectParameters as param



class Environment5:
    def __init__(self, user_class_id, T):
        self.user_class_id = user_class_id
        self.T = T
        self.pricing_prob_per_phase = param.pricing_probabilities_per_phase
        self.n_phases = 3
        self.phase_size = math.ceil(T / self.n_phases)
        self.opt_price = {}
        self.opt_price_idx = {}
        self.opt_bid_idx = {}

        self.n_arms = len(param.prices)
        self.n_clicks = np.round(param.n_clicks_per_bid_per_class[self.user_class_id](param.bids)).astype(np.int32)
        self.tot_costs = param.total_cost_per_bid_per_class[self.user_class_id](param.bids)


        for phase in range(1, self.n_phases + 1):
            self.opt_price_idx[phase] = np.argmax(self.pricing_prob_per_phase[phase] * (param.prices - param.cost))

            self.opt_price[phase] = np.max(self.pricing_prob_per_phase[phase][self.opt_price_idx[phase]] * self.n_clicks * (
                        param.prices[self.opt_price_idx[phase]] - param.cost) - self.tot_costs)

            self.opt_bid_idx[phase] = np.argmax(self.pricing_prob_per_phase[phase][self.opt_price_idx[phase]] * self.n_clicks *
                                                (param.prices[self.opt_price_idx[phase]] - param.cost) - self.tot_costs)





    def round(self, pulled_price_arm, pulled_bid_arm, t):
        phase = min(math.floor(t / self.phase_size) + 1, self.n_phases)

        result = np.random.binomial(1, self.pricing_prob_per_phase[phase][pulled_price_arm], self.n_clicks[pulled_bid_arm]) # Simulation of the clicks of the users

        reward = np.sum(result) * (param.prices[pulled_price_arm] - param.cost) - self.tot_costs[pulled_bid_arm] # Calculation of rewards

        return np.sum(result), self.n_clicks[pulled_bid_arm] - np.sum(result), reward, result   # Users who clicked, users who didn't click, reward, users who clicked (1) and users who didn't click (0)

    def get_optimal_price(self, t):
        phase = min(math.floor(t / self.phase_size) + 1, self.n_phases)
        return self.opt_price[phase]

    def get_phase(self, t):
        return min(math.floor(t / self.phase_size) + 1, self.n_phases)


