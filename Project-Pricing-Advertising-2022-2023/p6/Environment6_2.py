import utils.projectParameters as param
import numpy as np
import math
import numpy as np
from utils.clairvoyant_tools import get_optimal_parameters
import math


class Environment6_2:
    def __init__(self, class_id, T):
        self.class_id = class_id
        self.probabilities = param.pricing_probabilities_per_phase_6_2
        self.T = T
        self.n_phases = len(self.probabilities.keys())
        self.phase_size = 14
        _, optimum_bid, _ = get_optimal_parameters(class_index=class_id)
        self.n_clicks = np.round(param.n_clicks_per_bid_per_class[self.class_id](optimum_bid)).astype(np.int32)
        self.cum_costs = param.total_cost_per_bid_per_class[self.class_id](optimum_bid)
        self.optimal_price_idx = {}
        self.optimal = {}
        self.n_arms = len(param.prices)
        for phase in range(1, self.n_phases + 1):
            self.optimal_price_idx[phase] = np.argmax(self.probabilities[phase]*(param.prices-param.cost))
            self.optimal[phase] = self.probabilities[phase][self.optimal_price_idx[phase]] * self.n_clicks * (param.prices[self.optimal_price_idx[phase]] - param.cost) - self.cum_costs

    def round(self, pulled_arm, t):
        phase = (math.floor(t / self.phase_size)) % self.n_phases + 1
        result = np.random.binomial(1, self.probabilities[phase][pulled_arm], self.n_clicks)
        reward = np.sum(result) * (param.prices[pulled_arm] - param.cost) - self.cum_costs

        return np.sum(result), self.n_clicks - np.sum(result), reward, result

    def get_opt(self, t):
        phase = (math.floor(t / self.phase_size)) % self.n_phases+1
        return self.optimal[phase]

    def get_opt_arm(self, t):
        phase = (math.floor(t / self.phase_size)) % self.n_phases + 1
        return self.optimal_price_idx[phase]
    def compute_reward_max(self): # not used

        return np.max(list(self.optimal.values()))


    def compute_clairvoyant_upperbound(self): # not used

        clairvoyant_upperbound = (sum(list(self.optimal.values())))*self.T/len(self.probabilities.keys())

        return clairvoyant_upperbound

    def compute_reward_min(self): # not used
        optimal = {}
        optimal_price_idx = {}

        for phase in range(1, self.n_phases + 1):
            optimal_price_idx[phase] = np.argmin(self.probabilities[phase] * (param.prices - param.cost))
            optimal[phase] = self.probabilities[phase][optimal_price_idx[phase]] * self.n_clicks * (
                        param.prices[optimal_price_idx[phase]] - param.cost) - self.cum_costs

        return np.min(list(optimal.values()))