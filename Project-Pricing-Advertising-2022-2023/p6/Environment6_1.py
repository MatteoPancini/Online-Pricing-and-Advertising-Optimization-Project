import utils.projectParameters as param
from utils.clairvoyant_tools import get_optimal_parameters
import numpy as np
import math


class Environment6_1:
    def __init__(self, class_id, T):
        self.class_id = class_id
        self.pricing_probabilities_per_phase = param.pricing_probabilities_per_phase
        self.T = T
        self.n_phases = len(self.pricing_probabilities_per_phase.keys())
        self.phase_size = math.ceil(T / self.n_phases)
        _, optimum_bid, _ = get_optimal_parameters(class_index=class_id)
        self.n_clicks = np.round(param.n_clicks_per_bid_per_class[self.class_id](optimum_bid)).astype(np.int32)
        self.cum_costs = param.total_cost_per_bid_per_class[self.class_id](optimum_bid)
        self.optimal_price_idx = {}
        self.optimal = {}
        self.n_arms = len(param.prices)
        
        for phase in range(1, self.n_phases + 1):
            self.optimal_price_idx[phase] = np.argmax(self.pricing_probabilities_per_phase[phase]*(param.prices-param.cost))
            self.optimal[phase] = self.pricing_probabilities_per_phase[phase][self.optimal_price_idx[phase]] * self.n_clicks * (param.prices[self.optimal_price_idx[phase]] - param.cost) - self.cum_costs

    def round(self, pulled_arm, t):
        phase = min(math.floor(t / self.phase_size) + 1, self.n_phases)
        result = np.random.binomial(1, self.pricing_probabilities_per_phase[phase][pulled_arm], self.n_clicks)
        reward = np.sum(result) * (param.prices[pulled_arm] - param.cost) - self.cum_costs
        return np.sum(result), self.n_clicks - np.sum(result), reward, result

    def get_opt(self, t):
        phase = min(math.floor(t / self.phase_size) + 1, self.n_phases)
        return self.optimal[phase]