import numpy as np
from utils.User_Classes import UserClass


class Environment:
    def __init__(self, feature):
        self.feature = feature
        self.user_class = self.compute_class_from_features(self.feature)
        self.prices = np.array([50, 100, 150, 200, 250])
        self.bids = np.linspace(0.01, 3.0, 100)
        self.n_bids_arms = len(self.bids)
        self.n_prices_arms = len(self.prices)
        self.pricing_probabilities = self.user_class.get_conversion_probabilities()
        self.optimal_price_idx = np.argmax(self.pricing_probabilities * (self.prices - self.prices * 0.3)) #TODO check this optimal formula
        self.n_clicks = self.user_class.get_click_bids(self.bids)
        self.cum_costs = self.user_class.get_total_cost(self.bids)#param.cum_cost_per_bid_by_feature[self.feature](param.bids)
        self.optimal = np.max(self.pricing_probabilities[self.optimal_price_idx] * self.n_clicks * ( #TODO: checks this optimal formula
                self.prices[self.optimal_price_idx] - self.prices[self.optimal_price_idx] * 0.3) - self.cum_costs)
        self.optimal_bid_idx = np.argmax(self.pricing_probabilities[self.optimal_price_idx] * self.n_clicks * ( #TODO: checks this optimal formula
                self.prices[self.optimal_price_idx] - self.prices[self.optimal_price_idx] * 0.3) - self.cum_costs)

    def round(self, pulled_bids_arm, pulled_prices_arm):
        bid = self.bids[pulled_bids_arm]
        price = self.prices[pulled_prices_arm]
        n_clicks = np.round(max(0,self.user_class.get_click_bids(bid) + np.random.normal(0, 10))).astype(np.int32)
        cum_cost = max(0, self.user_class.get_total_cost(bid) + np.random.normal(0, 200))
        result = np.random.binomial(1, self.pricing_probabilities[pulled_prices_arm], n_clicks)
        reward = np.sum(result) * (price - price*0.3) - cum_cost
        return np.sum(result), n_clicks, cum_cost, reward
    
    def compute_class_from_features(self, f):
        return UserClass(f1=f[0], f2=f[1])