import numpy as np
from utils.User_Classes import UserClass

class Environment:
    def __init__(self, user_class):
        self.user_class = user_class
        self.class_id = user_class.user_index
        self.prices = [50, 100, 150, 200, 250]
        self.n_arms_prices = len(self.prices)
        self.time = 0
        self.pricing_probabilities = user_class.get_conversion_probabilities()
        self.optimal_price_idx = np.argmax(self.pricing_probabilities * (self.prices -  (self.prices/100)*30)) #highest margin
        self.n_arms_bids = len(self.bids)
        self.bids = np.linspace(0.01, 3.0, 100)
        self.clicks = self.generate_observations(noise_std_clicks=0, bids=self.bids, user_class=user_class)
        self.cum_costs = self.get_total_cost(bids=self.bids, index=self.class_id)
        optimal_values = self.pricing_probabilities[self.optimal_price_idx] * self.clicks * (self.prices[self.optimal_price_idx] - self.prices[self.optimal_price_idx]*30/100) - self.cum_costs
        self.optimal = np.max(optimal_values)
        self.optimal_bid_idx = np.argmax(optimal_values)
        
    def generate_observations(self, user_class, noise_std_clicks, bids):
        observations = []
        for bid in bids:
            func = user_class.get_click_bids(bid)
            observations.append(func)
        observations = np.concatenate(observations)
        return observations + np.random.normal(0, noise_std_clicks, size=observations.shape)

    def get_conversion_price_probability(self, class_index, price_index):
        prob = self.classes[class_index].get_conversion_probabilities()[price_index]
        return prob
    
    def get_total_cost(self, bids, index, noise_std_cost = 200):
        costs = []
        for bid in bids:
            func = self.classes[index].get_total_cost(bid)
            costs.append(func)
        costs = np.concatenate(costs)
        return costs + np.random.normal(0, noise_std_cost, size=costs.shape)

    def round(self, pulled_bids_arm, pulled_prices_arm):
        bid = self.bids[pulled_bids_arm]
        price = self.prices[pulled_prices_arm]
        n_clicks = self.user_class.get_click_bids(bid)
        cum_cost = self.user_class.get_total_cost(bid)
        result = np.random.binomial(1, self.pricing_probabilities[pulled_prices_arm], n_clicks)
        reward = np.sum(result) * (price - price*30/100) - cum_cost
        return np.sum(result), n_clicks, cum_cost, reward