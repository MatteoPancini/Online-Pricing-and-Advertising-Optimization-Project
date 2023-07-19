import numpy as np
from utils.User_Classes import UserClass
from utils.projectParameters import clicks_sigma, cost_sigma

class Environment:
    def __init__(self, user_class):
        self.user_class = user_class
        self.class_id = user_class.user_index
        self.prices = [50, 100, 150, 200, 250]
        self.bids = np.linspace(0.01, 3.0, 100)
        self.n_arms_prices = len(self.prices)
        self.time = 0
        self.pricing_probabilities = user_class.get_conversion_probabilities()
        self.optimal_price_idx = np.argmax(self.pricing_probabilities * (np.array(self.prices) - (np.array(self.prices)/100)*30))
        self.n_arms_bids = len(self.bids)
        self.bids = np.linspace(0.01, 3.0, 100)
        self.clicks = self.generate_observations(noise_std_clicks=clicks_sigma, bids=self.bids, user_class=user_class)
        self.cum_costs = self.get_total_costs(bids=self.bids, user_class=user_class, noise_std_cost=cost_sigma)
        optimal_values = self.pricing_probabilities[self.optimal_price_idx] * self.clicks * (self.prices[self.optimal_price_idx] - self.prices[self.optimal_price_idx]*30/100) - self.cum_costs
        self.optimal = np.max(optimal_values)
        self.optimal_bid_idx = np.argmax(optimal_values)
        
    def generate_observations(self, user_class, noise_std_clicks, bids):
        observations = np.array([])
        for bid in bids:
            func = user_class.get_click_bids(bid)
            observations = np.concatenate([observations, [func]])
        return observations + np.random.normal(0, noise_std_clicks, size=observations.shape)
    
    def generate_observation(self, user_class, noise_std_clicks, bid):
        observations = user_class.get_click_bids(bid)
        return observations + np.random.normal(0, noise_std_clicks, size=1)

    def get_conversion_price_probability(self, user_class, price_index):
        prob = user_class.get_conversion_probabilities()[price_index]
        return prob
    
    def get_total_costs(self, bids, user_class, noise_std_cost):
        costs = np.array([], dtype=np.float32)
        for bid in bids:
            func = user_class.get_total_cost(bid)
            costs = np.concatenate([costs, [func]])
        return costs + np.random.normal(0, noise_std_cost, size=costs.shape)

    def get_total_cost(self, bid, user_class, noise_std_cost):
        cost = user_class.get_total_cost(bid)
        cost = cost + np.random.normal(0, noise_std_cost, size=1)
        cost = np.maximum(0, cost)
        return cost

    def round(self, pulled_bids_arm, pulled_prices_arm):
        bid = self.bids[pulled_bids_arm]
        price = self.prices[pulled_prices_arm]
        n_clicks = max(0, int(self.generate_observation(self.user_class, noise_std_clicks=clicks_sigma, bid=bid)))
        cum_cost = self.get_total_cost(bid=bid, user_class=self.user_class, noise_std_cost=cost_sigma)
        margin = price - (price/100)*30

        clicks = np.random.binomial(1, self.pricing_probabilities[pulled_prices_arm], n_clicks)
        clicks = np.maximum(0, clicks)
        reward = np.sum(clicks) * margin - cum_cost
        self.time += 1
        return np.sum(clicks), n_clicks, cum_cost, reward