import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from utils.User_Classes import *
from p1.advertising_environment import Advertising_Environment
from utils.parameters import cost_sigma, clicks_sigma
    
class Environment_Pricing(): #m
    def __init__(self, n_arms, p):
        self.classes = [
            UserClass(name = 'C1'),
            UserClass(name = 'C2'),
            UserClass(name = 'C3')
        ]
        self.prices = [50, 100, 150, 200, 250]
        self.time = 0
        self.n_arms = n_arms
        self.p = p
        self.ad_env = Advertising_Environment()

    def get_conversion_price_probability(self, class_index, price_index):
        prob = self.classes[class_index].get_conversion_probabilities()[price_index]
        return prob

    def round(self, class_index, price_index, bid=1):
        prices = [50,100,150,200,250]
        clicks = self.ad_env.generate_observations(noise_std_clicks=clicks_sigma, bid=bid, index=class_index)
        conversion_prob = np.random.binomial(1, self.get_conversion_price_probability(0, price_index))
        margin = prices[price_index] - (prices[price_index]/100)*30
        costs = self.ad_env.get_total_cost(noise_std_cost=cost_sigma, bid=bid, index=class_index)
        reward = clicks * conversion_prob * margin - costs
        self.time += 1
        return reward