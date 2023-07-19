from utils.User_Classes import UserClass
import numpy as np

n_arms = 100

classes = [UserClass('C1'), UserClass('C2'), UserClass('C3')]
prices = [50, 100, 150, 200, 250]
bid_values = np.linspace(0.01, 3, num=100)

clicks_sigma = 20
cost_sigma = 7


feature_combos = ['00', '01', '10', '11']
features_names = ["F1", "F2"]
