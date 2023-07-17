import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from utils.User_Classes import UserClass

#Definition of Advertising Environment
class Advertising_Environment():
    def __init__(self):
        #Class vector
        self.classes = [
            UserClass(name = 'C1'),
            UserClass(name = 'C2'),
            UserClass(name = 'C3')
            ]
    #Adds gaussian noise to bid-click function
    def generate_observations(self, noise_std_clicks, bid, index):
        func = self.classes[index].get_click_bids(bid)
        return func + np.random.normal(0, noise_std_clicks, size = func.shape)
    def get_total_cost(self, noise_std_cost, bid, index):
        func = self.classes[index].get_total_cost(bid)
        return func + np.random.normal(0, noise_std_cost, size = func.shape)