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

if __name__ == "__main__":
    env = Advertising_Environment()
    #n_obs = 1 #just for simplicity
    noise_std_clicks = 10.0
    noise_std_cost = 2.0
    bids = np.linspace(0.0, 1, 20)
    T = 365
    # Initialize arrays to store observed bids and clicks
    x_obs = np.array([])
    y_obs = np.array([])
    #total_cost= np.zeros(len(bids))
    bids = np.linspace(0.0, 1, 20)
    x_pred = np.atleast_2d(bids).T
    cost_curve = env.get_total_cost(noise_std_cost, x_pred, 0)
    plt.plot(x_pred, cost_curve, 'r:', label=r'Bid-Cost Total')
    plt.xlabel('$Bid$')
    plt.ylabel('$Cost Total$')
    plt.legend(loc='lower right')
    plt.show()