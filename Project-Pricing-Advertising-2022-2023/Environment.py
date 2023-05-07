import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from User_Classes import *


#Definition of Advertising Environment
class Environment():
    def __init__(self):
        #Class vector
        self.classes = [
            UserClass('C1',0,0),
            UserClass('C2',0,1),
            UserClass('C3',1,0)
            ]
    #Adds gaussian noise to bid-click function
    def generate_observations(self, noise_std, bid, index):
        func = self.classes[index].get_click_bids(bid)
        return func + np.random.normal(0, noise_std, size = func.shape)


env = Environment()
n_obs = 1 #just for simplicity
noise_std = 5.0
bids = np.linspace(0.0, 3, 20)
T = 365
# Initialize arrays to store observed bids and clicks
x_obs = np.array([])
y_obs = np.array([])
total_cost= np.zeros(len(bids))
clicks_per_bid = []
for t in range(0,T):# Generate observations by randomly selecting bids and adding noise to the expected clicks
    for i in range(0, n_obs):
        new_x_obs = np.random.choice(bids, 1)
        x_obs = np.append(x_obs, new_x_obs)
        X = np.atleast_2d(x_obs).T
        x_pred = np.atleast_2d(bids).T
    #plt.figure(i)
        #Computation done for class 0, get total cost by multiplying the two functions
        cost_curve = env.classes[0].get_total_cost(x_pred)
        plt.plot(x_pred, (np.log(x_pred+1)**0.5)*3, 'r:', label=r'$Bid-Cost$')
        plt.plot(x_pred, (1.0 - np.exp(-5.0 * x_pred)) * 200, 'g:', label=r'$Bid-Click$')
        plt.xlabel('$Bid$')
        plt.ylabel('$Click$')
        plt.legend(loc = 'lower right')
        #plt.show()
    total_cost += cost_curve.ravel()
#print(x_pred)
plt.figure()
#total cumulative day average by dividing by 365
plt.plot(x_pred, total_cost/T, 'b:', label='Average Cumulative Daily Click Cost')
plt.xlabel('$Bid$')
plt.ylabel('$Cost$')
plt.legend(loc='upper left')
plt.show()