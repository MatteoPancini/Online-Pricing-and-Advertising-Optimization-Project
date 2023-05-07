import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class UserClass():
    def __init__(self, name, f1, f2):
        self.f1 = f1
        self.f2 = f2
        self.name = name
    # In the context of bidding in online advertising, a function that represents the
    # dependence between the number of clicks and the bid should be bounded
    # because it is not realistic to expect an unlimited number of clicks regardless
    # of the bid value
    def get_click_bids(self, bid):
        if self.name == 'C1':
            #Giovane Appassionato
            return (1.0 - np.exp(-5.0*bid)) * 200
        if self.name == 'C2':
            #Adulto appassionato
            return (1.0 - np.exp(-5.0*bid)) * 100
        if self.name == 'C3':
            #Giovane non appassionato
            return (1.0 - np.exp(-5.0*bid)) * 50
    def get_total_cost(self, bid):
        cost_bid = (np.log(bid+1)**0.5)*3
        return cost_bid * self.get_click_bids(bid)



class Environment():
    def __init__(self):
        self.classes = [
            UserClass('C1',0,0),
            UserClass('C2',0,1),
            UserClass('C3',1,0)
            ]

    def generate_observations(self, noise_std, bid, index):
        func = self.classes[index].get_click_bids(bid)
        return func + np.random.normal(0, noise_std, size = func.shape)

    def get_cumulative_cost(self, bid, noise_std, index):
        clicks = self.classes[index].get_click_bids(bid)
        avg_cost = clicks * 2.5
        #avg_cost = np.cumsum(clicks) / np.arange(1, len(clicks) + 1) ** 0.725  # define a concave average cost curve

        return avg_cost + np.random.normal(0, noise_std, size=avg_cost.shape)  # add Gaussian noise to the average cost

    #def get_cumulative_cost(self, bid, noise_std):
    #    clicks = self.classes[0].get_click_bids(bid)
    #    return np.cumsum(clicks + np.random.normal(0, noise_std, size=clicks.shape))

env = Environment()
n_obs = 10 #just for simplicity
noise_std = 5.0
bids = np.linspace(0.0, 3, 20)
T = 365
# Initialize arrays to store observed bids and clicks
x_obs = np.array([])
y_obs = np.array([])
total_cost= np.zeros(len(bids))
clicks_per_bid = []
for t in range(0,T):# Generate observations by randomly selecting bids and adding noise to the expected clicks
    for i in range(0, 1):
        new_x_obs = np.random.choice(bids, 1)
        x_obs = np.append(x_obs, new_x_obs)
        X = np.atleast_2d(x_obs).T
        x_pred = np.atleast_2d(bids).T
    #plt.figure(i)
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
plt.plot(x_pred, total_cost/T, 'b:', label='Average Cumulative Daily Click Cost')
plt.xlabel('$Bid$')
plt.ylabel('$Cost$')
plt.legend(loc='upper left')
plt.show()