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


class Environment():
    def __init__(self):
        self.classes = [UserClass('C1',0,0), UserClass('C2',0,1), UserClass('C3',1,0)]
    def generate_observations(self, noise_std, bid, index):
        func = self.classes[index].get_click_bids(bid)
        return func + np.random.normal(0, noise_std, size = func.shape)

    def get_cumulative_cost(self, bid, noise_std, index):
        clicks = self.classes[index].get_click_bids(bid)
        avg_cost = np.cumsum(clicks) #/ np.arange(1, len(clicks) + 1) ** 0.5  # define a concave average cost curve
        return avg_cost + np.random.normal(0, noise_std, size=avg_cost.shape)  # add Gaussian noise to the average cost

    #def get_cumulative_cost(self, bid, noise_std):
    #    clicks = self.classes[0].get_click_bids(bid)
    #    return np.cumsum(clicks + np.random.normal(0, noise_std, size=clicks.shape))

env = Environment()
n_obs = 50
noise_std = 5.0
bids = np.linspace(0.0, 2.50, 20)
# Initialize arrays to store observed bids and clicks
x_obs = np.array([])
y_obs = np.array([])
clicks_per_bid = []
# Generate observations by randomly selecting bids and adding noise to the expected clicks
for i in range(0, 20):
    new_x_obs = np.random.choice(bids, 1)
    new_y_obs = env.generate_observations(noise_std, new_x_obs, 0)

    x_obs = np.append(x_obs, new_x_obs)
    y_obs = np.append(y_obs, new_y_obs)

# To use Gaussian processes, we need to normalize the data. In this case, the data is already normalized.
# Then specify a kernel function, a good choice is to set scale = l = 1.
# We train the GP and estimate the hyperparameters by maximizing the marginal likelihood.
    # Convert the bid and click observations to arrays and normalize the data
    X = np.atleast_2d(x_obs).T
    Y = y_obs.ravel()

    # Set the kernel hyperparameters and initialize the kernel
    theta = 1.0
    l = 1.0
    kernel = C(theta, (1e-3, 1e3)) * RBF(l, (1e-3, 1e3))
    # Initialize the Gaussian Process Regressor and fit it to the data
    gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, normalize_y=False, n_restarts_optimizer=10)
    gp.fit(X,Y)
    # Generate predictions for the bid range using the trained GP
    x_pred = np.atleast_2d(bids).T
    y_pred, sigma = gp.predict(x_pred, return_std = True)
    # Plot the results, including the true click values, the observed click values, the predicted click values,
    # and the 95% confidence intervals
    #plt.figure(i)
    #plt.plot(x_pred, env.get_cumulative_cost(x_pred, noise_std,0), 'r:', label = r'$C1_CS$')
    plt.plot(x_pred, (1.0 - np.exp(-10.0 * x_pred)) * 200, 'g:', label=r'$C1$')
    #plt.plot(x_pred, (1.0 - np.exp(-3.0 * x_pred)) * 50, 'b:', label=r'$C3$')
    #plt.plot(X.ravel(), Y, 'ro', label = u'Observed Clicks')
    #plt.plot(x_pred, y_pred, 'b-', label=u'Predicted Clicks')
    #plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
    #         np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96*sigma)[::-1]]),
    #         alpha=.5, fc='b', ec='None', label='95% conf interval')
    #clicks_per_bid.append((1.0 - np.exp(-10.0 * x_pred)) * 200)
    plt.xlabel('$Bid$')
    plt.ylabel('$Click$')
    plt.legend(loc = 'lower right')
    plt.show()
#print(x_pred)
#plt.figure(0)
#plt.ylabel('y')
#plt.xlabel('x')
#plt.plot(np.cumsum(np.mean(clicks_per_bid, axis =0)), 'b')
#plt.show()
