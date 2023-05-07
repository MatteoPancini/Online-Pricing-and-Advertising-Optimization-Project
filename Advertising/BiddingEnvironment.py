import numpy as np
#Maps the bid to the corresponding number of clicks
def fun(x):
    return 100 * (1.0 - np.exp(-4*x + 3*x)**3)

class BiddingEnvironment():
    def __init__(self, bids, sigma):
        self.bids = bids
        #Initialize means of reward functions
        self.means = fun(bids)
        #Initialize sigma array equal for all bid values
        self.sigmas = np.ones(len(bids))*sigma

#Given the bid index, returns the reward
    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
    
