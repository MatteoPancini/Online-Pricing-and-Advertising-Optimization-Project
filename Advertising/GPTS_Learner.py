from bandit_algorithms.Learner import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class GPTS_Learner(Learner):
    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms)*10
        self.pulled_arms = []
        alpha = 10.0
        #kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        #kernel_BlueCow
        kernel = C(100, (100, 1e6)) * RBF(1, (1e-1, 1e1))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha = alpha**2, normalize_y=True, n_restarts_optimizer= 9)

#Override to update the value of the pulled arms list
    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])
#update the means and sigmas with the new predictions
    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x,y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)
#calls both update lists
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()
#returns the index of the max value drawn from the arm normal distribution
    def pull_arm(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        return np.argmax(sampled_values)

