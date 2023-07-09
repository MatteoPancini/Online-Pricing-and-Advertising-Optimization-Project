from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
from utils.learners.Learner import Learner

class GPTS_Learner3(Learner):
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
    def pull_arm(self, pricing_learner, price_idx, margin):
        if self.t < self.n_arms:
            return self.t  # % self.n_arms

        try:
            conv_rate = pricing_learner.beta_parameters[price_idx, 0] / (pricing_learner.beta_parameters[price_idx, 0]
                                                                         + pricing_learner.beta_parameters[
                                                                             price_idx, 1])
        except ZeroDivisionError:
            conv_rate = 0
            print('DIV 0')
        poisson = pricing_learner.poisson_vector[price_idx, 0] + 1

        exp_rew = np.random.normal(self.acc_means * (np.ones(shape=self.n_arms)
                                                     * margin * conv_rate * poisson - self.cost_means), 50)
        bid_idx = np.argmax(exp_rew)

        return bid_idx