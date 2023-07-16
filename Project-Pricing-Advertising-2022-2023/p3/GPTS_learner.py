from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
from utils.learners.Learner import Learner

#This learner consider the clicks and the cost
class GPTS_Learner3(Learner):
    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.clicks_means = np.zeros(self.n_arms)
        self.clicks_sigmas = np.ones(self.n_arms)
        self.cost_means = np.zeros(self.n_arms)
        self.cost_sigmas = np.ones(self.n_arms)
        self.pulled_arms = []
        self.collected_clicks = []
        self.collected_costs = []
        alpha_clicks = 1000
        kernel_clicks = C(100, (100, 1e6)) * RBF(10, (1e-1, 1e6))
        self.gp_clicks = GaussianProcessRegressor(kernel=kernel_clicks, alpha=alpha_clicks, normalize_y=False, n_restarts_optimizer=1)

        alpha_cost = 0.3
        kernel_cost = C(0.1, (1, 1e2)) * RBF(0.1, (1, 1e2))
        self.gp_cost = GaussianProcessRegressor(kernel=kernel_cost, alpha=alpha_cost, normalize_y=False, n_restarts_optimizer=1)

    def update_observations_gpts(self, pulled_arm, clicks, costs):
        # self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_clicks = np.append(self.collected_clicks, clicks)
        self.collected_costs = np.append(self.collected_costs, costs)
        self.pulled_arms.append(self.arms[pulled_arm])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_clicks
        self.gp_clicks.fit(x, y)
        self.clicks_means, self.clicks_sigmas = self.gp_clicks.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.clicks_sigmas = np.maximum(self.clicks_sigmas, 30)

        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_costs
        self.gp_cost.fit(x, y)
        self.cost_means, self.cost_sigmas = self.gp_cost.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.cost_sigmas = np.maximum(self.cost_sigmas, 0.01)

    def update(self, pulled_arm, clicks, costs):
        self.t += 1
        self.update_observations_gpts(pulled_arm, clicks, costs)
        self.update_model()

    def pull_arm(self, conv_rate, margin):
        exp_rew = np.random.normal(self.clicks_means * (np.ones(shape=self.n_arms) * margin * conv_rate - self.cost_means), 50) #TODO: check if a variance of 50 is ok
        bid_idx = np.argmax(exp_rew)

        return bid_idx