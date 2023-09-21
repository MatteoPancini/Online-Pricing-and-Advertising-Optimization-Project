from utils.learners.Learner import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class GPUCB_Learner3(Learner):
    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.empirical_clicks_means = np.zeros(n_arms)
        self.clicks_confidence = np.array([np.inf] * n_arms)
        self.empirical_costs_means = np.zeros(n_arms)
        self.costs_confidence = np.array([np.inf] * n_arms)

        self.pulled_arms = []

        self.clicks_per_arm = [[] for i in range(n_arms)]
        self.costs_per_arm = [[] for i in range(n_arms)]
        self.collected_clicks = np.array([])
        self.collected_costs = np.array([])

        alpha_clicks = 10
        kernel_clicks =   C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp_clicks = GaussianProcessRegressor(kernel=kernel_clicks, alpha=alpha_clicks ** 2, normalize_y=True, n_restarts_optimizer=9)

        alpha_costs = 0.5
        kernel_costs =  C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp_costs = GaussianProcessRegressor(kernel=kernel_costs, alpha=alpha_costs, normalize_y=False, n_restarts_optimizer=1)

    def update_observations(self, pulled_arm, clicks, costs):
        self.clicks_per_arm[pulled_arm].append(clicks)
        self.costs_per_arm[pulled_arm].append(costs)
        self.collected_clicks = np.append(self.collected_clicks, clicks)
        self.collected_costs = np.append(self.collected_costs, costs)
        
        self.pulled_arms.append(self.arms[pulled_arm])


    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_clicks
        self.gp_clicks.fit(x,y)
        self.clicks_means, self.clicks_sigmas = self.gp_clicks.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.clicks_sigmas = np.maximum(self.clicks_sigmas, 1e-2)

        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_costs
        self.gp_costs.fit(x,y)
        self.costs_means, self.costs_sigmas = self.gp_costs.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.costs_sigmas = np.maximum(self.costs_sigmas, 1e-2)

    def pull_arm(self, conv_rate, margin):
        upper_conf_clicks = self.empirical_clicks_means + self.clicks_confidence
        lower_conf_costs = self.empirical_costs_means - self.costs_confidence

        sampled_reward = conv_rate * upper_conf_clicks * margin - lower_conf_costs
        
        return np.random.choice(np.where(sampled_reward == sampled_reward.max())[0])

    def update(self, pull_arm, clicks, costs):
        self.t += 1
        self.empirical_clicks_means[pull_arm] = (self.empirical_clicks_means[pull_arm] * (self.t - 1) + clicks) / self.t
        for a in range(self.n_arms):
            n_samples = len(self.clicks_per_arm[a])
            self.clicks_confidence[a] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf
        
        self.empirical_costs_means[pull_arm] = (self.empirical_costs_means[pull_arm] * (self.t - 1) + costs) / self.t
        for a in range(self.n_arms):
            n_samples = len(self.costs_per_arm[a])
            self.costs_confidence[a] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf
        

        self.update_observations(pull_arm, clicks, costs)
        self.update_model()