from utils.learners.Learner import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class GPUCB_Learner(Learner):
    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf] * n_arms)
        self.pulled_arms = []
        alpha = 1
        kernel = RBF(1, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=True, n_restarts_optimizer=9)

    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])
    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x,y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)
    def pull_arm(self):
        upper_conf = self.empirical_means + self.confidence
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])

    def update(self, pull_arm, reward):
        self.t += 1
        self.empirical_means[pull_arm] = (self.empirical_means[pull_arm] * (self.t - 1) + reward) / self.t
        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf
        self.update_observations(pull_arm, reward)
        self.update_model()
