from utils.learners.Learner import Learner
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C



class GPUCBLearner(Learner):
    def __init__(self, arms):
        super().__init__(arms.shape[0])
        self.arms = arms
        self.confidence = np.array([np.inf] * self.n_arms)
        self.empirical_means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 7
        self.pulled_arms = []
        alpha = 1.0
        kernel = RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=True, n_restarts_optimizer=9)

    def pull_arm(self):
        upper_conf = self.empirical_means + self.confidence
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])

    def update(self, pull_arm, reward):
        self.t += 1
        self.update_observations(pull_arm, reward)
        self.update_model()
        self.confidence = self.sigmas * np.sqrt(2 * np.log(self.t))

    def update_bulk(self, pulled_arms, rewards):
        self.t += len(pulled_arms)
        self.update_observations_bulk(pulled_arms, rewards)
        self.update_model()

    def update_observations(self, pulled_arm, reward):
        super().update_observations(pulled_arm, reward)
        self.pulled_arms.append(self.arms[pulled_arm])

    def update_observations_bulk(self, pulled_arms, rewards):
        for sample in range(len(pulled_arms)):
            self.update_observations(pulled_arms[sample], rewards[sample])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.empirical_means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)