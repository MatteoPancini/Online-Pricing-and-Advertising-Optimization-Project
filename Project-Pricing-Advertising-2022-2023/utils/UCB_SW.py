from utils.UCB5 import *
import numpy as np
import utils.projectParameters as param


class SWUCB(UCB5Learner):
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.window_size = window_size
        self.pulled_arms = np.array([])

        self.num_samples_per_day = np.array([])
        self.num_conversions_per_day = np.array([])

    def pull_arm(self):
        upper_conf = (self.empirical_means + self.confidence)
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])



    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward[2])
        self.pulled_arms = np.append(self.pulled_arms, pulled_arm)

        self.num_samples_per_day = np.append(self.num_samples_per_day, reward[0] + reward[1])
        self.num_conversions_per_day = np.append(self.num_conversions_per_day, reward[0])

        for arm in range(self.n_arms):
            n_samples = np.sum(self.num_samples_per_day[-self.window_size:] * (self.pulled_arms[-self.window_size:] == arm))
            n_conversions = np.sum(self.num_conversions_per_day[-self.window_size:] * (self.pulled_arms[-self.window_size:] == arm))

            self.empirical_means[arm] = n_conversions / n_samples if n_samples > 0 else 0
            self.confidence[arm] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf
