from utils.UCB5 import *
import numpy as np
import utils.projectParameters as param
from utils.Cumsum import CUMSUM


class CusumUCBLearner(UCB5Learner):
    def __init__(self, n_arms, M=100, eps=0.05, h=5, alpha=0.01):
        super().__init__(n_arms)
        self.change_detection = [CUMSUM(M, eps, h) for _ in range(n_arms)]
        self.num_converted_per_arm = [0 for _ in range(n_arms)]
        self.num_total_samples_per_arm = [0 for _ in range(n_arms)]
        self.valid_t_per_arm = [0 for _ in range(n_arms)]
        self.detections = [[] for _ in range(n_arms)]
        self.alpha = alpha


    def pull_arm(self):
        if np.random.binomial(1, 1 - self.alpha):
            upper_conf = (self.empirical_means + self.confidence) # * (param.prices - param.cost)
            return np.random.choice(np.where(upper_conf == upper_conf.max())[0])
        else:
            return np.random.randint(0, self.n_arms)

    def update(self, pulled_arm, reward):
        self.t += 1
        for sample in reward[3]:
            if self.change_detection[pulled_arm].update(sample):
                self.detections[pulled_arm].append(self.t)
                self.num_total_samples_per_arm[pulled_arm] = 0
                self.num_converted_per_arm[pulled_arm] = 0
                self.valid_t_per_arm[pulled_arm] = 0
                self.change_detection[pulled_arm].reset()

        self.valid_t_per_arm[pulled_arm] += 1
        self.update_observations(pulled_arm, reward)
        self.empirical_means[pulled_arm] = self.num_converted_per_arm[pulled_arm] / self.num_total_samples_per_arm[pulled_arm]
        total_valid_t = sum([x for x in self.valid_t_per_arm])
        for a in range(self.n_arms):
            n_samples = self.num_total_samples_per_arm[a]
            self.confidence[a] = (2 * np.log(total_valid_t) / n_samples) ** 0.5 if n_samples > 0 else np.inf

    def update_observations(self, pulled_arm, reward):
        super().update_observations(pulled_arm, reward[2])
        self.num_converted_per_arm[pulled_arm] += reward[0]
        self.num_total_samples_per_arm[pulled_arm] += reward[0] + reward[1]