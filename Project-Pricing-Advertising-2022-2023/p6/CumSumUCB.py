from utils.learners.Learner import Learner
import numpy as np
import utils.projectParameters as param


class UCBLearner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf] * n_arms)
        self.n_samples = np.zeros(n_arms)

    def pull_arm(self):
        upper_conf = (self.empirical_means + self.confidence)*(np.array(param.prices) * 0.7)
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * self.n_samples[pulled_arm] + reward[0]) / (self.n_samples[pulled_arm] + reward[0] +reward[1])
        for a in range(self.n_arms):
            self.confidence[a] = (2 * np.log(self.t) / self.n_samples[a]) ** 0.5 if self.n_samples[a] > 0 else np.inf
        self.update_observations(pulled_arm, reward[2])
        self.n_samples[pulled_arm] += reward[0] + reward[1]

class Cusum:
    def __init__(self, M, eps, h):
        self.M = M
        self.eps = eps
        self.h = h
        self.t = 0
        self.reference = 0
        self.g_plus = 0
        self.g_minus = 0

    def update(self, sample):
        self.t += 1
        if self.t <= self.M:
            self.reference += sample / self.M
            return 0
        else:
            s_plus = (sample - self.reference) - self.eps
            s_minus = -(sample - self.reference) - self.eps
            self.g_plus = max(0, self.g_plus + s_plus)
            self.g_minus = max(0, self.g_minus + s_minus)
            return self.g_plus > self.h or self.g_minus > self.h

    def reset(self):
        self.t = 0
        self.reference = 0
        self.g_minus = 0
        self.g_plus = 0

class CusumUCBLearner(UCBLearner):
    def __init__(self, n_arms, M=100, eps=0.05, h=5, alpha=0.01):
        super().__init__(n_arms)
        self.change_detection = [Cusum(M, eps, h) for _ in range(n_arms)]
        self.n_valid_conversions_per_arm = [0 for _ in range(n_arms)]
        self.n_valid_samples_per_arm = [0 for _ in range(n_arms)]
        self.valid_t_per_arm = [0 for _ in range(n_arms)]
        self.detections = [[] for _ in range(n_arms)]
        self.alpha = alpha

    def pull_arm(self):
        if np.random.binomial(1, 1 - self.alpha):
            upper_conf = (self.empirical_means + self.confidence) * (np.array(param.prices) * 0.7)
            return np.random.choice(np.where(upper_conf == upper_conf.max())[0])
        else:
            return np.random.randint(0, self.n_arms)

    def update(self, pulled_arm, reward):
        self.t += 1
        for sample in reward[3]:
            if self.change_detection[pulled_arm].update(sample):
                self.detections[pulled_arm].append(self.t)
                self.n_valid_samples_per_arm[pulled_arm] = 0
                self.n_valid_conversions_per_arm[pulled_arm] = 0
                self.valid_t_per_arm[pulled_arm] = 0
                self.change_detection[pulled_arm].reset()
        self.valid_t_per_arm[pulled_arm] += 1
        self.update_observations(pulled_arm, reward)
        self.empirical_means[pulled_arm] = self.n_valid_conversions_per_arm[pulled_arm] / self.n_valid_samples_per_arm[pulled_arm]
        total_valid_t = sum([x for x in self.valid_t_per_arm])
        for a in range(self.n_arms):
            n_samples = self.n_valid_samples_per_arm[a]
            self.confidence[a] = (2 * np.log(total_valid_t) / n_samples) ** 0.5 if n_samples > 0 else np.inf

    def update_observations(self, pulled_arm, reward):
        super().update_observations(pulled_arm, reward[2])
        self.n_valid_conversions_per_arm[pulled_arm] += reward[0]
        self.n_valid_samples_per_arm[pulled_arm] += reward[0] + reward[1]