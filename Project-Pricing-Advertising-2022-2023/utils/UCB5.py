from utils.learners.Learner import Learner
import numpy as np
import utils.projectParameters as param

class UCB5Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf] * n_arms)
        self.n_samples = np.zeros(n_arms)

    def pull_arm(self):
        upper_conf = (self.empirical_means + self.confidence)*(param.prices - param.cost)
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * self.n_samples[pulled_arm] + reward[0]) / (self.n_samples[pulled_arm] + reward[0] +reward[1])
        for a in range(self.n_arms):
            self.confidence[a] = (2 * np.log(self.t) / self.n_samples[a]) ** 0.5 if self.n_samples[a] > 0 else np.inf
        self.update_observations(pulled_arm, reward[2])
        self.n_samples[pulled_arm] += reward[0] + reward[1]
