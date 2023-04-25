from Learner import Learner
import numpy as np

class PBM_UCB(Learner):
    def __init__(self, n_arms, n_positions, position_probabilities, delta):
        super().__init__(n_arms)
        self.position_probabilities = position_probabilities
        self.n_arms = n_arms
        self.n_positions = n_positions
        assert n_positions == len(self.position_probabilities)
        self.S_kl = np.zeros((n_arms, n_positions))
        self.S_k =np.zeros(n_arms)
        self.N_kl = np.zeros((n_arms, n_positions))
        self.N_k = np.zeros(n_arms)
        self.tilde_N_kl = np.zeros((n_arms, n_positions))
        self.tilde_N_k = np.zeros(n_arms)
        self.delta = delta
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf]*n_arms)

    def pull_arm(self):
        upper_conf = self.empirical_means + self.confidence
        return np.argsort(upper_conf)[::-1][:self.n_positions]

    def update(self, super_arm, reward):
        self.t += 1
        for pos, arm in enumerate(super_arm):
            self.S_kl[arm, pos] += reward[pos]
            self.N_kl[arm, pos] += 1
            self.tilde_N_kl[arm, pos] += self.position_probabilities[pos]

        self.S_k = self.S_kl.sum(axis=1)
        self.N_k = self.N_kl.sum(axis=1)
        self.tilde_N_k = self.tilde_N_kl.sum(axis=1)

        self.empirical_means = self.S_k/self.tilde_N_k
        self.confidence = np.sqrt(self.N_k/self.tilde_N_k)*np.sqrt(self.delta/(2*self.tilde_N_k))
        self.empirical_means[self.N_k == 0] = np.inf
        self.confidence[self.tilde_N_k == 0] = np.inf
        self.update_observations(super_arm, reward)

    def update_observations(self, pulled_arm, reward):
        self.collected_rewards = np.append(self.collected_rewards, reward.sum())
