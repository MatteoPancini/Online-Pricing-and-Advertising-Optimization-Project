from Learner import Learner
import numpy as np
import scipy.stats
import scipy.special as sc

class PBM_TS(Learner):
    def __init__(self, n_arms, n_positions, position_probabilities, M=10):
        super().__init__(n_arms)
        self.position_probabilities = position_probabilities
        self.n_arms = n_arms
        self.n_positions = n_positions
        assert n_positions == len(self.position_probabilities)
        self.tilde_N_kl = np.zeros((n_arms, n_positions))
        self.tilde_N_k = np.zeros(n_arms)
        self.N_kl = np.zeros((n_arms, n_positions))
        self.N_k = np.zeros(n_arms)
        self.S_kl = np.zeros((n_arms, n_positions))
        self.S_k = np.zeros(n_arms)
        self.M = M
        self.beta_parameters = np.ones((n_arms, 2))

    def _beta_pdf(self, arm, theta):
        a = self.beta_parameters[arm,0]
        b = self.beta_parameters[arm, 1]
        return scipy.stats.beta.pdf(theta,a,b)

    def _real_pdf(self, arm, theta):
        p = 0
        for pos in range(self.n_positions):
            pos_prob = self.position_probabilities[pos]
            a = self.S_kl[arm, pos]
            b = self.N_kl[arm, pos] - self.S_kl[arm, pos]
            p += sc.xlog1py(b, -theta*pos_prob) + sc.xlogy(a, theta)
            p = sc.betaln(a,b)
            p += a*np.log(pos_prob)
        return np.exp(p)
    def _rejection_sample(self, arm):
        count = 0
        while count < self.M:
            count += 1
            theta = np.random.beta(self.beta_parameters[arm, 0], self.beta_parameters[arm,1])
            u = np.random.uniform()
            if u*self._beta_pdf(arm, theta) < self._real_pdf(arm, theta):
                return theta
        return theta

    def pull_arm(self):
        samples = np.array([self._rejection_sample(k) for k in range(self.n_arms)])
        return np.argsort(samples)[::-1][:self.n_positions]

    def update(self, super_arm, reward):
        self.t += 1
        for pos, arm in enumerate(super_arm):
            self.S_kl[arm, pos] += reward[pos]
            self.tilde_N_kl[arm, pos] += self.position_probabilities[pos]
            self.N_kl[arm,pos] += 1
        self.S_k = self.S_kl.sum(axis=1)
        self.N_k = self.N_kl.sum(axis=1)
        self.tilde_N_k = self.tilde_N_kl.sum(axis=1)

        for arm in super_arm:
            pos = np.argmax(self.tilde_N_kl[arm,:])
            self.beta_parameters[arm, 0] = max(self.S_kl[arm, pos]+1,1)
            self.beta_parameters[arm,1] =max(self.tilde_N_kl[arm, pos] - self.S_kl[arm, pos]+1,1)

        self.update_observations(super_arm, reward)

    def update_observations(self, pulled_arm, reward):
        self.collected_rewards = np.append(self.collected_rewards, reward.sum())