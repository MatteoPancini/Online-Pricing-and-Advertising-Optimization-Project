""""""

from Learner import *

class Greedy_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros(n_arms)

    #Selection by maxinizing the expected reward array, but each arm needs to be pulled once
    def pull_arm(self):
        #if not every arm has been pulled once
        if(self.t < self.n_arms):
            return self.t
        #else draw the one that maximizes the expected reward
        idxs = np.argwhere(self.expected_rewards == self.expected_rewards.max()).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.expected_rewards[pulled_arm] = self.expected_rewards[pulled_arm]*(self.t -1) + reward / self.t


