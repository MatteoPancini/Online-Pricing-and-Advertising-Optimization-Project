"""Selects the arm with the highest expected reward
based on its past observations. The expected rewards are
estimated for each arm and updated incrementally as the learner
pulls arms and observes rewards from the environment.
If an arm has not been pulled at least once, it is selected with equal
probability. """

from Learner import *

class Greedy_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros(n_arms)

    # Selection by maxinizing the expected reward array, but each arm needs to be pulled once
    def pull_arm(self):
        # Check if not every arm has been pulled once
        if(self.t < self.n_arms):
            return self.t
        # Else draw the one that maximizes the expected reward
        idxs = np.argwhere(self.expected_rewards == self.expected_rewards.max()).reshape(-1)
        # Randomly select an arm among the selected arms
        pulled_arm = np.random.choice(idxs)
        return pulled_arm

    def update(self, pulled_arm, reward):
        # Updates the expected reward for the pulled arm based on the reward received from the environment.
        self.t += 1
        self.update_observations(pulled_arm, reward)
        # Update the expected reward value for the pulled arm using a running average
        self.expected_rewards[pulled_arm] = self.expected_rewards[pulled_arm]*(self.t -1) + reward / self.t


