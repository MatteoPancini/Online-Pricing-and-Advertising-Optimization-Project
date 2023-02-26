"""Initialize the prior distribution for each arm.
Sample a value from each arm's prior distribution.
Select the arm with the highest sampled value.
Observe the reward for the selected arm.
Update the prior distribution of the selected arm based on the observed reward.
If the number of iterations is less than the sliding window size, go back to step 2.
If the number of iterations is greater than the sliding window size, remove the oldest observation from the window and update the prior distribution of the corresponding arm.
Repeat steps 2-7 for a fixed number of iterations or until convergence."""


from TS_Learner import TS_Learner
import numpy as np

class SWTS_Learner(TS_Learner):
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.window_size = window_size
        self.pulled_arms = np.array([])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.pulled_arms = np.append(self.pulled_arms, pulled_arm)
        for arm in range(self.n_arms):
            #count the number of times we pulled the arm we are updating
            n_samples = np.sum(self.pulled_arms[-self.window_size:] == arm)
            cum_rew = np.sum(self.rewards_per_arm[arm][-n_samples:])  if n_samples > 0 else 0
            self.beta_parameters[arm, 0] = cum_rew + 1.0
            self.beta_parameters[arm, 1] = n_samples - cum_rew + 1
