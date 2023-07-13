"""Initialize the prior distribution for each arm.
Sample a value from each arm's prior distribution.
Select the arm with the highest sampled value.
Observe the reward for the selected arm.
Update the prior distribution of the selected arm based on the observed reward.
Repeat steps 2-5 for a fixed number of iterations or until convergence"""

import numpy as np
from utils.learners.Learner import *

class TS_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        #Beta has 2 parameters per arm
        self.beta_parameters = np.ones((n_arms, 2))

    #Sampling a value for each arm from a Beta and then select the arm associated to the
    #Beta that generated the sample with the max value
    def pull_arm(self):
        #valid_b_values array is created by replacing any non-positive values 
        # in self.beta_parameters[:, 1] with 1.
        valid_b_values = np.where(self.beta_parameters[:, 1] <= 0, 1, self.beta_parameters[:, 1])
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], valid_b_values))
        return idx

    def update(self, pulled_arm, reward):
        #Updates the beta parameters based on the pulled arm and the reward from the environment.
        self.t +=1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm,0] + reward
        self.beta_parameters[pulled_arm,1] = self.beta_parameters[pulled_arm,1] + 1.0 - reward

