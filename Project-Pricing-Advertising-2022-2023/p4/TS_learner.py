from utils.learners.Learner import Learner
import numpy as np

class TS_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        #Beta has 2 parameters per arm
        self.beta_parameters = np.ones((n_arms, 2))

    #Sampling a value for each arm from a Beta and then select the arm associated to the
    #Beta that generated the sample with the max value
    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1]))
        return idx

    def update(self, pulled_arm, alpha, beta, reward):
        #Updates the beta parameters based on the pulled arm and the reward from the environment.
        self.t +=1
        self.update_observations(pulled_arm, reward)
        #print(f"reward {reward},\talpha {alpha},\tbeta {beta}")
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm,0] + beta
        self.beta_parameters[pulled_arm,1] = self.beta_parameters[pulled_arm,1] + alpha

    def update_bulk(self, pulled_arms, rewards):
        for i, arm in enumerate(pulled_arms):
            self.update_simple(arm, [rewards[0][i], rewards[1][i], rewards[2][i]])

    def update_simple(self, pulled_arm, reward):
        """
        Update history of observations and parameters of the Beta distribution corresponding to the pulled arm

        :param pulled_arm: number of the arm pulled at the last time step
        :param reward: (binary) reward obtained at the last time step
        """
        self.t += 1
        self.update_observations(pulled_arm, reward[2])
        # Add the success (if any) to alpha
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward[0]
        # Add the failure (if any) to beta (1-rew=1 iff rew=0 iff fail)
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + reward[1]