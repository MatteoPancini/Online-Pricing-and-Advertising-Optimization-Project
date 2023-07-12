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
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm,0] + alpha
        self.beta_parameters[pulled_arm,1] = self.beta_parameters[pulled_arm,1] + beta
