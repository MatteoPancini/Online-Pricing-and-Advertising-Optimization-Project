from utils.learners.Learner import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
'''
class GPTS_Learner(Learner):
    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms)*10
        self.pulled_arms = []
        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        #kernel = C(100, (100, 1e6)) * RBF(1, (1e-1, 1e1))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha = alpha**2, normalize_y=True, n_restarts_optimizer= 9)

#Override to update the value of the pulled arms list
    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])
#update the means and sigmas with the new predictions
    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x,y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

#calls both update lists
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

#returns the index of the max value drawn from the arm normal distribution
    def pull_arm(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        return np.argmax(sampled_values)

'''
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

""" class Learner:
    #Initializes the Learner class with the number of arms, and creates empty lists for rewards per arm and collected rewards.
    def _init_(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        #list of emty lists, one list for every arm
        self.rewards_per_arm = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])

    # input is pulled arm and environmental reward
    # update the list reward per arm and collect reward
    def update_observations(self, pulled_arm, reward):
        #append the reward value to the list related to the arm
        #pulled by the learn and passed as input
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward) """

class Learner:
    def _init_(self, n_arms):
        """

        :param n_arms: number of arms of the environment that can be pulled
        :var t: current round
        :var rewards_per_arm: list containing the rewards collected when pulled the corresponding arm
        :var collected_rewards: nparray containing the rewards collected for each time step
        """
        self.n_arms = n_arms
        # current round:
        self.t = 0
        # empty list of n_arms elems:
        self.rewards_per_arm = x = [[] for i in range(n_arms)]
        # rewards collected at each round:
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        """
        Update attributes of the learner storing the history of reward observations

        :param pulled_arm: arm pulled during the last time step
        :param reward: reward obtained from the pulled arm during the last time step
        """
        # append collected reward to list of rewards associated to the pulled arm:
        self.rewards_per_arm[pulled_arm].append(reward)
        # append collected reward to all the reward collected up to now:
        self.collected_rewards = np.append(self.collected_rewards, reward)



""" class GPTS_Learner(Learner):
    def _init_(self, n_arms, arms):
        super()._init_(n_arms)
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms)*10
        self.pulled_arms = []
        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        #kernel = C(100, (100, 1e6)) * RBF(1, (1e-1, 1e1))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha = alpha**2, normalize_y=True, n_restarts_optimizer= 9)

#Override to update the value of the pulled arms list
    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])
#update the means and sigmas with the new predictions
    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x,y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

#calls both update lists
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

#returns the index of the max value drawn from the arm normal distribution
    def pull_arm(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        return np.argmax(sampled_values)

 """

class GPTS_Learner(Learner):
    def _init_(self, arms):
        super()._init_(arms.shape[0])
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms)*10
        self.pulled_arms = []
        alpha = 1.0
        kernel = RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha**2, normalize_y=True, n_restarts_optimizer=9)

    def update_observations(self, pulled_arm, reward):
        super().update_observations(pulled_arm, reward)
        self.pulled_arms.append(self.arms[pulled_arm])

    def update_observations_bulk(self, pulled_arms, rewards):
        for sample in range(len(pulled_arms)):
            self.update_observations(pulled_arms[sample], rewards[sample])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def update_bulk(self, pulled_arms, rewards):
        self.t += len(pulled_arms)
        self.update_observations_bulk(pulled_arms, rewards)
        self.update_model()


    def pull_arm(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        return np.argmax(sampled_values)