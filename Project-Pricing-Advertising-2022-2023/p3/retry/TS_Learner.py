import utils.projectParameters as param
import numpy as np

class Learner:
    def __init__(self, n_arms):
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

class TSLearner(Learner):
    def __init__(self, n_arms):
        """
        Implements a learner following the Thompson Sampling algorithm

        :var beta_parameters: parameters for the Beta distribution used to draw the random sample
        """
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))

    def pull_arm(self):
        """
        Choose the arm to pull as the one giving the highest random sample from the corresponding Beta distribution

        :return: number of the arm to pull
        """
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])*(param.prices-param.cost))
        return idx

    def update(self, pulled_arm, reward):
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

    def update_bulk(self, pulled_arms, rewards):
        for i, arm in enumerate(pulled_arms):
            self.update(arm, [rewards[0][i], rewards[1][i], rewards[2][i]])

