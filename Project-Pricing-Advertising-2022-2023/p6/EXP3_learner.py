"""
REFERENCES
- theoretical guarantees: https://people.eecs.berkeley.edu/~jiantao/2902021spring/scribe/EE290_Lecture_10.pdf
- algorithm: https://courses.cs.washington.edu/courses/cse599s/14sp/scribes/lecture9/lecture9_draft.pdf
"""
import numpy as np
from utils.learners.Learner import Learner
import random
from math import sqrt, log
import utils.projectParameters as param

class EXP3(Learner):
    def __init__(self, n_arms, upperbound_total_reward=1.0, reward_min=0.0, reward_max=1.0):
        """
        Implements a learner following the EXP3 algorithm

        :var gamma: parameters for the ...
        """
        super().__init__(n_arms)
        self.weights = np.ones(n_arms)
        self.reward_min = reward_min  # the min of all the price phases
        self.reward_max = reward_max  # the max of all price phases
        upperbound_total_reward_scaled = (upperbound_total_reward - self.reward_min) / (self.reward_max - self.reward_min)
        exponential = 3
        self.gamma = min(1.0, sqrt(n_arms * log(n_arms) / (upperbound_total_reward_scaled * (exponential - 1))))
        # self.gamma = np.sqrt(2*log(n_arms)/(n_arms*upperbound_total_reward))

    def pull_arm(self):
        """
        Choose the arm to pull as a random draw form  a discrete distribution

        :return: number of the arm to pull
        """

        probability_distribution = distr(self.weights, self.gamma)
        idx = draw(probability_distribution)
        return idx

    def update(self, pulled_arm, reward):
        """
        Update history of observations and weights of the arm

        :param pulled_arm: number of the arm pulled at the last time step
        :param reward: (binary) reward obtained at the last time step
        """
        self.t += 1

        scaled_reward = ((reward[0] / (reward[0] + reward[1]) * (
                param.prices[pulled_arm] - param.prices[pulled_arm] * 0.3)) - self.reward_min) / (
                                self.reward_max - self.reward_min)  # rewards scaled to 0,1

        probability_distribution = distr(self.weights, self.gamma)
        estimated_reward = 1.0 * scaled_reward / probability_distribution[pulled_arm]
        self.weights[pulled_arm] *= np.exp(
            estimated_reward * self.gamma / self.n_arms)
        self.update_observations(pulled_arm, reward[2])


# draw: [float] -> int
# pick an index from the given list of floats proportionally
# to the size of the entry (i.e. normalize to a probability
# distribution and draw according to the probabilities).


def draw(weights):
    choice = random.uniform(0, sum(weights))
    choice_index = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choice_index

        choice_index += 1


def distr(weights, gamma=1.0):
    the_sum = float(sum(weights))
    return tuple((1.0 - gamma) * (w / the_sum) + (gamma / len(weights)) for w in weights)