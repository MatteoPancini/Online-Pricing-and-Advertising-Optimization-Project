from bandit_algorithms.Environment import Environment
import numpy as np

class Environment_PBM(Environment):
    def __init__(self, n_arms, n_positions, arm_probabilities, position_probabilities):
        super().__init__(n_arms, arm_probabilities)
        self.n_arms = n_arms
        self.n_positions = n_positions
        self.arm_probabilities = arm_probabilities
        self.position_probabilities = position_probabilities
        assert n_positions == len(position_probabilities)
        assert n_arms == len(arm_probabilities)

    def round(self, super_arm):
        assert len(super_arm) == self.n_positions
        position_obs = np.random.binomial(1, self.position_probabilities)
        arm_probabilities = np.random.binomial(1, self.arm_probabilities[super_arm])
        return arm_probabilities*position_obs


