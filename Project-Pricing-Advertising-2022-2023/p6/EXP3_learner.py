"""
REFERENCES
- theoretical guarantees: https://people.eecs.berkeley.edu/~jiantao/2902021spring/scribe/EE290_Lecture_10.pdf
- algorithm: https://courses.cs.washington.edu/courses/cse599s/14sp/scribes/lecture9/lecture9_draft.pdf
"""
import numpy as np
from utils.learners.Learner import Learner

class EXP3(Learner):
    def __init__(self, n_arms, gamma = 5e-6):
        super().__init__(n_arms)
        self.n_arms = n_arms
        self.gamma = gamma
        self.weights = np.ones(n_arms)
        self.rewards = np.zeros(n_arms)
    
    def pull_arm(self):
        #print(f"weights(nan)={np.sum(np.isnan(self.weights))}")
        probabilities = (1 - self.gamma) * self.weights / np.sum(self.weights) + self.gamma / self.n_arms
        #print(f"{np.sum(np.isnan(probabilities))}")
        arm = np.random.choice(np.arange(self.n_arms), p=probabilities)
        return arm
    
    def update(self, pulled_arm, reward):
        self.t +=1
        self.update_observations(pulled_arm, reward)
        self.rewards[pulled_arm] += reward
        estimate = self.rewards[pulled_arm] / (self.weights[pulled_arm] + 1e-10) # add epsilon to avoid division by zero
        self.weights[pulled_arm] *= np.exp(self.gamma * estimate / self.n_arms)
        self.weights /= np.sum(self.weights) # normalize weights to prevent overflow
        #print(f"weight factor: {np.exp(self.gamma * estimate / self.n_arms)}\tweight: {self.weights[pulled_arm]}")