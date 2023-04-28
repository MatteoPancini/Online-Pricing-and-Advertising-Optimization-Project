"""This file runs two learners, TS_Learner and Greedy_Learner,
on a multi-armed bandit problem with a given set of arm probabilities.
It calculates and plots the regret of each learner over a given number
 of experiments and time steps."""

import numpy as np
import matplotlib.pyplot as plt
from Environment import *
from TS_Learner import *
from Greedy_Learner import *
# Define the problem settings
n_arms = 4
p = np.array([0.15, 0.1, 0.1, 0.35]) #probabilities of success for each arm
opt = p[3] #optimal arm is the one with the highest probability of success

T = 300 #time steps for each experiment

n_experiments = 1000

ts_rewards_per_experiment = [] #list to store the collected rewards for TS_Learner over each experiment
gr_reward_per_experiment = [] #list to store the collected rewards for Greedy_Learner over each experiment

# Loop over the experiments
for e in range (0, n_experiments):
    env = Environment(n_arms=n_arms, probabilities = p)
    ts_learner = TS_Learner(n_arms=n_arms)
    gr_learner = Greedy_Learner(n_arms=n_arms)
    for t in range(0,T):
        #Thompson sampling
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

        # Greedy
        pulled_arm = gr_learner.pull_arm()
        reward = env.round(pulled_arm)
        gr_learner.update(pulled_arm, reward)


    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    gr_reward_per_experiment.append(gr_learner.collected_rewards)


plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis = 0)), 'r')
plt.plot(np.cumsum(np.mean(opt - gr_reward_per_experiment, axis = 0)), 'g')
plt.legend(["TS", "Greedy"])
plt.show()