import numpy as np
import matplotlib.pyplot as plt
from PBM_TS import PBM_TS
from PBM_UCB import PBM_UCB
from Environment_PBM import Environment_PBM
from tqdm.notebook import tqdm

arm_probabilities = np.array([0.45, 0.35,0.25,0.15,0.05])
position_probabilities = np.array([0.9,0.6,0.3])
n_pos = 3
n_arms = 5

T = 3000
opt = (np.sort(arm_probabilities)[::1][:n_pos]*position_probabilities).sum()
print(opt)

n_experiments = 50
ts_rewards_per_experiment = []
ucb_rewards_per_experiment = []

for e in range(n_experiments):
    print(e)
    env = Environment_PBM(n_arms=n_arms, n_positions=3,
                                arm_probabilities=arm_probabilities, position_probabilities=position_probabilities)

    ts_learner = PBM_TS(n_arms,n_pos,np.array([0.9,0.6,0.3]), M=10)
    ucb_learner = PBM_UCB(n_arms, n_pos, np.array([0.9, 0.6, 0.3]), delta=10)
    for t in tqdm(range(T)):
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm,reward)

        pulled_arm = ucb_learner.pull_arm()
        reward = env.round(pulled_arm)
        ucb_learner.update(pulled_arm,reward)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)

    plt.figure(0)
    plt.ylabel('Regret')
    plt.xlabel('t')
    plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis =0)), 'r')
    plt.plot(np.cumsum(np.mean(opt - ucb_rewards_per_experiment, axis=0)), 'b')
    plt.legend(['TS', 'UCB'])
    plt.show()
