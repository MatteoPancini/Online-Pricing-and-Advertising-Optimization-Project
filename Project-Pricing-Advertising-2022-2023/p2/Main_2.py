import matplotlib.pyplot as plt
import Environment2 as env
from GPTS_Learner2 import *
from GPUCB_Learner2 import *
import utils.parameters as param
from tqdm import tqdm


T = 365
n_experiments = 15

class_id = 1

env = env.Bidding_Environment_2(class_id)

optimal_bid = env.optimal_bid
n_arms = env.n_arms

ts_rewards_per_experiment = []
ucb_rewards_per_experiment = []

cumregret_ts = []
cumregret_ucb = []

cumreward_ts = []
cumreward_ucb = []

for e in tqdm(range(0,n_experiments)):

    # Create environment and learners
    gpts_optimizer = GPTS_Learner2(param.bid_values, class_id=class_id)
    gpucb_optimizer = GPUCB_Learner2(param.bid_values, class_id=class_id)

    for t in tqdm(range (0,T)):
        # Pull arms and update learners

        # Thompson sampling
        pulled_arm = gpts_optimizer.pull_arm()
        reward = env.round(pulled_arm)
        gpts_optimizer.update(pulled_arm, *reward)

        # UCB
        pulled_arm = gpucb_optimizer.pull_arm()
        reward = env.round(pulled_arm)
        gpucb_optimizer.update(pulled_arm, *reward)

    # Store collected rewards
    ts_rewards_per_experiment.append(gpts_optimizer.collected_rewards)
    ucb_rewards_per_experiment.append(gpucb_optimizer.collected_rewards)

    cumregret_ts.append(np.cumsum(optimal_bid - ts_rewards_per_experiment[e]))
    cumregret_ucb.append(np.cumsum(optimal_bid - ucb_rewards_per_experiment[e]))

    cumreward_ts.append(np.cumsum(ts_rewards_per_experiment[e]))
    cumreward_ucb.append(np.cumsum(ucb_rewards_per_experiment[e]))


plt.figure(0)
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
plt.plot(np.mean(cumregret_ts, axis=0), 'r')
plt.plot(np.mean(cumregret_ucb, axis=0), 'g')
plt.fill_between(range(T), np.mean(cumregret_ts, axis=0) - np.std(cumregret_ts, axis=0), np.mean(cumregret_ts, axis=0) + np.std(cumregret_ts, axis=0), color = "r", alpha = 0.2)
plt.fill_between(range(T), np.mean(cumregret_ucb, axis=0) - np.std(cumregret_ucb, axis=0), np.mean(cumregret_ucb, axis=0) + np.std(cumregret_ucb, axis=0), color = "g", alpha = 0.2)
plt.legend(["TS","UCB"])
plt.show()

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Instantaneous Regret")
plt.plot(np.mean(optimal_bid - ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(optimal_bid - ucb_rewards_per_experiment, axis=0), 'g')
plt.fill_between(range(T), np.mean(optimal_bid - ts_rewards_per_experiment, axis=0) - np.std(optimal_bid - ts_rewards_per_experiment, axis=0), np.mean(optimal_bid - ts_rewards_per_experiment, axis=0) + np.std(optimal_bid - ts_rewards_per_experiment, axis=0), color ="r", alpha = 0.2)
plt.fill_between(range(T), np.mean(optimal_bid - ucb_rewards_per_experiment, axis=0) - np.std(optimal_bid - ucb_rewards_per_experiment, axis=0), np.mean(optimal_bid - ucb_rewards_per_experiment, axis=0) + np.std(optimal_bid - ucb_rewards_per_experiment, axis=0), color ="g", alpha = 0.2)
plt.legend(["TS","UCB"])
plt.show()

plt.figure(2)
plt.xlabel("t")
plt.ylabel("Cumulative Reward")
plt.plot(np.mean(cumreward_ts, axis=0), 'r')
plt.plot(np.mean(cumreward_ucb, axis=0), 'g')
plt.fill_between(range(T), np.mean(cumreward_ts, axis=0) - np.std(cumreward_ts, axis=0), np.mean(cumreward_ts, axis=0) + np.std(cumreward_ts, axis=0), color = "r", alpha = 0.2)
plt.fill_between(range(T), np.mean(cumreward_ucb, axis=0) - np.std(cumreward_ucb, axis=0), np.mean(cumreward_ucb, axis=0) + np.std(cumreward_ucb, axis=0), color = "g", alpha = 0.2)
plt.legend(["TS","UCB"])
plt.show()

plt.figure(3)
plt.xlabel("t")
plt.ylabel("Instantaneous Reward")
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(ucb_rewards_per_experiment, axis=0), 'g')
plt.fill_between(range(T), np.mean(ts_rewards_per_experiment, axis=0) - np.std(ts_rewards_per_experiment, axis=0), np.mean(ts_rewards_per_experiment, axis=0) + np.std(ts_rewards_per_experiment, axis=0), color = "r", alpha = 0.2)
plt.fill_between(range(T), np.mean(ucb_rewards_per_experiment, axis=0) - np.std(ucb_rewards_per_experiment, axis=0), np.mean(ucb_rewards_per_experiment, axis=0) + np.std(ucb_rewards_per_experiment, axis=0), color = "g", alpha = 0.2)
plt.legend(["TS","UCB"])
plt.show()