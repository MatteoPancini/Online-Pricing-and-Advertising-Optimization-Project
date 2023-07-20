import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.User_Classes import UserClass
from utils.clairvoyant_tools import find_optimal_bid_for_class
from utils.tools import calculate_margin
from p3.pricing_enviroment import Princing_Environment_3
from p3.bidding_enviroment import Bidding_Environment_3
from p3.GPTS_learner import GPTS_Learner3
from p3.GPUCB_learner import GPUCB_Learner3
from utils.learners.TS_Learner import TS_Learner

import warnings

warnings.filterwarnings("ignore")
# %%
# Environment
n_arms = 100

prices = [50, 100, 150, 200, 250]
bids = np.linspace(0.01, 3.0, n_arms)
clicks_sigma = 50
cost_sigma = 10

user = UserClass(name='C1')

T = 365
n_experiments = 100

gpts_rewards_per_experiment = []
gpucb_rewards_per_experiment = []

opt_vec = []
for price_index in range(len(prices)):
    opt_vec.append(find_optimal_bid_for_class(user.user_index, price_index))

#print(opt_vec)  # reward of optimum bid
optimum_bid_reward = np.max(np.array(opt_vec))
#print(optimum_bid_reward)  # optimum bid

# %%
# Create Environment
pr_env = Princing_Environment_3(n_arms, user)
bid_env = Bidding_Environment_3(bids, clicks_sigma, cost_sigma, user, n_arms)
# %%
for e in tqdm(range(n_experiments), desc='Number of experiments'):
    # Create Learner
    pricing_learner = TS_Learner(len(prices))
    bidding_GPTS_learner = GPTS_Learner3(n_arms, arms=bids)
    ts_rewards = []

    # generate empty deque
    arms = []

    # Simulate experiments
    for d in range(T):
        # choose arms
        pricing_pulled_arm = pricing_learner.pull_arm()
        arms.append(pricing_pulled_arm)
        # Calculate the conversion rate of the price choose
        conv_rate = pricing_learner.beta_parameters[pricing_pulled_arm, 0] / (
                    pricing_learner.beta_parameters[pricing_pulled_arm, 0]
                    + pricing_learner.beta_parameters[pricing_pulled_arm, 1])

        # GPTS
        # Fixed the price, I can find the bid that maximize the expected reward
        bidding_pulled_arm = bidding_GPTS_learner.pull_arm(conv_rate=conv_rate,
                                                           margin=calculate_margin(prices[pricing_pulled_arm]))
        # empty daily reward
        n_clicks, costperclick = bid_env.round(pulled_arm=bidding_pulled_arm)
        # how many successes?
        ts_successes = pr_env.round(user, pricing_pulled_arm, n_clicks)
        # update beta
        pricing_learner.update(pricing_pulled_arm, ts_successes)
        # Reward of the day
        daily_reward = ts_successes * calculate_margin(prices[pricing_pulled_arm]) - costperclick * n_clicks
        costs = costperclick * n_clicks
        bidding_GPTS_learner.update(pulled_arm=bidding_pulled_arm, costs=costs, clicks=n_clicks)
        # save daily reward
        ts_rewards.append(daily_reward)

    gpts_rewards_per_experiment.append(ts_rewards)
# %%
for e in tqdm(range(n_experiments), desc='Number of experiments'):
    # Create Learner
    pricing_learner = TS_Learner(len(prices))
    bidding_GPUCB_learner = GPUCB_Learner3(n_arms, arms=bids)
    ucb_rewards = []

    # generate empty deque
    arms = []

    # Simulate experiments
    for d in range(T):
        # choose arms
        pricing_pulled_arm = pricing_learner.pull_arm()
        arms.append(pricing_pulled_arm)
        # Calculate the conversion rate of the price choose
        conv_rate = pricing_learner.beta_parameters[pricing_pulled_arm, 0] / (
                    pricing_learner.beta_parameters[pricing_pulled_arm, 0]
                    + pricing_learner.beta_parameters[pricing_pulled_arm, 1])

        # GPUCB
        # Fixed the price, I can find the bid that maximize the expected reward
        bidding_pulled_arm = bidding_GPUCB_learner.pull_arm(conv_rate=conv_rate,
                                                            margin=calculate_margin(prices[pricing_pulled_arm]))
        # empty daily reward
        n_clicks, costperclick = bid_env.round(pulled_arm=bidding_pulled_arm)
        # how many successes?
        ucb_successes = pr_env.round(user, pricing_pulled_arm, n_clicks)
        # update beta
        pricing_learner.update(pricing_pulled_arm, ucb_successes)
        # Reward of the day
        daily_reward = ucb_successes * calculate_margin(prices[pricing_pulled_arm]) - costperclick * n_clicks
        costs = costperclick * n_clicks
        bidding_GPUCB_learner.update(pull_arm=bidding_pulled_arm, costs=costs, clicks=n_clicks)
        # save daily reward
        ucb_rewards.append(daily_reward)

    gpucb_rewards_per_experiment.append(ucb_rewards)
# %% md
# Cumulative Regret
# %%
plt.figure(0)
plt.ylabel("Cumulative Regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(optimum_bid_reward - gpucb_rewards_per_experiment, axis=0)), 'r', label='GPUCB')
for index, line in enumerate(gpts_rewards_per_experiment):
    plt.plot(np.nancumsum(optimum_bid_reward - line, axis=0), "r",
             alpha=0.3 / np.power(len(gpts_rewards_per_experiment), 2 / 3))
plt.plot(np.cumsum(np.mean(optimum_bid_reward - gpts_rewards_per_experiment, axis=0)), 'g', label='GPTS')
for index, line in enumerate(gpucb_rewards_per_experiment):
    plt.plot(np.nancumsum(optimum_bid_reward - line, axis=0), "g",
             alpha=0.3 / np.power(len(gpts_rewards_per_experiment), 2 / 3))
plt.legend()
plt.show()
# %% md
# Cumulative Reward
# %%
plt.figure(0)
plt.ylabel("Cumulative Reward")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(gpts_rewards_per_experiment, axis=0)), 'r', label='GPTS')
for index, line in enumerate(gpts_rewards_per_experiment):
    plt.plot(np.nancumsum(line, axis=0), "r", alpha=0.3 / np.power(len(gpts_rewards_per_experiment), 2 / 3))
plt.plot(np.cumsum(np.mean(gpucb_rewards_per_experiment, axis=0)), 'g', label='GPUCB')
for index, line in enumerate(gpucb_rewards_per_experiment):
    plt.plot(np.nancumsum(line, axis=0), "g", alpha=1 / np.power(len(gpucb_rewards_per_experiment), 2 / 3))
plt.legend()
plt.show()
# %% md
# Istantaneous Regret
# %%
plt.figure(0)
plt.ylabel("Istantaneous Regret")
plt.xlabel("t")
plt.plot(np.mean(optimum_bid_reward - gpts_rewards_per_experiment, axis=0), 'r', label='GPTS')
for index, line in enumerate(gpts_rewards_per_experiment):
    plt.plot(optimum_bid_reward - line, "r", alpha=0.3 / np.power(len(gpts_rewards_per_experiment), 2 / 3))
plt.plot(np.mean(optimum_bid_reward - gpucb_rewards_per_experiment, axis=0), 'g', label='GPUCB')
for index, line in enumerate(gpucb_rewards_per_experiment):
    plt.plot(optimum_bid_reward - line, "g", alpha=0.3 / np.power(len(gpucb_rewards_per_experiment), 2 / 3))
plt.legend()
plt.show()
# %% md
# Instantaneous Reward
# %%
plt.figure(0)
plt.ylabel("Istantaneous Reward")
plt.xlabel("t")
plt.plot(np.mean(gpts_rewards_per_experiment, axis=0), 'r', label='GPTS')
for line in gpts_rewards_per_experiment:
    plt.plot(line, "r", alpha=0.3 / np.power(len(gpts_rewards_per_experiment), 2 / 3))

plt.plot(np.mean(gpucb_rewards_per_experiment, axis=0), 'g', label='GPUCB')
for line in gpucb_rewards_per_experiment:
    plt.plot(line, "g", alpha=0.3 / np.power(len(gpucb_rewards_per_experiment), 2 / 3))
plt.legend()
plt.show()