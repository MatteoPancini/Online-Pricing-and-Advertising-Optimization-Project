import numpy as np
from utils.clairvoyant_tools import get_optimal_parameters, classes

if __name__ == "__main__":

    # Find optimal prices and bids for each class
    optimal_prices = []
    optimal_bids = []
    optimal_rewards = []
    for i in range(len(classes)):
        price, bid, reward = get_optimal_parameters(i)
        optimal_prices.append(price)
        optimal_bids.append(bid)
        optimal_rewards.append(reward)

# Calculate total reward
    print("Optimal Prices: ", optimal_prices)
    print("Optimal Bids: ", optimal_bids)
    print("Total Reward: ", sum(optimal_rewards))


'''Optimal Prices:  [200, 200, 150]
Optimal Bids:  [1.0,1.0,1.0]
Total Reward:  21441.417667270194
'''