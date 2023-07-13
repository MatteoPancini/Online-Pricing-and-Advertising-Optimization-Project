from utils.User_Classes import UserClass
from p1.advertising_environment import Advertising_Environment
from utils.tools import calculate_margin
import numpy as np

classes = [UserClass('C1'), UserClass('C2'), UserClass('C3')]
prices = [50,100,150,200,250]
bid_values = np.linspace(0.01, 3, num=100)
#Bid-Cost curve
def bid_cost_fn(bid):
    return np.log(bid+1)

ad_env = Advertising_Environment()
# Define function to calculate daily reward for a single class


def calculate_reward_for_class(class_index, price_index, bid):
    clicks = ad_env.generate_observations(0,bid, class_index)
    conversion_prob = classes[class_index].get_conversion_probabilities()[price_index]
    #margin = prices[price_index] - (prices[price_index]/100)*30
    margin = calculate_margin(prices[price_index])

    #costs = clicks * bid_cost_fn(bid)
    costs = ad_env.get_total_cost(0,bid,class_index)
    reward = clicks * conversion_prob * margin - costs
    return reward

# Define function to find the optimal bid for a single class
def find_optimal_bid_for_class(class_index, price_index):
    rewards = np.array([calculate_reward_for_class(class_index, price_index, bid) for bid in bid_values])
    optimal_bid_index = np.argmax(rewards)
    optimal_bid = bid_values[optimal_bid_index]
    return optimal_bid, rewards[optimal_bid_index]




def get_optimal_parameters(class_index):
    i = class_index

    optimal_class_reward = -np.inf
    for j in range(len(prices)):
        optimal_price = prices[j]
        optimal_bid, reward = find_optimal_bid_for_class(i, j)
        if reward > optimal_class_reward:
            optimal_class_reward = reward

    return optimal_price, optimal_bid

if __name__ == "__main__":




    # Find optimal prices and bids for each class
    optimal_prices = []
    optimal_bids = []
    for i in range(len(classes)):
        optimal_class_price = None
        optimal_class_bid = None
        optimal_class_reward = -np.inf
        for j in range(len(prices)):
            optimal_price = prices[j]
            optimal_bid, reward = find_optimal_bid_for_class(i, j)
            if reward > optimal_class_reward:
                optimal_class_price = optimal_price
                optimal_class_bid = optimal_bid
                optimal_class_reward = reward
        optimal_prices.append(optimal_class_price)
        optimal_bids.append(optimal_class_bid)

# Calculate total reward
    total_reward = sum([calculate_reward_for_class(i, prices.index(optimal_prices[i]) , optimal_bids[i]) for i in range(0,3)])
    print("Optimal Prices: ", optimal_prices)
    print("Optimal Bids: ", optimal_bids)
    print("Total Reward: ", total_reward)


'''Optimal Prices:  [200, 200, 150]
Optimal Bids:  [1.0,1.0,1.0]
Total Reward:  21441.417667270194
'''