from p1.advertising_environment import Advertising_Environment
from utils.tools import calculate_margin
import numpy as np
from utils.projectParameters import classes, prices, bid_values

ad_env = Advertising_Environment()

#Bid-Cost curve
def bid_cost_fn(bid):
    return np.log(bid+1)

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

# Define function to find the optimal parameters for a single class
def get_optimal_parameters(class_index):

    optimal_class_price = prices[0]
    optimal_class_bid, optimal_class_reward = optimal_bid, reward = find_optimal_bid_for_class(class_index, 0)
    
    for i in range(1, len(prices)):
        optimal_bid, reward = find_optimal_bid_for_class(class_index, i)
        if reward > optimal_class_reward:
            optimal_class_reward = reward
            optimal_class_price = prices[i]
            optimal_class_bid = optimal_bid


    return optimal_class_price, optimal_class_bid, optimal_class_reward