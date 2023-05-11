import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from User_Classes import *

class Environment_Pricing(): #m
    def __init__(self, n_arms, p):
        self.classes = [
            UserClass(name = 'C1'),
            UserClass(name = 'C2'),
            UserClass(name = 'C3')
        ]
        self.prices = [50, 100, 150, 200, 250]
        self.time = 0
        self.n_arms = n_arms
        self.p = p

    def get_conversion_price_probability(self, class_index, price_index):
        prob = self.classes[class_index].get_conversion_probabilities()[price_index]
        return prob

    def round(self, class_index, price_index):
        p = self.get_conversion_price_probability(class_index, price_index)
        reward = np.random.binomial(1, p)
        self.time += 1
        return reward

if __name__ == "__main":
    env = Environment_Pricing()
    x = env.prices              # list of prices
    classes = env.classes # list of categories

    colors = ['r', 'g', 'b']
    plt.figure(figsize=(14,8))
    for i in range(len(classes)):
        for j in range(len(env.prices)):
            y = classes[i].get_conversion_probabilities()
            smooth = interp1d(x, y, kind='cubic')
            plt.plot(x, smooth(x), color=colors[i], label = classes[i].name)
            plt.scatter(x, y, color=colors[i])
            plt.title("Conversion Rates")
            plt.xlabel("Price (€)")
            plt.ylabel("Conversion Rate")

        plt.legend()
        plt.show()
    #env = Environment_Pricing()
    #purchases = []
    #for i in range(1000):
        # choose a price based on the user class
    #    user_class = np.random.choice(range(0,3))
    #    price_index = np.random.choice(range(0,5))
    #    conversion_rate = env.get_conversion_price_probability(user_class,price_index)
        # generate a Bernoulli sample for this user at this price
    #    purchase = np.random.binomial(n=1, p=conversion_rate)
        # add the purchase to the list of purchases
    #    purchases.append((user_class,price_index, purchase))

    #num_purchases = sum([1 for p in purchases if p[0] == 2 and p[1] == 4 and p[2] == 1])
    #num_users = sum([1 for p in purchases if p[0] == 2 and p[1] == 4])
    #conversion_rate =  num_purchases / num_users if num_users > 0 else 0
    #print(f"Class: {env.classes[2].name}, Price: {env.prices[4]}, Conversion rate: {conversion_rate}")
