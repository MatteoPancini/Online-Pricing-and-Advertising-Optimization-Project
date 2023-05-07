import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class UserClass():
    def __init__(self, name, f1, f2):
        self.f1 = f1
        self.f2 = f2
        self.name = name

    def get_conversion_probability(self):
        if self.name == 'C1':
            # Giovane Appassionato
            return [0.9, 0.7, 0.6, 0.5, 0.4]
        if self.name == 'C2':
            # Adulto appassionato
            return [0.9, 0.7, 0.5, 0.4, 0.3]
        if self.name == 'C3':
            # Giovane non appassionato
            return [0.7, 0.5, 0.4, 0.3, 0.2]

class Environment_Pricing():
    def __init__(self):
        self.classes = [
            UserClass('C1', 0, 0),
            UserClass('C2', 0, 1),
            UserClass('C3', 1, 0)
        ]
        self.prices = [50, 100, 150, 200, 250]

    def get_conversion_price_probability(self, index,price):
        prob = self.classes[index].get_conversion_probability()[price]
        return prob


env = Environment_Pricing()
purchases = []
for i in range(1000):
    # choose a price based on the user class
    user_class = np.random.choice(range(0,3))
    price_index = np.random.choice(range(0,5))
    conversion_rate = env.get_conversion_price_probability(user_class,price_index)
    # generate a Bernoulli sample for this user at this price
    purchase = np.random.binomial(n=1, p=conversion_rate)
    # add the purchase to the list of purchases
    purchases.append((user_class,price_index, purchase))

num_purchases = sum([1 for p in purchases if p[0] == 2 and p[1] == 4 and p[2] == 1])
num_users = sum([1 for p in purchases if p[0] == 2 and p[1] == 4])
conversion_rate =  num_purchases / num_users if num_users > 0 else 0
print(f"Class: {env.classes[2].name}, Price: {env.prices[4]}, Conversion rate: {conversion_rate}")
