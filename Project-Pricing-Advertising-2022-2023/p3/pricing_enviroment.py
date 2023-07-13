import numpy as np

class Princing_Environment_3(): #m
    def __init__(self, n_arms, user_class):
        self.user_class = user_class
        self.time = 0
        self.n_arms = n_arms

    def round(self, user_class, price_idx, n):
        successes = np.random.binomial(n, user_class.get_conversion_probabilities()[price_idx]) # Number of samples n
        return successes