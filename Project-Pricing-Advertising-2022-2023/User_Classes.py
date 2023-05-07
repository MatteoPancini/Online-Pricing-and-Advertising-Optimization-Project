import numpy as np
"""Defines the 3 user classes and their bid-click, bid-price curves and price conversion probabilities"""

class UserClass():
    def __init__(self, name, f1, f2):
        self.f1 = f1
        self.f2 = f2
        self.name = name
    # In the context of bidding in online advertising, a function that represents the
    # dependence between the number of clicks and the bid should be bounded
    # because it is not realistic to expect an unlimited number of clicks regardless
    # of the bid value
    def get_click_bids(self, bid):
        if self.name == 'C1':
            #Gioavne Appassionato
            return (1.0 - np.exp(-5.0*bid)) * 200
        if self.name == 'C2':
            #Adulto Appassionato
            return (1.0 - np.exp(-5.0*bid)) * 100
        if self.name == 'C3':
            #Giovane Non Appassionato
            return (1.0 - np.exp(-5.0*bid)) * 50

    #Returns the function matching bid to total cost considering clicks per bid
    def get_total_cost(self, bid):
        #Sublinear cost function
        cost_bid = (np.log(bid+1)**0.5)*3
        return cost_bid * self.get_click_bids(bid)

    #Defines probabilities for conversion rates of each class
    def get_conversion_probabilities(self):
        if self.name == 'C1':
            # Giovane Appassionato
            return [0.9, 0.7, 0.6, 0.5, 0.4]
        if self.name == 'C2':
            # Adulto appassionato
            return [0.9, 0.7, 0.5, 0.4, 0.3]
        if self.name == 'C3':
            # Giovane non appassionato
            return [0.7, 0.5, 0.4, 0.3, 0.2]
    #oppure tutte curve tipo \frac{1}{1+e^{\left(4x-5\right)}}
