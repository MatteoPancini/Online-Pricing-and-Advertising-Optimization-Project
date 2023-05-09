import numpy as np

class UserClass():
    '''
    Defines the 3 user classes and their bid-click curve, bid-price curve and price conversion probabilities
        F1	F2	C
        0	0	0
        0	1	1
        1	*	2
    '''

    def __init__(self, name = None, f1 = None, f2 = None):
        if name != None:
            self.name = name
            if self.name == "C0":
                self.f1 = 0
                self.f2 = 0
            elif self.name == "C1":
                self.f1 = 0
                self.f2 = 1
            else:
                self.f1 = 1
                self.f2 = 0
        else:
            self.f1 = f1
            self.f2 = f2
            if f1 == 1:
                self.name = "C2"
            elif f2 == 1:
                self.name = "C1"
            else: 
                self.name = "C0"        

    # In the context of bidding in online advertising, a function that represents the
    # dependence between the number of clicks and the bid should be bounded
    # because it is not realistic to expect an unlimited number of clicks regardless
    # of the bid value
    def get_click_bids(self, bid):
        '''Returns the number of clicks given a bid for the instanced class'''
        if self.name == 'C1':
            #Giovane Appassionato
            return (1.0 - np.exp(-5.0*bid)) * 200
        if self.name == 'C2':
            #Adulto Appassionato
            return (1.0 - np.exp(-5.0*bid)) * 100
        if self.name == 'C3':
            #Giovane Non Appassionato
            return (1.0 - np.exp(-5.0*bid)) * 50

    def get_total_cost(self, bid):
        '''Returns the function matching the bid to the total cost, considering the number of clicks per bid'''
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
