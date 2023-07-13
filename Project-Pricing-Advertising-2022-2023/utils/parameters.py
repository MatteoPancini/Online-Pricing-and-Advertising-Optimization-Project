from utils.User_Classes import UserClass
import numpy as np

classes = [UserClass('C1'), UserClass('C2'), UserClass('C3')]
prices = [50, 100, 150, 200, 250]
bid_values = np.linspace(0.01, 3, num=100)