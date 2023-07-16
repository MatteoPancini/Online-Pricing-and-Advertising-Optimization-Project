import numpy as np

bids = np.linspace(0.01, 3, 100)
prices = np.array([50, 100, 150, 200, 250])
cost = 35

seed = 42


pricing_probabilities_per_user = {
    1: np.array([0.25, 0.35, 0.25, 0.15, 0.10]), # C1 - Giovane Appassionato
    2: np.array([0.15, 0.25, 0.35, 0.30, 0.25]), # C2 - Adulto Appassionato
    3: np.array([0.35, 0.30, 0.20, 0.15, 0.10]), # C3 - Giovane Non Appassionato
}

'''
pricing_probabilities_per_phase = { # User C1
    1: np.array([0.25, 0.35, 0.25, 0.15, 0.10]),  # Phase 1: Regular behaviours (best price is the middle one, April) -> 150€
    # price * probability [12.5, 35, 37.5, 30, 25]
    2: np.array([0.24, 0.32, 0.35, 0.29, 0.18]),  # Phase 2: Holiday Season (higher price, middle one as good gift, Christmas) -> 200€
    # price * probability [12, 32, 52.5, 58, 45]
    3: np.array([0.3, 0.25, 0.15, 0.11, 0.07]),  # Phase 3: Sale Season (second lowest price best one, September) -> 100€
    # price * probability [15, 25, 22.5, 22, 17.5]
}
'''
# Strongest differences
pricing_probabilities_per_phase = { # User C1
    1: np.array([0.25, 0.35, 0.25, 0.15, 0.10]),  # Phase 1: Regular behaviours (best price is the middle one, April) -> 150€
    # price * probability [12.5, 35, 37.5, 30, 25]
    2: np.array([0.1, 0.20, 0.3, 0.4, 0.25]),  # Phase 2: Holiday Season (higher price, middle one as good gift, Christmas) -> 200€
    # price * probability [5, 20, 45, 80, 62.5]
    3: np.array([0.4, 0.25, 0.12, 0.08, 0.05]),  # Phase 3: Sale Season (second lowest price best one, September) -> 100€
    # price * probability [20, 25, 18, 16, 12.5]
}

n_clicks_per_bid_per_class = {
    1: lambda x: (1.0 - np.exp(-5.0*x)) * 200,
    2: lambda x: (1.0 - np.exp(-5.0*x)) * 100,
    3: lambda x: (1.0 - np.exp(-5.0*x)) * 50,
}

total_cost_per_bid_per_class = {
    1: lambda x: (1.0 - np.exp(-5.0 * x)) * 200 * (np.log(x+1)),
    2: lambda x: (1.0 - np.exp(-5.0 * x)) * 100 * (np.log(x+1)),
    3: lambda x: (1.0 - np.exp(-5.0 * x)) * 50 * (np.log(x+1)),
}