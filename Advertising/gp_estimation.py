import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Define a function that returns the expected number of clicks for a given bid value
def n(x):
    # This function is used to generate the "true" click values, which are then perturbed with noise to generate observed clicks
    return (1.0 - np.exp(-5.0*x)) * 100

# Define a function that generates observed click values by adding noise to the expected click value
def generate_observations(x, noise_std):
    return n(x) + np.random.normal(0, noise_std, size = n(x).shape)

# Set the number of observations to generate, the range of bid values to use, and the noise standard deviation
n_obs = 50
noise_std = 5.0
bids = np.linspace(0.0, 1.0, 20)
# Initialize arrays to store observed bids and clicks
x_obs = np.array([])
y_obs = np.array([])


# Generate observations by randomly selecting bids and adding noise to the expected clicks
for i in range(0, n_obs):
    new_x_obs = np.random.choice(bids, 1)
    new_y_obs = generate_observations(new_x_obs, noise_std)

    x_obs = np.append(x_obs, new_x_obs)
    y_obs = np.append(y_obs, new_y_obs)

# To use Gaussian processes, we need to normalize the data. In this case, the data is already normalized.
# Then specify a kernel function, a good choice is to set scale = l = 1.
# We train the GP and estimate the hyperparameters by maximizing the marginal likelihood.
    # Convert the bid and click observations to arrays and normalize the data
    X = np.atleast_2d(x_obs).T
    Y = y_obs.ravel()

    # Set the kernel hyperparameters and initialize the kernel
    theta = 1.0
    l = 1.0
    kernel = C(theta, (1e-3, 1e3)) * RBF(l, (1e-3, 1e3))
    # Initialize the Gaussian Process Regressor and fit it to the data
    gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, normalize_y=True, n_restarts_optimizer=10)
    gp.fit(X,Y)
    # Generate predictions for the bid range using the trained GP
    x_pred = np.atleast_2d(bids).T
    y_pred, sigma = gp.predict(x_pred, return_std = True)
    # Plot the results, including the true click values, the observed click values, the predicted click values,
    # and the 95% confidence intervals
    plt.figure(i)
    plt.plot(x_pred, n(x_pred), 'r:', label = r'$n(x)$')
    plt.plot(X.ravel(), Y, 'ro', label = u'Observed Clicks')
    plt.plot(x_pred, y_pred, 'b-', label=u'Predicted Clicks')
    plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
             np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96*sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% conf interval')
    plt.xlabel('$x$')
    plt.ylabel('$n(x)$')
    plt.legend(loc = 'lower right')
    plt.show()
