import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


"""
Given the bid returns the expected number of clicks --> to be estimated
"""
def n(x):
    return (1.0 - np.exp(-5.0*x)) * 100


"""
Observation : noise + n(x)
"""
def generate_observation(x, noise_std):
    return n(x) + np.random.normal(0, noise_std, size=n(x).shape)

if __name__ == '__main__':

    n_obs = 50
    bids = np.linspace(0.0, 1.0, 20)

    x_obs = np.array([])
    y_obs = np.array([])

    noise_std = 5.0


    for i in range(n_obs):
        new_x_obs = np.random.choice(bids, 1)
        new_y_obs = generate_observation(new_x_obs, noise_std)

        x_obs = np.append(x_obs, new_x_obs)
        y_obs = np.append(y_obs, new_y_obs)

        # for the GP
        x = np.atleast_2d(x_obs).T
        Y = y_obs.ravel()

        theta = 1.0
        l = 1.0
        kernel = C(theta, (1e-3, 1e3)) * RBF(l, (1e-3, 1e3))

        gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=noise_std**2,
                                      normalize_y=True,
                                      n_restarts_optimizer=9)

        gp.fit(x, Y)


        x_pred = np.atleast_2d(bids).T
        y_pred, sigma = gp.predict(x_pred, return_std=True)



        plt.figure(i)
        plt.plot(x_pred, n(x_pred), 'r:', label= r'$n(x)$')
        plt.plot(x.ravel(), Y, 'ro', label= u'Observed Clicks')
        plt.plot(x_pred, y_pred, 'b-', label=u'Predicted Clicks')


        # uncertainty 95%
        plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
                 np.concatenate([y_pred - 1.96*sigma, (y_pred + 1.96*sigma)[::-1]]),
                 alpha = .5, fc='b', ec='None', label='95% confidence interval')

        plt.xlabel('$x$')
        plt.ylabel('$n(x)$')
        plt.legend(loc='lower right')
        plt.show()



