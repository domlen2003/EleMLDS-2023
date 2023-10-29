import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def getLogLikelihood(means_list: list[list[float]], weight_list: list[float],
                     covariance_matrices: list[list[list[float]]], datapoints: list[list[float]]):
    """Log Likelihood estimation

        N is number of data points

        D is the dimension of the data points

        K is number of gaussians

        Parameters
        ----------
        means_list: list[list[float]]
            Mean for each Gaussian KxD
        weight_list: list[float]
            Weight vector 1xK for K Gaussians
        covariance_matrices: list[list[list[float]]]
            Covariance matrices for each gaussian DxDxK
        datapoints: list[list[float]]
            Input data NxD

        Returns
        -------
        log_likelihood:
            The log Likelihood estimation
    """

    # Setup as numpy
    means = np.array(means_list)
    weights = np.array(weight_list)
    covariances = np.array(covariance_matrices)
    X = np.array(datapoints)

    # Number of data points
    N = X.shape[0]

    # Number of gaussians
    K = means.shape[0]

    log_likelihood = 0
    for n in range(N):
        sum_for_data_point = 0
        for k in range(K):
            # calculate probability
            probability = multivariate_normal.pdf(X[n], mean=means[k], cov=covariances[:, :, k])
            # weight the probability
            weighted_probability = weights[k] * probability
            # sum over K
            sum_for_data_point += weighted_probability
        # sum over N
        log_likelihood += np.log(sum_for_data_point)

    return log_likelihood
