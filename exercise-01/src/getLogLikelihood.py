import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def getLogLikelihood(means: list[list[float]], weights: list[float], covariances: list[list[list[float]]],
                     X: list[list[float]]):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Insert your code here for subtask 6a#####
    # Setup as numpy
    means = np.array(means)
    weights = np.array(weights)
    covariances = np.array(covariances)
    X = np.array(X)

    # Number of data points
    N = X.shape[0]

    # Number of gaussians
    K = means.shape[0]

    log_likelihood = 0
    for n in range(N):
        # Calculate the likelihood for each Gaussian component for the given data point
        sum_for_data_point = 0
        for k in range(K):
            sum_for_data_point += weights[k] * multivariate_normal.pdf(X[n], mean=means[k], cov=covariances[:, :, k]);
        # Add the log of the likelihood for this data point to the total log likelihood
        log_likelihood += np.log(sum_for_data_point)

    return log_likelihood
