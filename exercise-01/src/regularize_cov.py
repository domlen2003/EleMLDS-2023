import numpy as np


def regularize_cov(covariance, epsilon):
    # regularize a covariance matrix, by enforcing a minimum
    # value on its singular values. Explanation see exercise sheet.
    #
    # INPUT:
    #  covariance: matrix
    #  epsilon:    minimum value for singular values
    #
    # OUTPUT:
    # regularized_cov: reconstructed matrix

    #####Insert your code here for subtask 6d#####
    
    regularized_cov = np.zeros(np.shape(covariance)) # init matrix

    regularized_cov = covariance + (epsilon * np.eye(len(covariance))) # regularize covariances

    return regularized_cov
