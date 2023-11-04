import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####

    # init key variables
    N = X.shape[0]
    D = X.shape[1]
    K = gamma.shape[1]
    M = np.zeros([K])
    weights = np.zeros([K])
    means = np.zeros((K,D))
    covariances = np.zeros((D, D, K))

    #calculate output for each gaussian 
    for j in range(K): 

        # calculate N-dach_j 
        for n in range(N):
            M[j] += gamma[n][j]
        weights[j] = M[j] / N  # final weights 

        # sum for mean calculation 
        for n in range(N): 
            means[j] += gamma[n][j] * X[n]
        means[j] = (1/M[j]) * means[j] # final means

        # sum for covariance calculation
        for n in range(N):  
            covariances[:,:,j] += gamma[n][j] * np.transpose(np.asmatrix(X[n] - means[j])).dot(np.asmatrix( X[n] - means[j])) 
        covariances[:,:,j] = (1/M[j]) * covariances[:,:,j] #final covariances

    return weights, means, covariances, getLogLikelihood(means, weights, covariances, X)
