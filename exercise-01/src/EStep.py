import numpy as np
from getLogLikelihood import getLogLikelihood
def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####
    N = X.shape[0]
    D = X.shape[1]
    K = len(means)
    gamma = np.zeros((N,K))
    for n in range(N):
        for j in range(K):
            top = weights[j] * getMultiDimNormForEM(means[j], covariances[:,:,j], X[n], D)
            bottom = 0
            for k in range(K):
                bottom += weights[k] * getMultiDimNormForEM(means[k], covariances[:,:,k], X[n], D)
            gamma[n][j] = top / bottom     
    return [getLogLikelihood(means, weights, covariances, X), gamma]

def getMultiDimNormForEM(mean, covariance, x, D):

    divider = 1 / (np.power(2*np.pi, D/2) * np.power(np.linalg.det(covariance),1/2))
    vector = np.asmatrix(x - mean)
    e = (-(1/2) * vector.dot(np.linalg.inv(covariance)).dot(np.transpose(vector))).item()
    multiDimNorm = divider * np.exp(e)
    return multiDimNorm