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

    # init key variables
    N = X.shape[0]
    D = X.shape[1]
    K = len(means)
    gamma = np.zeros((N,K))

    # outer sum over all datapoints
    for n in range(N): 

        # for all gaussians
        for j in range(K): 

            top = weights[j] * getMultiDimNormForEM(means[j], covariances[:,:,j], X[n], D) # numerator of E-Step function
            bottom = 0 # reset sum in denomimator for each gaussian
            
            # sum in denominator
            for k in range(K): 
                bottom += weights[k] * getMultiDimNormForEM(means[k], covariances[:,:,k], X[n], D) # denominator of E-Step function
            gamma[n][j] = top / bottom #final gamma for each combination of datapoint and gaussian
    return [getLogLikelihood(means, weights, covariances, X), gamma]

def getMultiDimNormForEM(mean, covariance, x, D): #same as getMultiDimNorm() in getLogLikelihood.py

    divider = 1 / (np.power(2*np.pi, D/2) * np.power(np.linalg.det(covariance),1/2))
    vector = np.asmatrix(x - mean)
    e = (-(1/2) * vector.dot(np.linalg.inv(covariance)).dot(np.transpose(vector))).item()
    multiDimNorm = divider * np.exp(e)
    return multiDimNorm