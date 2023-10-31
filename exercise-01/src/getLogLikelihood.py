import numpy as np
def getLogLikelihood(means, weights, covariances, X):
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
    N = X.shape[0]
    D = X.shape[1]
    K = len(means)
    logLikelihood = 0

    for n in range(N):
        secSum  = 0
        for k in range(K):
            secSum += weights[k] * getMultiDimNorm(means[k], covariances[:,:,k], X[n], D)
        logLikelihood += np.log(secSum)
    return logLikelihood

def getMultiDimNorm(mean, covariance, x, D):

    divider = 1 / (np.power(2*np.pi, D/2) * np.power(np.linalg.det(covariance),1/2))
    vector = np.asmatrix(x - mean)
    e = (-(1/2) * vector.dot(np.linalg.inv(covariance)).dot(np.transpose(vector))).item()
    multiDimNorm = divider * np.exp(e)
    return multiDimNorm