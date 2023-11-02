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


    #init key Variables
    N = X.shape[0]
    D = X.shape[1]
    K = len(means)
    logLikelihood = 0

    # outer sum of logliklihood calculations
    for n in range(N):  
        secSum  = 0
        
        #inner sum of logliklihood calculations 
        for k in range(K): 
            secSum += weights[k] * getMultiDimNorm(means[k], covariances[:,:,k], X[n], D)
        logLikelihood += np.log(secSum)
    return logLikelihood

def getMultiDimNorm(mean, covariance, x, D):

    divider = 1 / (np.power(2*np.pi, D/2) * np.power(np.linalg.det(covariance),1/2)) #first part of the Normaldistibution 
    vector = np.asmatrix(x - mean) #init vector to be able to transpose later
    e = (-(1/2) * vector.dot(np.linalg.inv(covariance)).dot(np.transpose(vector))).item() #inside of exp()
    multiDimNorm = divider * np.exp(e) # final Normaldistribution
    return multiDimNorm