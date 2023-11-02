import numpy as np
from EStep import EStep
from MStep import MStep
from regularize_cov import regularize_cov
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def estGaussMixEM(data, K, n_iters, epsilon):
    # EM algorithm for estimation gaussian mixture mode
    #
    # INPUT:
    # data           : input data, N observations, D dimensional
    # K              : number of mixture components (modes)
    #
    # OUTPUT:
    # weights        : mixture weights - P(j) from lecture
    # means          : means of gaussians
    # covariances    : covariancesariance matrices of gaussians

    #####Insert your code here for subtask 6e#####

    # init key variables
    N = data.shape[0]
    D = data.shape[1]
    covariances = np.zeros((D, D, K))

    # code stolen from exercise sheet
    weights = np.ones(K) / K
    kmeans = KMeans(n_clusters = K, n_init = 10).fit(data)
    cluster_idx = kmeans.labels_
    means = kmeans.cluster_centers_
    # Create initial covariance matrices
    for j in range(K):
        data_cluster = data[cluster_idx == j]
        min_dist = np.inf
        for i in range(K):
            # compute sum of distances in cluster
            dist = np.mean(euclidean_distances(data_cluster, [means[i]], squared=True))
            if dist < min_dist:
                min_dist = dist
        covariances[:, :, j] = np.eye(D) * min_dist
    
    # do n_inters iterationen of the EM Algo
    for n in range(n_iters):

        # for each mixture component regularize covariances
        for k in range(K):
            covariances[:,:,k] = regularize_cov(covariances[:,:,k],epsilon)

        l, gamma = EStep(means, covariances, weights, data) # E-Step
        weights, means, covariances, l = MStep(gamma, data) # M-Step
    
    return [weights, means, covariances]
