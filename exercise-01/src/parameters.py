def parameters():
    epsilon = 0.0001 # regularization
    K = 5  # number of desired clusters
    n_iter = 5  # number of iterations
    skin_epsilon = 0.0001
    skin_n_iter = 1
    skin_K = 1
    theta = 100  # threshold for skin detection

    return epsilon, K, n_iter, skin_n_iter, skin_epsilon, skin_K, theta
