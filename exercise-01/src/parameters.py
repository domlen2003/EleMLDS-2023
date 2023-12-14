def parameters():
    epsilon = 0.0001 # regularization
    K = 5  # number of desired clusters
    n_iter = 5  # number of iterations
    skin_epsilon = 10
    skin_n_iter = 10
    skin_K = 5
    theta = 1  # threshold for skin detection

    return epsilon, K, n_iter, skin_n_iter, skin_epsilon, skin_K, theta
