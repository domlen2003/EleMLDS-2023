import numpy as np
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getMultiDimNorm


def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    #####Insert your code here for subtask 1g#####
    print("skindata")
    s_weights, s_means, s_covariances = estGaussMixEM(sdata, K, n_iter, epsilon)
    print("nonskindata")
    n_weights, n_means, n_covariances = estGaussMixEM(ndata, K, n_iter, epsilon)

    print(n_weights.shape, n_means.shape, n_covariances.shape, img.shape)
    result = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    D = img.shape[2]
    for col in range(len(img)):
        row_len = len(img[col])
        for row in range(row_len):
            for rgb in range(3):
                skin_prob = 0
                not_skin_prob = 0
                for k in range(K):
                    skin_prob += s_weights[k] * getMultiDimNorm(s_means[k], s_covariances[:, :, k], img[col][row], D)
                    not_skin_prob += n_weights[k] * getMultiDimNorm(n_means[k], n_covariances[:, :, k], img[col][row], D)
                    if not_skin_prob == 0:
                        print(col, row, rgb, k)
                is_skin = bool(skin_prob/not_skin_prob > theta)
                result[col, row, rgb] = int(is_skin)

    print(result)

    return result
