import numpy as np
from PECalc import pea


def mapminmax(arr, mi, mx):
    return (arr - np.min(arr)) * (mx - mi) / (np.max(arr) - np.min(arr)) + mi


def getentropy(input, granule_size):
    x, y = input.shape
    # ! divide the long trail into sub trails, trails number are g
    granule_num = y // granule_size

    # ! can find the maximum and minimum directly
    mi = np.min(input)
    mx = np.max(input)

    entropy = np.zeros((x, granule_num, 6))
    min_max_entropy_feature_C1 = np.zeros((x, granule_num, 2))

    for i in range(x):
        x1 = input[i, :]
        for k in range(granule_num):
            print(x1[(k * granule_size):((k + 1) * granule_size)])
            entropy[i, k, :] = pea(x1[(k * granule_size):((k + 1) * granule_size)], 3, 1)

    entropy = np.reshape(mapminmax(np.reshape(entropy, -1), mi, mx), (x, granule_num, 6))

    for i in range(x):
        for j in range(granule_num):
            current_granule = entropy[i, j, :]
            min_max_entropy_feature_C1[i, j, 0] = np.min(current_granule)
            min_max_entropy_feature_C1[i, j, 1] = np.max(current_granule)

    return min_max_entropy_feature_C1