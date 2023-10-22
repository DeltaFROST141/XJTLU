import numpy as np
import itertools

    # ! restrain from 0 to 1
def mapminmax(arr, mi, mx):
    return (arr - np.min(arr)) * (mx - mi) / (np.max(arr) - np.min(arr)) + mi

def getgradient(x, w):
    t, l = x.shape
    g = l // w

    # ! can find the maximum and minimum directly
    mi = np.min(x)
    mx = np.max(x)

    # ! store the gradient values
    grad = np.zeros((t, g, w))

    # ! traverse through each rows and calculate
    # ! each the gradients of each windows
    # ! the shape of grad is (t, g, w)
    # ! for each time series and each windows have a length `w` vector
    for i in range(t):
        x1 = x[i, :]
        for k in range(g):
            grad[i, k, :] = np.gradient(x1[(k * w):((k + 1) * w)])

    # ! after the two times reshape function, the shape is also
    # ! (t, g, w)
    grad = np.reshape(mapminmax(np.reshape(grad, -1), mi, mx), (t, g, w))
    
    # ! use the combination to range the comparison between
    # ! different trials(avoid compare themself)
    comb_indices = list(itertools.combinations(range(t), 2))
    y = np.zeros((len(comb_indices), 2, 2 * g))

    # ! for each pair of (i, j), calculate the min and max
    # ! values.
    # ! index will show the position of specified pair 
    for idx, (i, j) in enumerate(comb_indices):
        min_grad1 = np.min(grad[i, :, :], axis=1)
        max_grad1 = np.max(grad[i, :, :], axis=1)
        min_grad2 = np.min(grad[j, :, :], axis=1)
        max_grad2 = np.max(grad[j, :, :], axis=1)

        for k in range(g):
            g1_min = min_grad1[k]
            g1_max = max_grad1[k]
            g2_min = min_grad2[k]
            g2_max = max_grad2[k]

            G1_min, G2_min = np.meshgrid(g1_min, g2_min)
            G1_max, G2_max = np.meshgrid(g1_max, g2_max)

            # ! use the matlab code idea
            # ! represent the overlapping structure
            y[idx, :, 2 * k] = np.column_stack((G1_min.ravel(), G2_min.ravel()))
            y[idx, :, 2 * k + 1] = np.column_stack((G1_max.ravel(), G2_max.ravel()))

            # y[idx, 0, 2 * k] = G1_min
            # y[idx, 1, 2 * k] = G2_min
            # y[idx, 0, 2 * k + 1] = G1_max
            # y[idx, 1, 2 * k + 1] = G2_max
    print(y)
    return y