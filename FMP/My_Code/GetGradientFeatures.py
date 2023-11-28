import numpy as np


# ! restrain from 0 to 1
def mapminmax(arr, mi, mx):
    return (arr - np.min(arr)) * (mx - mi) / (np.max(arr) - np.min(arr)) + mi


def getgradient(input, granule_size):
    x, y = input.shape
    granule_num = y // granule_size

    # ! can find the maximum and minimum directly
    mi = np.min(input)
    mx = np.max(input)

    # ! store the gradient values
    grad = np.zeros((x, granule_num, granule_size))
    min_max_granule = np.zeros((x, granule_num, 2))

    # ! traverse through each row and calculate
    # ! each the gradients of each window
    # ! the shape of grad is (t, g, w)
    # ! for each time series and each window have a length `w` vector
    for i in range(x):
        x1 = input[i, :]
        for k in range(granule_num):
            grad[i, k, :] = np.gradient(x1[(k * granule_size):((k + 1) * granule_size)])

    # ! after the two times reshape function, the shape is also
    # ! (t, g, w)
    grad = np.reshape(mapminmax(np.reshape(grad, -1), mi, mx), (x, granule_num, granule_size))

    for i in range(x):
        for k in range(granule_num):
            granule = grad[i, k, :]
            min_max_granule[i, k, 0] = np.min(granule)  # Min value in the third dimension's first position
            min_max_granule[i, k, 1] = np.max(granule)

    return min_max_granule