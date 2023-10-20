import numpy as np

# ! The enhanced edition about getting the time domain
# ! ONLY use two layers loops

def gettime(x, w):
    t, l = x.shape
    g = l // w
    n = 0
    y = np.zeros((t * (t - 1) // 2, 4, 2 * g))

    # Get all the combinations at first, by this operation can reduce the complexity ?
    combinations = [(i, j) for i in range(t) for j in range(i + 1, t)]

    for (i, j) in combinations:
        x1 = x[i, :]
        x2 = x[j, :]
        for k in range(g):
            t1 = [x1[k * w:(k + 1) * w].min(), x1[k * w:(k + 1) * w].max()]
            t2 = [x2[k * w:(k + 1) * w].min(), x2[k * w:(k + 1) * w].max()]
            T1, T2 = np.meshgrid(t1, t2)
            combined = np.column_stack((T1.ravel(), T2.ravel()))
            y[n, :, 2 * k:2 * (k + 1)] = combined
        n += 1

    return y


# Testing
x = np.random.rand(4, 4)
w = 2
result = gettime(x, w)
print(result)
