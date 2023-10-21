import numpy as np
from OverR import OverR

def gr2gsi(g1, x):
    g = g1.shape[1]
    h = 1.0 / x
    gd = np.arange(0, 1 + h, h)

    y = np.zeros((x, x))

    # ! two layers loop traverse each grids
    # ! pay attention to the [i:i+2] and [j:j+2]
    # ! the slice range of Python and Matlab are different
    # ! A = [1, 2, 3, 4] 
    # ! For Python: A[0:3]=[1, 2, 3]
    # ! For Matlab: A[1:3]=[1, 2, 3]
    # ! So if I want to get two elements, plus two is necessary
    for i in range(x):
        gx = gd[i:i+2]
        for j in range(x):
            gy = gd[j:j+2]
            X, Y = np.meshgrid(gx, gy)
            
            # ! Flatten the two 2D matrices X and Y into one-dimensional arrays 
            # ! and stack them vertically to form a 2D matrix, 
            # ! then transpose them so that each row contains a pair of (x, y) 
            # ! coordinates.
            gc = np.vstack((X.ravel(), Y.ravel())).T

            # ! g/2 represents the number of granule
            r = np.array([OverR(gc, g1[:, 2*k:2*(k+1)]) for k in range(int(g/2))])

            y[i, j] = np.sum(r)

    y = np.fliplr(y)
    y = y.T

    return y

g1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
x = 3
print(gr2gsi(g1, x))
