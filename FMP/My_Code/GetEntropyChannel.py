import numpy as np
import itertools
from PECalc import pea

def mapminmax(arr, mi, mx):
    return (arr - np.min(arr)) * (mx - mi) / (np.max(arr) - np.min(arr)) + mi 


def getentropy(x, w):
    
    t, l = x.shape
    # ! divide the long trail into sub trails, trails number are g
    g = l // w  

    # ! can find the maximum and minimum directly
    mi = np.min(x)
    mx = np.max(x)

    # ! store the entropy values
    # for example: (1, 3, 1)
    # means that find the second trails and the fourth windows
    # and the number of [(0, 2, 1)]
    entropy = np.zeros((t, g, 6))
    
    # ! loop over each window and calculate the entropy
    for i in range(t):
        x1 = x[i, :]
        for k in range(g):
            entropy[i, k, :] = pea(x1[(k * w):((k + 1) * w)] ,3 ,1)
            
    entropy = np.reshape(mapminmax(np.reshape(entropy, -1), mi, mx), (t, g, 6))
    
    comb_indices = list(itertools.combinations(range(t), 2))
    y = np.zeros((len(comb_indices), 2, 2 * g))

    # ! for each pair of (i, j), calculate the min and max
    # ! values.
    # ! index will show the position of specified pair 
    for idx, (i, j) in enumerate(comb_indices):
        min_entro1 = np.min(entropy[i, :, :], axis=1)
        max_entro1 = np.max(entropy[i, :, :], axis=1)
        min_entro2 = np.min(entropy[j, :, :], axis=1)
        max_entro2 = np.max(entropy[j, :, :], axis=1)

        for k in range(g):
            e1_min = min_entro1[k]
            e1_max = max_entro1[k]
            e2_min = min_entro2[k]
            e2_max = max_entro2[k]

            E1_min, E2_min = np.meshgrid(e1_min, e1_max)
            E1_max, E2_max = np.meshgrid(e2_min, e2_max)

            # ! use the matlab code idea
            # ! represent the overlapping structure
            y[idx, :, 2 * k] = np.column_stack((E1_min.ravel(), E1_max.ravel()))
            y[idx, :, 2 * k + 1] = np.column_stack((E2_min.ravel(), E2_max.ravel()))
    
    # print(y)
    return y