import numpy as np
from itertools import permutations

def pea(y, m, t):
    ly = len(y)
    
    
    permlist = list(permutations(range(m)))
    c = np.zeros(len(permlist))
    print(permlist)

    # ! Be advise the range
    for j in range(ly - t * (m - 1) - t + 1):
        sorted_idx = list(np.argsort(y[j : j + t * (m - 1) + t : t]))
        for jj, perm in enumerate(permlist):
            if sorted_idx == list(perm):
                c[jj] += 1
    
    print("The numbers:")
    print(c)
    if np.sum(c) == 0:
        raise ValueError("The sum of permutation counts is zero!")

    p = c / np.sum(c)
    print("The prop:")
    print(p)
    
    pe = np.zeros_like(c)
    for i, count in enumerate(c):
        if count != 0:
            pe[i] = -p[i] * np.log2(p[i])

    return pe

# ? Why to use sort function instead of the directly calculation?
y = [4, 7, 9, 10, 6, 11, 3]
m = 3
t = 1
print(pea(y, m, t))

