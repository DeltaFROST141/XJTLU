import numpy as np
from itertools import permutations

# ! Only calculate the single trial entropy and store them(each index entropy) in a list 
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

    # ! The number list   
    p = c / np.sum(c)
    print("The prop:")
    print(p)
    
    # ! Also the list, but the probability list
    pe = np.zeros_like(c)
    for i, count in enumerate(c):
        if count != 0:
            # ! Entropy calculation
            pe[i] = -p[i] * np.log2(p[i])
    print("The entropy list:") 
    return pe

# ? Why to use sort function instead of the directly calculation?
y = [4, 7, 9, 10, 6, 11, 3]
m = 3
t = 1   
print(pea(y, m, t))