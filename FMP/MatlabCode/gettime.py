# ! w是 window size，

import numpy as np

def gettime(x, w):
    t, l = x.shape # 获取时序信号的形状
    g = l // w # 记录段数
    y = [] # 存储输出矩阵
    
    for i in range(t-1):
        x1 = x[i, :]
        for j in range(i+1, t):
            x2 = x[j, :]
            for k in range(g):
                t1 = [np.min(x1[k*w:(k+1)*w]), np.max(x1[k*w:(k+1)*w])]
                t2 = [np.min(x2[k*w:(k+1)*w]), np.max(x2[k*w:(k+1)*w])]
                
                T1, T2 = np.meshgrid(t1, t2)
                y.append(np.column_stack((T1.ravel(), T2.ravel())))
    
    return np.array(y)