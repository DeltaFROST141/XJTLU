from scipy.io import loadmat
import numpy as np
from GetTimeChannel import gettime
from GetGradientChannel import getgradient
# from GetEntropyChannel import getentropy
from PECalc import pea

data = loadmat('/Users/lianyiyu/Documents/文稿 - 涟漪羽的MacBook Air/Code/FMP/My_Code/Data/BCI_data.mat')

C1_data = data['C1']
C2_data = data['C2']

# 任取三条进行两两比较计算，一共有三种情况
num_samples = 3
C1_samples = C1_data[np.random.choice(C1_data.shape[0], num_samples, replace=False)]
C2_samples = C2_data[np.random.choice(C2_data.shape[0], num_samples, replace=False)]

# 如何对三维的原始数据进行降维？ 因为输入数据需要是二维的，行是条数、列是不同的时间
reshaped_data = C1_samples.reshape(3, 118*5000)

gettime(reshaped_data, 3)
getgradient(reshaped_data, 3)
