import itertools
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
from scipy.io import loadmat
from GranularToGsi import gr2gsi
from GetGradientFeatures import getGraFea
from GetGradientChannel import getgradient
from DatasetSplit import DatasetSplit

# 原始数据划分，两类分为测试和训练
data_Path = 'BCI_data.mat'
data = loadmat(data_Path)
data_C1 = data['C1']
data_C2 = data['C2']
np.random.seed(0)
indices = np.random.permutation(105)
C1_train = np.mean(data_C1[indices[:84], :], 1)
C2_train = np.mean(data_C2[indices[:84], :], 1)

gradient_feature_C1 = getgradient(C1_train, 50)  # 84, 100, 2 granule number is 100
gradient_feature_C2 = getgradient(C2_train, 50)  # 84, 100, 2
min_C1 = gradient_feature_C1[:, :, 0]  # 存储着 C1 中每个窗口的最小值
max_C1 = gradient_feature_C1[:, :, 1]
min_C2 = gradient_feature_C2[:, :, 0]  # 存储着 C1 中每个窗口的最小值
max_C2 = gradient_feature_C2[:, :, 1]

# t1 = np.array([min_C1[0, 0], max_C1[0, 0]])
# t2 = np.array([min_C2[0, 0], max_C2[0, 0]])

mat_data = {'gradient_granule_result': np.zeros((1 * 1, 100, 2, 4))}

for i, j in itertools.product(range(1), range(1)):
    index = i * 1 + j
    for k in range(100):
        t1 = np.array([min_C1[i, k], max_C1[i, k]])
        t2 = np.array([min_C2[j, k], max_C2[j, k]])
        T1, T2 = np.meshgrid(t1, t2)  # cartesian product
        gradient_granule_result = np.transpose(np.column_stack((T1.ravel(), T2.ravel())))
        mat_data['gradient_granule_result'][index, k, :, :] = gradient_granule_result

mat_data = mat_data['gradient_granule_result']  # 4,100,2,4

gradient_feature_4_200 = np.zeros((1, 4, 200))
for i, j in itertools.product(range(1), range(100)):
    # reshape from 2x4 to 4x2
    window = mat_data[i, j].reshape(2, 4).T.flatten()
    # assign to the correct location in the result matrix
    gradient_feature_4_200[i, :, j * 2:(j + 1) * 2] = window.reshape(4, 2)
print(gradient_feature_4_200[0, :, :].shape)

print("The GSI information for (C1, C2) will be generated!")
gradient_feature_GSI = np.zeros((1, 1, 224, 224))

for i in range(1):
    gradient_feature_GSI[i, 0, :, :] = gr2gsi(gradient_feature_4_200[i, :, :], 224)
    print(f"The No.{i} GSI plot has been generated!")
    print("Current System time:", time.ctime())

scipy.io.savemat('/GF_GSI_C1andC2_4.mat', {'GF_GSI': gradient_feature_GSI})