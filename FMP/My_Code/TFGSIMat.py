import itertools
from scipy.io import savemat
import numpy as np
import time
import scipy
from GranularToGsi import gr2gsi
from GetTimeFeatures import getTimeFea
from DatasetSplit import DatasetSplit
import h5py

if __name__ == '__main__':
# divide the original data into train set and test set
    C1_train, C1_test, C2_train, C2_test = DatasetSplit()  #
    C1_train = np.mean(C1_train, 1)
    C2_train = np.mean(C2_train, 1)

    # ! get corresponding features
    time_feature_C1 = getTimeFea(C1_train, 50)  # 84, 100, 2
    time_feature_C2 = getTimeFea(C2_train, 50)  # 84, 100, 2
    min_C1 = time_feature_C1[:, :, 0]  # 存储着 C1 中每个窗口的最小值
    max_C1 = time_feature_C1[:, :, 1]  # 存储着 C1 中每个窗口的最大值

    min_C2 = time_feature_C2[:, :, 0]  # 存储着 C1 中每个窗口的最小值
    max_C2 = time_feature_C2[:, :, 1]  # 存储着 C1 中每个窗口的最大值

    t1 = np.array([min_C1[0, 0], max_C1[0, 0]])
    t2 = np.array([min_C2[0, 0], max_C2[0, 0]])

    mat_data = {'time_granule_result': np.zeros((42 * 42, 100, 2, 4))}

    for i, j in itertools.product(range(42), range(42)):
        index = i * 42 + j
        for k in range(100):
            t1 = np.array([min_C1[i, k], max_C1[i, k]])
            t2 = np.array([min_C2[j, k], max_C2[j, k]])
            T1, T2 = np.meshgrid(t1, t2)  # cartesian product
            time_granule_result = np.transpose(np.column_stack((T1.ravel(), T2.ravel())))
            mat_data['time_granule_result'][index, k, :, :] = time_granule_result

    mat_data = mat_data['time_granule_result']  # 12 * 12, 100, 2, 4

    time_feature_4_200 = np.zeros((1764, 4, 200))
    for i, j in itertools.product(range(1764), range(100)):
        # reshape from 2x4 to 4x2
        window = mat_data[i, j].reshape(2, 4).T.flatten()
        # assign to the correct location in the result matrix
        time_feature_4_200[i, :, j * 2:(j + 1) * 2] = window.reshape(4, 2)
    print(time_feature_4_200[0, :, :].shape)

    print("The GSI information for (C1, C2) will be generated!")
    time_feature_GSI = np.zeros((1764, 1, 224, 224))

    for i in range(1764):
        time_feature_GSI[i, 0, :, :] = gr2gsi(time_feature_4_200[i, :, :], 224)
        print(f"The No.{i} GSI plot has been generated！")
        print("Current System time:", time.ctime())

    scipy.io.savemat('/Users/lianyiyu/Documents/Python/FMP/OutputResult_Plot/TF_GSI_C1andC2_1764_50.mat', {'TF_GSI': time_feature_GSI})
