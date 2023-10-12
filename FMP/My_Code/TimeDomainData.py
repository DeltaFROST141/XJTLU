import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from FMP.MatlabCode.gettime import gettime
from FMP.My_Code.ReadTool import get_feature

filepath = 'Data/BCI_data.mat'
after_processing = get_feature(filepath)
train_data_C1, test_data_C1, train_data_C2, test_data_C2  = after_processing
print(train_data_C1.shape)

features_results = []

for i in range(train_data_C1.shape[0]):
    slide_2d_trainDataC1 = train_data_C1[i , :, :]
    for i in range(slide_2d_trainDataC1.shape[0]):
        sequence_1d_trainDataC1 = slide_2d_trainDataC1[i, :]

        mean = np.mean(sequence_1d_trainDataC1)
        variance = np.var(sequence_1d_trainDataC1)
        max_value = np.max(sequence_1d_trainDataC1)
        peaks = find_peaks(sequence_1d_trainDataC1)

        rms = np.sqrt(np.mean(np.square(sequence_1d_trainDataC1)))

        features = {
            'mean': mean,
            'variance': variance,
            'max': max_value,
            'peaks': peaks
        }

        features_results.append(features)

print(features_results)

# 此时数据是已经划分完测试集和训练集的三维数据
# pca = PCA(n_components=5000)
# train_data_C1_2d = pca.fit_transform(train_data_C1.reshape(94 * 118, 5000))
# gettime(train_data_C1_2d, 3)

# gettime(train_data_C1, 3)
# my_array = np.array(after_processing, dtype=object)
# print(type(my_array))
# print(my_array)


# gettime(after_processing, 3)
