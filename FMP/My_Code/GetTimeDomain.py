import scipy.io
import numpy as np

from FMP.My_Code.ReadTool import get_feature

mat_data = scipy.io.loadmat('Data/BCI_data.mat')

print(mat_data)

train_data_C1, test_data_C1, train_data_C2, test_data_C2 = get_feature(mat_data)

features_results = []

for i in range(train_data_C1.shape[0]):
    slide_2d_trainDataC1 = train_data_C1[i , :, :]
    for i in range(slide_2d_trainDataC1.shape[0]):
        sequence_1d_trainDataC1 = slide_2d_trainDataC1[i, :]

        mean = np.mean(sequence_1d_trainDataC1)
        variance = np.var(sequence_1d_trainDataC1)

        features = {
            'mean': mean,
            'variance': variance
        }

        features_results.append(features)

