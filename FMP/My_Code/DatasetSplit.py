from scipy.io import loadmat, savemat
import numpy as np


def DatasetSplit():
    path_to_data = '/Users/lianyiyu/Documents/Python/FMP/Data/BCI_data.mat'

    data = loadmat(path_to_data)

    data_C1 = data['C1']
    data_C2 = data['C2']

    # assert data_C1.shape == (105, 5000)
    # assert data_C2.shape == (105, 5000)

    np.random.seed(0)
    indices = np.random.permutation(105)

    C1_train = data_C1[indices[:84], :]
    C1_test = data_C1[indices[84:], :]

    C2_train = data_C2[indices[:84], :]
    C2_test = data_C2[indices[84:], :]

    return C1_train, C1_test, C2_train, C2_test
