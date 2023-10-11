import os

import numpy as np
import scipy
import torch as t


def get_feature(dp):
    # Read mat file
    try:
        _mat = scipy.io.loadmat(dp)
    except NotImplementedError:
        import hdf5storage
        _mat = hdf5storage.loadmat(dp)
    # print the keys
    print(_mat.keys())
    # Extract the data
    if '_1' in dp:
        _mat = _mat['gsi1']
    elif '_2' in dp:
        _mat = _mat['gsi2']
    else:
        raise ValueError('Wrong data path, key not found.')
    """"
    # The mat file is a 4 dimensional array
    # The first dimension is the number of samples
    # The second dimension is the number of channels
    # The third and fourth dimensions are the height and width of the image
    """
    # Split the data based on the first dimension
    # The first 90% of the data is used for training
    # The last 10% of the data is used for testing
    _tra = _mat[:int(_mat.shape[0] * 0.9), :, :, :]
    _tes = _mat[int(_mat.shape[0] * 0.9):, :, :, :]
    # Convert the data to a numpy array
    _tra = np.array(_tra)
    _tes = np.array(_tes)
    # Print the shape of the data
    print('The shape of the training feature is: {}'.format(_tra.shape))
    print('The shape of the testing feature is: {}'.format(_tes.shape))
    return _tra, _tes


def get_label(dp):
    # Read mat file
    try:
        _mat = scipy.io.loadmat(dp)
    except NotImplementedError:
        import hdf5storage
        _mat = hdf5storage.loadmat(dp)
    # print the keys
    print(_mat.keys())
    # Extract the data
    if '_1' in dp:
        try:
            _mat = _mat['label1']
        except KeyError:
            _mat = _mat['label_1']
    elif '_2' in dp:
        try:
            _mat = _mat['label2']
        except KeyError:
            _mat = _mat['label_2']
    else:
        raise ValueError('Wrong data path, key not found.')
    # print the data shape
    print(_mat.shape)
    """"
    # The mat file is a 2 dimensional array
    # The first dimension is the number of samples
    # The second dimension is the number of classes
    """
    # Split the data based on the first dimension
    # The first 90% of the data is used for training
    # The last 10% of the data is used for testing
    _tra_1 = _mat[:int(_mat.shape[0] * 0.9), :]
    _tes_1 = _mat[int(_mat.shape[0] * 0.9):, :]
    # Convert the data to a numpy array
    _tra = np.array(_tra_1)
    _tes = np.array(_tes_1)
    # Print the shape of the data
    print('The shape of the training label is: {}'.format(_tra.shape))
    print('The shape of the testing label is: {}'.format(_tes.shape))
    return _tra, _tes


class EEGData(t.utils.data.Dataset):
    """Dataset wrapping tensors."""

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


os.system('clear')

# Subject one -------------------------------------------------------
dp_fea_1 = '../data/sample_gsi_b_224_1.mat'
dp_fea_2 = '../data/sample_gsi_b_224_2.mat'
dp_lab_1 = '../data/sample_label_1.mat'
dp_lab_2 = '../data/sample_label_2.mat'
# Get the feature ---------------------------------------------------
tra_1, tes_1 = get_feature(dp_fea_1)
tra_2, tes_2 = get_feature(dp_fea_2)
# Get the label -----------------------------------------------------
tra_lab_1, tea_lab_1 = get_label(dp_lab_1)
tra_lab_2, tes_lab_2 = get_label(dp_lab_2)
# Concatenate the data ----------------------------------------------
tra_fea_b = np.concatenate((tra_1, tra_2), axis=0)
tra_lab_b = np.concatenate((tra_lab_1, tra_lab_2), axis=0)
tes_fea_b = np.concatenate((tes_1, tes_2), axis=0)
tea_lab_b = np.concatenate((tea_lab_1, tes_lab_2), axis=0)

# Subject two -------------------------------------------------------
dp_fea_1 = '../data/sample_gsi_c_224_1.mat'
dp_fea_2 = '../data/sample_gsi_c_224_2.mat'
dp_lab_1 = '../data/sample_label_1.mat'
dp_lab_2 = '../data/sample_label_2.mat'
# Get the feature ---------------------------------------------------
tra_1, tes_1 = get_feature(dp_fea_1)
tra_2, tes_2 = get_feature(dp_fea_2)
# Get the label -----------------------------------------------------
tra_lab_1, tea_lab_1 = get_label(dp_lab_1)
tra_lab_2, tes_lab_2 = get_label(dp_lab_2)
# Concatenate the data ----------------------------------------------
tra_fea_c = np.concatenate((tra_1, tra_2), axis=0)
tra_lab_c = np.concatenate((tra_lab_1, tra_lab_2), axis=0)
tes_fea_c = np.concatenate((tes_1, tes_2), axis=0)
tes_lab_c = np.concatenate((tea_lab_1, tes_lab_2), axis=0)
# Concatenate the data ----------------------------------------------
tra_fea = np.concatenate((tra_fea_b, tra_fea_c), axis=0)
tra_lab = np.concatenate((tra_lab_b, tra_lab_c), axis=0)
tes_fea = np.concatenate((tes_fea_b, tes_fea_c), axis=0)
tes_lab = np.concatenate((tea_lab_b, tes_lab_c), axis=0)
del tra_fea_b, tra_lab_b, tes_fea_b, tea_lab_b
del tra_fea_c, tra_lab_c, tes_fea_c, tes_lab_c

# Subject three -----------------------------------------------------
dp_fea_1 = '../data/sample_gsi_d_224_1.mat'
dp_fea_2 = '../data/sample_gsi_d_224_2.mat'
dp_lab_1 = '../data/sample_label_1.mat'
dp_lab_2 = '../data/sample_label_2.mat'
# Get the feature ---------------------------------------------------
tra_1, tes_1 = get_feature(dp_fea_1)
tra_2, tes_2 = get_feature(dp_fea_2)
# Get the label -----------------------------------------------------
tra_lab_1, tea_lab_1 = get_label(dp_lab_1)
tra_lab_2, tes_lab_2 = get_label(dp_lab_2)
# Concatenate the data ----------------------------------------------
tra_fea_d = np.concatenate((tra_1, tra_2), axis=0)
lab_tra_d = np.concatenate((tra_lab_1, tra_lab_2), axis=0)
tes_fea_d = np.concatenate((tes_1, tes_2), axis=0)
lab_tes_d = np.concatenate((tea_lab_1, tes_lab_2), axis=0)
# Concatenate the data ----------------------------------------------
tra_fea = np.concatenate((tra_fea, tra_fea_d), axis=0)
tra_lab = np.concatenate((tra_lab, lab_tra_d), axis=0)
tes_fea = np.concatenate((tes_fea, tes_fea_d), axis=0)
tes_lab = np.concatenate((tes_lab, lab_tes_d), axis=0)
del tra_fea_d, lab_tra_d, tes_fea_d, lab_tes_d

# Subject four ------------------------------------------------------
dp_fea_1 = '../data/sample_gsi_e_224_1.mat'
dp_fea_2 = '../data/sample_gsi_e_224_2.mat'
dp_lab_1 = '../data/sample_label_1.mat'
dp_lab_2 = '../data/sample_label_2.mat'
# Get the feature ---------------------------------------------------
tra_1, tes_1 = get_feature(dp_fea_1)
tra_2, tes_2 = get_feature(dp_fea_2)
# Get the label -----------------------------------------------------
tra_lab_1, tea_lab_1 = get_label(dp_lab_1)
tra_lab_2, tes_lab_2 = get_label(dp_lab_2)
# Concatenate the data ----------------------------------------------
tra_fea_e = np.concatenate((tra_1, tra_2), axis=0)
lab_tra_e = np.concatenate((tra_lab_1, tra_lab_2), axis=0)
tes_fea_e = np.concatenate((tes_1, tes_2), axis=0)
lab_tes_e = np.concatenate((tea_lab_1, tes_lab_2), axis=0)
# Concatenate the data ----------------------------------------------
tra_fea = np.concatenate((tra_fea, tra_fea_e), axis=0)
tra_lab = np.concatenate((tra_lab, lab_tra_e), axis=0)
tes_fea = np.concatenate((tes_fea, tes_fea_e), axis=0)
tes_lab = np.concatenate((tes_lab, lab_tes_e), axis=0)
del tra_fea_e, lab_tra_e, tes_fea_e, lab_tes_e

# Subject five ------------------------------------------------------
dp_fea_1 = '../data/sample_gsi_g_224_1.mat'
dp_fea_2 = '../data/sample_gsi_g_224_2.mat'
dp_lab_1 = '../data/sample_label_1.mat'
dp_lab_2 = '../data/sample_label_2.mat'
# Get the feature ---------------------------------------------------
tra_1, tes_1 = get_feature(dp_fea_1)
tra_2, tes_2 = get_feature(dp_fea_2)
# Get the label -----------------------------------------------------
tra_lab_1, tea_lab_1 = get_label(dp_lab_1)
tra_lab_2, tes_lab_2 = get_label(dp_lab_2)
# Concatenate the data ----------------------------------------------
tra_fea_g = np.concatenate((tra_1, tra_2), axis=0)
lab_tra_g = np.concatenate((tra_lab_1, tra_lab_2), axis=0)
tes_fea_g = np.concatenate((tes_1, tes_2), axis=0)
lab_tes_g = np.concatenate((tea_lab_1, tes_lab_2), axis=0)
# Concatenate the data ----------------------------------------------
tra_fea = np.concatenate((tra_fea, tra_fea_g), axis=0)
tra_lab = np.concatenate((tra_lab, lab_tra_g), axis=0)
tes_fea = np.concatenate((tes_fea, tes_fea_g), axis=0)
tes_lab = np.concatenate((tes_lab, lab_tes_g), axis=0)
del tra_fea_g, lab_tra_g, tes_fea_g, lab_tes_g

# Print the shape of the data ---------------------------------------
print(' ---------- Final concatenation ---------- ')
print('The shape of the training feature is: {}'.format(tra_fea.shape))
print('The shape of the training label is: {}'.format(tra_lab.shape))
print('The shape of the testing feature is: {}'.format(tes_fea.shape))
print('The shape of the testing label is: {}'.format(tes_lab.shape))

# Convert the data to a tensor -------------------------------------
tra_fea = t.from_numpy(tra_fea).float()
tra_lab = t.from_numpy(tra_lab).float()
tes_fea = t.from_numpy(tes_fea).float()
tes_lab = t.from_numpy(tes_lab).float()

Tra = EEGData(tra_fea, tra_lab)
Tes = EEGData(tes_fea, tes_lab)

tra = t.utils.data.DataLoader(
    Tra,
    batch_size=64,
    shuffle=True,
)
tes = t.utils.data.DataLoader(
    Tes,
    batch_size=tes_fea.shape[0],
    shuffle=True,
)

del tra_fea, tra_lab, tes_fea, tes_lab, Tra, Tes
del tra_1, tra_2, tes_1, tes_2
del tra_lab_1, tra_lab_2, tea_lab_1, tes_lab_2