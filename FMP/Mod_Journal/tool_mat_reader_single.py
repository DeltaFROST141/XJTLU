import os

import numpy as np
import scipy
import torch as t

"""
Loads image feature data from a .mat file and splits it into training and test sets.

Loads the .mat file using scipy.io.loadmat. If that fails, tries hdf5storage.loadmat. 
Extracts the '_gsi1' or '_gsi2' key based on the filename. Splits the data into 
90% training and 10% test sets. Converts the data into numpy arrays.

Args:
   dp: The file path to the .mat file.

Returns:
   _tra: The training feature data as a numpy array.
   _tes: The test feature data as a numpy array.
   
Raises:
   ValueError: If the filename does not contain '_1' or '_2'.

"""
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

# 根据代码,有以下几点分析:

# get_feature和get_label这两个函数分别用于加载图像特征和标签数据。

# 它们都从.mat文件中加载数据,然后将数据划分为训练集和测试集。

# 之所以要分别划分训练集和测试集,主要有以下原因:

# 特征数据和标签数据可能来源不同,数量也可能不完全一致,需要分别处理。

# 将它们分开处理,可以更灵活地设置训练集和测试集的比例,对二者进行不同的预处理。

# 分开处理特征和标签,符合单一职责原则,增加代码的可维护性。

# 在机器学习流程中,训练和测试通常需要分别准备特征数据和标签数据。分开划分可以方便模型的训练和测试。

# 所以为了处理方便、增加灵活性以及提高可维护性,选择对特征数据和标签数据分别进行划分是合理的。
# 总之,分别对特征和标签划分训练集和测试集,可以让数据加载和预处理更加模块化和可控,这在机器学习项目中是常见和推荐的做法。


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
dp_fea_1 = '../data/gsi_g_224_1.mat'
dp_fea_2 = '../data/gsi_g_224_2.mat'
dp_lab_1 = '../data/label_b_1.mat'
dp_lab_2 = '../data/label_b_2.mat'
# Get the feature ---------------------------------------------------
tra_1, tes_1 = get_feature(dp_fea_1)
tra_2, tes_2 = get_feature(dp_fea_2)
# Get the label -----------------------------------------------------
tra_lab_1, tea_lab_1 = get_label(dp_lab_1)
tra_lab_2, tes_lab_2 = get_label(dp_lab_2)
# Concatenate the data ----------------------------------------------
tra_fea = np.concatenate((tra_1, tra_2), axis=0)
tra_lab = np.concatenate((tra_lab_1, tra_lab_2), axis=0)
tes_fea = np.concatenate((tes_1, tes_2), axis=0)
tes_lab = np.concatenate((tea_lab_1, tes_lab_2), axis=0)

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
