import os
import argparse
import pandas as pd
import numpy as np
import scipy
import torch as t
import gc

# Clear the cache --------------------------------------------------------
gc.enable()


def get_feature(dp):
    # Read mat file
    try:
        _mat = scipy.io.loadmat(dp)
    except NotImplementedError:
        import hdf5storage
        print("Using hdf5storage to load .mat file.")
        _mat = hdf5storage.loadmat(dp)
    _items = []
    for i in _mat.keys():
        _items.append(i)
    print("The keys of feature are: {}".format(_items))
    for i in _items:
        print("The feature shape of each class {} is: {}".format(
            i, _mat[i].shape))
    print("Concatenate the feature of each class.")
    # Randomly select 128 samples from each class
    _res = np.concatenate(
        (_mat[_items[0]][np.random.randint(
            0, _mat[_items[0]].shape[0],
            SELECTION)], _mat[_items[1]][np.random.randint(
                0, _mat[_items[1]].shape[0],
                SELECTION)], _mat[_items[2]][np.random.randint(
                    0, _mat[_items[2]].shape[0],
                    SELECTION)], _mat[_items[3]][np.random.randint(
                        0, _mat[_items[3]].shape[0], SELECTION)]),
        axis=0,
    )
    print("Concatenate the feature of each class successfully.")
    return _res


def get_label(dp):
    # Read mat file
    try:
        _mat = scipy.io.loadmat(dp)
    except NotImplementedError:
        import hdf5storage
        print("Using hdf5storage to load .mat file.")
        _mat = hdf5storage.loadmat(dp)
    _items = []
    for i in _mat.keys():
        _items.append(i)
    print("The keys of label .mat file are: {}".format(_items))
    for i in _items:
        print("The label shape of each class {} is: {}".format(
            i, _mat[_items[0]].shape))
    # Randomly select 128 samples from each class
    _res = np.concatenate(
        (_mat[_items[0]][np.random.randint(
            0, _mat[_items[0]].shape[0],
            SELECTION)], _mat[_items[1]][np.random.randint(
                0, _mat[_items[1]].shape[0],
                SELECTION)], _mat[_items[2]][np.random.randint(
                    0, _mat[_items[2]].shape[0],
                    SELECTION)], _mat[_items[3]][np.random.randint(
                        0, _mat[_items[3]].shape[0], SELECTION)]),
        axis=0,
    )
    print("Concatenate the label of each class successfully.")
    return _res


def get_tes_feature(dp):
    # Read mat file
    try:
        _mat = scipy.io.loadmat(dp)
    except NotImplementedError:
        import hdf5storage
        print("Using hdf5storage to load .mat file.")
        _mat = hdf5storage.loadmat(dp)
    _items = []
    for i in _mat.keys():
        _items.append(i)
    print("The keys of feature are: {}".format(_items))
    for i in _items:
        print("The feature shape of each class {} is: {}".format(
            i, _mat[i].shape))
    print("Concatenate the feature of each class.")
    # Randomly select 128 samples from each class
    _res = np.concatenate(
        (_mat[_items[0]], _mat[_items[1]], _mat[_items[2]], _mat[_items[3]]),
        axis=0,
    )
    print("Concatenate the feature of each class successfully.")
    return _res


def get_tes_label(dp):
    # Read mat file
    try:
        _mat = scipy.io.loadmat(dp)
    except NotImplementedError:
        import hdf5storage
        print("Using hdf5storage to load .mat file.")
        _mat = hdf5storage.loadmat(dp)
    _items = []
    for i in _mat.keys():
        _items.append(i)
    print("The keys of label .mat file are: {}".format(_items))
    for i in _items:
        print("The label shape of each class {} is: {}".format(
            i, _mat[_items[0]].shape))
    # Randomly select 128 samples from each class
    _res = np.concatenate(
        (_mat[_items[0]], _mat[_items[1]], _mat[_items[2]], _mat[_items[3]]),
        axis=0,
    )
    print("Concatenate the label of each class successfully.")
    return _res


# def concat_data(_d1, _d2, _d3, _d4):
#     # Concatenate data
#     _d1 = np.asarray(_d1)
#     _d2 = np.asarray(_d2)
#     _d3 = np.asarray(_d3)
#     _d4 = np.asarray(_d4)
#     # Check the shape of each data
#     if _d1.shape != _d2.shape or _d3.shape != _d4.shape or _d2.shape != _d4.shape:
#         raise ValueError("The shape of each data is not equal.")
#     return np.concatenate((_d1, _d2, _d3, _d4), axis=0)


# def concat_label(_l1, _l2, _l3, _l4):
#     # Concatenate label
#     _l1 = np.asarray(_l1) - np.ones(_l1.shape)
#     _l2 = np.asarray(_l2) - np.ones(_l2.shape)
#     _l3 = np.asarray(_l3) - np.ones(_l3.shape)
#     _l4 = np.asarray(_l4) - np.ones(_l4.shape)
#     # Check the shape of each label
#     if _l1.shape != _l2.shape or _l3.shape != _l4.shape or _l2.shape != _l4.shape:
#         raise ValueError("The shape of each label is not equal.")
#     return np.concatenate((_l1, _l2, _l3, _l4), axis=0)


def one_hot_encoding(_label):
    # One-hot encoding the label with 4 classes using pandas
    # Convert values to string
    _label = _label.astype(str)
    _label = pd.DataFrame(_label)
    _label = pd.get_dummies(_label)
    _label = np.asarray(_label)
    return _label


class EEGData(t.utils.data.Dataset):
    """Dataset wrapping tensors."""

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


os.system("cls")
# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dp_tra_fea",
    "-dptf",
    type=str,
    default="../data/train_1.mat",
)
parser.add_argument(
    "--dp_tra_lab",
    "-dptl",
    type=str,
    default="../data/label_tr_4.mat",
)
parser.add_argument(
    "--dp_tes_fea",
    "-dpsf",
    type=str,
    default="../data/test_1.mat",
)
parser.add_argument(
    "--dp_tes_lab",
    "-dpsl",
    type=str,
    default="../data/label_te_4.mat",
)
parser.add_argument(
    "--selection",
    "-s",
    type=int,
    default=1024,
)
args = parser.parse_args()
# Get the data
dp_tes_fea = args.dp_tes_fea
dp_tes_lab = args.dp_tes_lab
dp_tra_fea = args.dp_tra_fea
dp_tra_lab = args.dp_tra_lab
SELECTION = args.selection
print("--------------------------------------------------")
print("Garbage collector: {}".format(gc.isenabled()))
print("--------------------------------------------------")
print("The path of train feature is: {}".format(dp_tra_fea))
tra_data = get_feature(dp_tra_fea)
print("The shape of tra_data is: {}".format(tra_data.shape))
print("--------------------------------------------------")
print("The path of train label is: {}".format(dp_tra_lab))
tra_label = get_label(dp_tra_lab)
print("The shape of tra_label is: {}".format(tra_label.shape))
print("--------------------------------------------------")
print("The path of test feature is: {}".format(dp_tes_fea))
tes_data = get_tes_feature(dp_tes_fea)
print("The shape of tes_data is: {}".format(tes_data.shape))
print("--------------------------------------------------")
print("The path of test label is: {}".format(dp_tes_lab))
tes_label = get_tes_label(dp_tes_lab)
print("The shape of tes_label is: {}".format(tes_label.shape))
print("--------------------------------------------------")
# Convert to tensor
tes_data = t.from_numpy(tes_data).float()
tes_label = t.from_numpy(tes_label.squeeze(1))
tra_data = t.from_numpy(tra_data).float()
tra_label = t.from_numpy(tra_label.squeeze(1))
# Create the dataset
tes = EEGData(tes_data, tes_label)
tra = EEGData(tra_data, tra_label)
# Create the dataloader
tra = t.utils.data.DataLoader(
    dataset=tra,
    batch_size=32,
    shuffle=True,
)
tes = t.utils.data.DataLoader(
    dataset=tes,
    batch_size=32,
    shuffle=False,
)
print("The shape of tes_data is: {}".format(tes_data.shape))
print("The shape of tes_label is: {}".format(tes_label.shape))
print("The shape of tra_data is: {}".format(tra_data.shape))
print("The shape of tra_label is: {}".format(tra_label.shape))
print("--------------------------------------------------")
