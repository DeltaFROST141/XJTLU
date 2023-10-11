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
        _mat = hdf5storage.loadmat(dp)
    _items = []
    for i in _mat.keys():
        _items.append(i)
    print("The keys of feature are: {}".format(_items))
    return _mat[_items[0]], _mat[_items[1]], _mat[_items[2]], _mat[_items[3]]


def get_label(dp):
    # Read mat file
    try:
        _mat = scipy.io.loadmat(dp)
    except NotImplementedError:
        import hdf5storage
        _mat = hdf5storage.loadmat(dp)
    _items = []
    for i in _mat.keys():
        _items.append(i)
    print("The keys of label .mat file are: {}".format(_items))
    return _mat[_items[0]], _mat[_items[1]], _mat[_items[2]], _mat[_items[3]]


def concat_data(_d1, _d2, _d3, _d4):
    # Concatenate data
    _d1 = np.asarray(_d1)
    _d2 = np.asarray(_d2)
    _d3 = np.asarray(_d3)
    _d4 = np.asarray(_d4)
    # Check the shape of each data
    if _d1.shape != _d2.shape or _d3.shape != _d4.shape or _d2.shape != _d4.shape:
        raise ValueError("The shape of each data is not equal.")
    return np.concatenate((_d1, _d2, _d3, _d4), axis=0)


def concat_label(_l1, _l2, _l3, _l4):
    # Concatenate label
    _l1 = np.asarray(_l1) - np.ones(_l1.shape)
    _l2 = np.asarray(_l2) - np.ones(_l2.shape)
    _l3 = np.asarray(_l3) - np.ones(_l3.shape)
    _l4 = np.asarray(_l4) - np.ones(_l4.shape)
    # Check the shape of each label
    if _l1.shape != _l2.shape or _l3.shape != _l4.shape or _l2.shape != _l4.shape:
        raise ValueError("The shape of each label is not equal.")
    return np.concatenate((_l1, _l2, _l3, _l4), axis=0)


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


# def test():
#     os.system("clear")

#     print("--------------------------------------------------")
#     print("Garbage collector: {}".format(gc.isenabled()))
#     print("--------------------------------------------------")
#     dp_tra_fea = "../data/te1.mat"
#     d1, d2, d3, d4 = get_feature(dp_tra_fea)
#     data = concat_data(d1, d2, d3, d4)
#     print("The shape of data is: {}".format(data.shape))
#     print("--------------------------------------------------")
#     dp_tra_lab = "../data/le1.mat"
#     l1, l2, l3, l4 = get_label(dp_tra_lab)
#     label = concat_label(l1, l2, l3, l4)
#     print("The shape of label is: {}".format(label.shape))
#     # label = one_hot_encoding(label)
#     # print("The shape of label after one-hot encoding is: {}".format(
#     #     label.shape))
#     print(label)
#     print("--------------------------------------------------")
#     """
#     The keys of feature are: ['gsi1', 'gsi2', 'gsi3', 'gsi4']
#     The shape of each is: (512, 3, 224, 224)
#     The keys of label .mat file are: ['c1', 'c2', 'c3', 'c4']
#     The shape of each is: (512, 1)
#     """


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
    default="../old_data/label_tr_4.mat",
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
    default="../old_data/label_te_4.mat",
)
args = parser.parse_args()
# Get the data
dp_tes_fea = args.dp_tes_fea
dp_tes_lab = args.dp_tes_lab
dp_tra_fea = args.dp_tra_fea
dp_tra_lab = args.dp_tra_lab
print("--------------------------------------------------")
print("Garbage collector: {}".format(gc.isenabled()))
print("--------------------------------------------------")
d1, d2, d3, d4 = get_feature(dp_tes_fea)
tes_data = concat_data(d1, d2, d3, d4)
print("--------------------------------------------------")
l1, l2, l3, l4 = get_label(dp_tes_lab)
# tes_label = one_hot_encoding(concat_label(l1, l2, l3, l4))
tes_label = concat_label(l1, l2, l3, l4)
print("--------------------------------------------------")
d1, d2, d3, d4 = get_feature(dp_tra_fea)
tra_data = concat_data(d1, d2, d3, d4)
print("--------------------------------------------------")
l1, l2, l3, l4 = get_label(dp_tra_lab)
# tra_label = one_hot_encoding(concat_label(l1, l2, l3, l4))
tra_label = concat_label(l1, l2, l3, l4)
print("--------------------------------------------------")
# Convert to tensor
tes_data = t.from_numpy(tes_data).float()
tes_label = t.from_numpy(tes_label.squeeze(1))
# tes_label = t.from_numpy(tes_label).float()
tra_data = t.from_numpy(tra_data).float()
tra_label = t.from_numpy(tra_label.squeeze(1))
# tra_label = t.from_numpy(tra_label).float()
# Create the dataset
tes = EEGData(tes_data, tes_label)
tra = EEGData(tra_data, tra_label)
# Create the dataloader
tes = t.utils.data.DataLoader(
    dataset=tes,
    batch_size=32,
    shuffle=False,
)
tra = t.utils.data.DataLoader(
    dataset=tra,
    batch_size=16,
    shuffle=True,
)
print("The shape of tes_data is: {}".format(tes_data.shape))
print("The shape of tes_label is: {}".format(tes_label.shape))
print("The shape of tra_data is: {}".format(tra_data.shape))
print("The shape of tra_label is: {}".format(tra_label.shape))
print("--------------------------------------------------")
