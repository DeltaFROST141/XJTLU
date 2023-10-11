import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_and_inspect_mat_file(filepath):
    try:
        data = scipy.io.loadmat(filepath)
    except FileNotFoundError:
        print(f"File {filepath} not found.")
        return
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return

    print(f"Data type of the loaded file: {type(data)}\n")

    # Print all keys in the .mat file
    print("Keys in the .mat file:")
    for key in data:
        print(key)

    # Check and print the shape of data associated with a specific key
    specific_key = 'C1'
    if specific_key in data:
        print(f"\nData type of {specific_key}: {type(data[specific_key])}")
        print(f"Shape of {specific_key}: {data[specific_key].shape}")
    else:
        print(f"\nKey {specific_key} not found in the .mat file.")


def get_feature(dp):
    # Read mat file
    try:
        _mat = scipy.io.loadmat(dp)
    except NotImplementedError:
        import hdf5storage
        _mat = hdf5storage.loadmat(dp)
    # print the keys
    print(_mat.keys())

    data_C1 = _mat['C1']
    data_C2 = _mat['C2']

    # 将两个键的数据切分为训练集和测试集
    train_data_C1, test_data_C1 = train_test_split(data_C1, test_size=0.10, random_state=42)

    train_data_C2, test_data_C2 = train_test_split(data_C2, test_size=0.10, random_state=58)

    # Extract the data
    # if '_1' in dp:
    #     _mat = _mat['gsi1']
    # elif '_2' in dp:
    #     _mat = _mat['gsi2']
    # else:
    #     raise ValueError('Wrong data path, key not found.')
    """"
    # The mat file is a 4 dimensional array
    # The first dimension is the number of samples
    # The second dimension is the number of channels
    # The third and fourth dimensions are the height and width of the image
    """
    # # Convert the data to a numpy array
    train_data_C1 = np.array(train_data_C1)
    test_data_C1 = np.array(test_data_C1)
    train_data_C2 = np.array(train_data_C2)
    test_data_C2 = np.array(test_data_C2)

    print(type(train_data_C1))
    # Print the shape of the data
    print('The shape of the training feature is: {}'.format(train_data_C1.shape))
    print('The shape of the testing feature is: {}'.format(test_data_C1.shape))
    print('The shape of the testing feature is: {}'.format(train_data_C2.shape))
    print('The shape of the testing feature is: {}'.format(test_data_C2.shape))

    return train_data_C1, test_data_C1, train_data_C2, test_data_C2

