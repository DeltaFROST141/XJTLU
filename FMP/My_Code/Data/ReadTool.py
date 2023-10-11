import scipy.io
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# data = scipy.io.loadmat('leo_03_loc_grc-main/My_Code/Data/BCI_data.mat')
# print(data.keys())

def get_feature(dp):
    # Read mat file
    try:
        _mat = scipy.io.loadmat(dp)
    except NotImplementedError:
        import hdf5storage
        _mat = hdf5storage.loadmat(dp)
    # print the keys
    print(_mat.keys())
    
    data = _mat['C1']  # 你需要替换 'data_key' 为你实际的数据键
    
    # 切分数据为训练集和测试集
    train_data, test_data = train_test_split(data, test_size=0.10, random_state=42)
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
    # Split the data based on the first dimension
    # The first 90% of the data is used for training
    # The last 10% of the data is used for testing
    # training_set = _mat[:int(_mat.shape[0] * 0.9), :, :, :]
    # testing_set = _mat[int(_mat.shape[0] * 0.9):, :, :, :]
    # # Convert the data to a numpy array
    # training_set = np.array(training_set)
    # testing_set = np.array(testing_set)
    # Print the shape of the data
    print('The shape of the training feature is: {}'.format(train_data.shape))
    print('The shape of the testing feature is: {}'.format(test_data.shape))
    return train_data, test_data

get_feature('/Users/zcq30/Desktop/FMP/FMP_Code/leo_03_loc_grc-main/My_Code/Data/BCI_data.mat')
