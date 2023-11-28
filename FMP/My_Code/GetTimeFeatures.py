import numpy as np

def getTimeFea(input, granule_size):
    x, y = input.shape
    granule_num = y // granule_size

    time_feature_output = np.zeros((x, granule_num, 2))

    for i in range(x):
        for j in range(granule_num):
            window_data = input[i, j * granule_size:(j + 1) * granule_size]
            time_feature_output[i, j, 0] = np.min(window_data)  # 第一列为最小，第二列为最大
            time_feature_output[i, j, 1] = np.max(window_data)  # (84,100,2)

    return time_feature_output