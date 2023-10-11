import scipy.io
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

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

filepath = 'leo_03_loc_grc-main/My_Code/Data/BCI_data.mat'
load_and_inspect_mat_file(filepath)

# TODO 输出的三维是(105, 118, 5000),应该如何理解？
# ? 数据中有105次独立的试验或观测。
# ? 每次试验都用118个电极记录。
# ? 每个电极都记录了5000个时间点的数据。

# ! 三维数据所以无法查看
# plt.plot(data[specific_key])
# plt.show()

# ! 查看第一个键的第一个通道的所有数据
# print(data['C1'][0,0,:])

# # ! 可视化第一个维度
chosen_channel = 0

# 可视化前10个试验
# fig, axs = plt.subplots(10, 1, figsize=(10, 15))

# for i in range(10):
#     axs[i].plot(data['C1'][i, chosen_channel, :])
#     axs[i].set_title('Trial {}'.format(i+1))
#     axs[i].set_xlabel('Time Points')
#     axs[i].set_ylabel('Amplitude')

# plt.tight_layout()
# plt.show()


# # ! 可视化第二个维度
# chosen_trial = 0

# all_channels_data = data['C1'][chosen_trial, :, chosen_time_point]

# plt.plot(all_channels_data)
# plt.xlabel('Channel')
# plt.ylabel('Amplitude')
# plt.title('Data for Trial {} at Time Point {}'.format(chosen_trial, chosen_time_point))
# plt.show()

# # ! 可视化第三个维度
# chosen_trial = 0
# chosen_channel = 0

# time_series_data = data['C1'][chosen_trial, chosen_channel, :]

# plt.plot(time_series_data)
# plt.xlabel('Time Point')
# plt.ylabel('Amplitude')
# plt.title('Time Series Data for Trial {} and Channel {}'.format(chosen_trial, chosen_channel))
# plt.show()


# ! 可视化第一个类三个维度的数据
# channel1 = data['C1'][0, 0, :]
# channel2 = data['C1'][0, 1, :]
# channel3 = data['C1'][0, 2, :]

# plt.figure(figsize=(10,6))
# plt.plot(channel1, label="Channel 1")
# plt.plot(channel2, label="Channel 2")
# plt.plot(channel3, label="Channel 3")
# plt.legend()
# plt.title('Data from Trial 1 for First Three Channels')
# plt.xlabel('Time Points')
# plt.ylabel('Amplitude')
# plt.show()

