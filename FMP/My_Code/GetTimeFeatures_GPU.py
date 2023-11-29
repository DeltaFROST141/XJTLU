import torch
import numpy as np
import time
from GetTimeFeatures import getTimeFea 

def getTimeFeaGPU(input, granule_size):
    # 将输入转换为 PyTorch 张量
    input_tensor = torch.from_numpy(input).float().to(device='mps')

    x, y = input_tensor.shape
    granule_num = y // granule_size

    # 初始化输出张量
    time_feature_output = torch.zeros((x, granule_num, 2), device=input_tensor.device)

    # 使用 PyTorch 的视图(view)和张量操作来计算最小值和最大值
    input_tensor = input_tensor.unfold(1, granule_size, granule_size)
    time_feature_output[:, :, 0] = torch.min(input_tensor, dim=2).values
    time_feature_output[:, :, 1] = torch.max(input_tensor, dim=2).values

    return time_feature_output

# 示例使用
time_start = time.time()
input_np = np.random.random((84, 5000))  # 例如，84x5000 的输入
granule_size = 100

print(getTimeFeaGPU(input_np, granule_size))
time_end = time.time()

print(f"Running time is:{time_end - time_start}")