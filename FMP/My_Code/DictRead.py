from scipy.io import loadmat

# 加载.mat文件，在 Windows 下需要更改路径
data = loadmat('Data/BCI_data.mat')

# 提取键为C1和C2的数据
C1_data = data['C1']
C2_data = data['C2']

print(C1_data.shape)
print(C2_data.shape)

print(C1_data)

# 现在，C1_data和C2_data分别包含了与C1和C2键相关联的数据
