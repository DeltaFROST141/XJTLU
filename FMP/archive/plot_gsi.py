import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Using Times New Roman font
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16

# Read the txt data, split by ',' and convert to float
m1 = pd.read_csv(
    './b1.txt',
    header=None,
    sep=',',
    dtype=np.float64,
)
m2 = pd.read_csv(
    './b2.txt',
    header=None,
    sep=',',
    dtype=np.float64,
)
m1 = m1.values[0, :]
m2 = m2.values[0, :]
# Plot the data
fig = plt.figure(figsize=(15, 5))
plt.plot(
    m1[1:],
    label='b1',
    linestyle='-',
    linewidth=1,
    color='red',
    alpha=0.5,
)
plt.plot(
    m2[1:],
    label='b2',
    linestyle='-',
    linewidth=1,
    color='blue',
    alpha=0.5,
)
plt.xlabel('Time')
plt.ylabel('Value')
plt.ylim(0, 100)
plt.grid(
    axis='y',
    linestyle='--',
)
plt.legend(frameon=False, loc='upper right', ncol=2,)
plt.savefig(
    '../model/input_example.png',
    bbox_inches='tight',
    dpi=256,
)