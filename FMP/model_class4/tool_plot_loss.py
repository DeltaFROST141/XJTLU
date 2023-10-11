# Plot the loss list
import numpy as np
import matplotlib.pyplot as plt

# Using times new roman font
plt.rcParams['font.sans-serif'] = ['Times New Roman']

# Load the loss list
loss_list = np.loadtxt(
    './model/loss_list_2023-05-23-14-08-56.txt',
    dtype=np.float32,
)

fig = plt.figure()
plt.plot(
    loss_list,
    color='blue',
    linewidth=0.5,
    linestyle='-',
)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(
    axis='y',
    linestyle='--',
    linewidth=0.5,
)
plt.ylim(0, 1)
plt.savefig(
    './model/loss_list.png',
    bbox_inches='tight',
    dpi=256,
)
