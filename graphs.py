import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('clean_dataset.txt')
fig, ax = plt.subplots(7, 2, sharex='col', sharey='row')
for i in range(7):
    x1 = data[data[:, -1] == 1][:, i]
    x2 = data[data[:, -1] == 2][:, i]
    x3 = data[data[:, -1] == 3][:, i]
    x4 = data[data[:, -1] == 4][:, i]
    bins_list = np.arange(min(data[:, i]), max(data[:, i]), step=1)
    ax[i, 0].hist([x1, x2, x3, x4], bins_list, stacked=True, density=True, rwidth=0.8)
data = np.loadtxt('noisy_dataset.txt')
for i in range(7):
    x1 = data[data[:, -1] == 1][:, i]
    x2 = data[data[:, -1] == 2][:, i]
    x3 = data[data[:, -1] == 3][:, i]
    x4 = data[data[:, -1] == 4][:, i]
    bins_list = np.arange(min(data[:, i]), max(data[:, i]), step=1)
    ax[i, 1].hist([x1, x2, x3, x4], bins_list, stacked=True, density=True, rwidth=0.8)
fig
