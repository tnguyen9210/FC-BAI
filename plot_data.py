
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import kurtosis

np.set_printoptions(precision=4)

def hill_estimator(data, k):
    """
    Estimate right-tail thickness using the Hill estimator (NumPy only).
    data: 1D NumPy array of samples
    k: number of top-order statistics (e.g., top 50 largest values)
    Returns estimated tail index alpha (smaller alpha => thicker tail)
    """
    data = np.sort(data)[::-1]  # sort descending
    x_k = data[k]
    top_k = data[:k]
    hill = np.mean(np.log(top_k) - np.log(x_k))
    return 1 / hill  # Tail index α


# filename = "all_stopping_time.txt"
# all_stopping_times = np.loadtxt(filename)
# print(all_stopping_times)
# print(len(all_stopping_times))

color_list = ['skyblue','g','r']

# filename = "results/all_stopping_time_fcsh-1.1_9_v21.txt"
# all_stopping_times = np.loadtxt(filename)
# print(len(all_stopping_times))
# # kurt = kurtosis(all_stopping_times, fisher=False)
# # print(f"kurt = {kurt}")
# # hill = hill_estimator(all_stopping_times, 5)
# # print(f"hill = {hill}")

# plt.hist(
#     all_stopping_times, bins=10, color=color_list[0],
#     alpha=0.5, edgecolor=color_list[0], label="FC-DSH", lw=3)

# plt.xlabel('Stopping time', fontsize=13)
# plt.ylabel('Number of Trials', fontsize=13)

# plt.legend(fontsize=15)
# plt.savefig(f"fc_dsh.png", format='png')

# plt.show()


filename = "results/all_stopping_times_lucb_1000_v41.txt"
all_stopping_times = np.loadtxt(filename)
print(len(all_stopping_times))
print(f"max = {np.max(all_stopping_times):0.4f}")
print(f"min = {np.min(all_stopping_times):0.4f}")
kurt = kurtosis(all_stopping_times, fisher=False)
print(f"kurt = {kurt}")
hill = hill_estimator(all_stopping_times, 5)
print(f"hill = {hill}")

plt.hist(
    all_stopping_times, bins=50, color=color_list[0],
    alpha=0.5, edgecolor=color_list[0], label="LUCB", lw=3)

plt.xlabel('Stopping time', fontsize=13)
plt.ylabel('Number of Trials', fontsize=13)

plt.legend(fontsize=15)
plt.savefig(f"fc_dsh.png", format='png')

plt.show()


# filename = "all_stopping_times_se_v11.txt"
# all_stopping_times = np.loadtxt(filename)
# kurt = kurtosis(all_stopping_times, fisher=False)
# print(f"kurt = {kurt}")
# hill = hill_estimator(all_stopping_times, 5)
# print(f"hill = {hill}")

# plt.hist(
#     all_stopping_times, bins=100, color=color_list[0],
#     alpha=0.5, edgecolor=color_list[0], label="SE with LUCB bounds", lw=3)

# plt.xlabel('Stopping time', fontsize=13)
# plt.ylabel('Number of Trials', fontsize=13)

# plt.legend(fontsize=15)
# plt.savefig(f"se_lucb.png", format='png')

# plt.show()


