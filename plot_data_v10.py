
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_palette("tab20")
colors = sns.color_palette("bright")

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

version = "v22"

algo_names = ['se_orig', 'se_t4', 'lucb', 'tstci', 'fcsh-1.01',
              'fcsh-1.1', 'fcsh-2', ]
algo_names = ['lucb', 'tstci', 'fcsh-1.01', 'fcsh-1.1']
algo_names = ['lucb', 'tstci', 'fcsh-1.01']
algo_names = ['fcsh-1.01', 'fcsh-1.1', 'fcsh-2']

algo_names = ['se_orig']
# algo_names = ['se_t4']

colors = ['skyblue','g','r', 'y', 'b', 'orange']

max_iter = 999999
n_trials = 1000

for algo_idx, algo_name in enumerate(algo_names):
    
    filename = f"final_results/all_stopping_times_{algo_name}_1000_{version}.txt"
    all_stopping_times = np.loadtxt(filename)
    all_stopping_times_ignore = []
    for tau in all_stopping_times:
        if tau != max_iter:
            all_stopping_times_ignore.append(tau)
    all_stopping_times = all_stopping_times_ignore
    num_fails = np.sum(all_stopping_times == max_iter)
    # print(f"num_fails = {num_fails}")
    # stop
    # all_stopping_times[all_stopping_times == max_iter] = 100000

    if algo_name == 'se_orig':
        algo_name = 'SE'
    # print(len(all_stopping_times))
    print(f"max = {np.max(all_stopping_times):0.4f}")
    print(f"min = {np.min(all_stopping_times):0.4f}")
    num_fails = np.sum(all_stopping_times == max_iter)
    print(f"num fails = {num_fails} ({num_fails/n_trials:0.2f}%)")
    # kurt = kurtosis(all_stopping_times, fisher=False)
    # print(f"kurt = {kurt}")
    # hill = hill_estimator(all_stopping_times, 5)
    # print(f"hill = {hill}")

    plt.hist(
        all_stopping_times, bins=200,
        label=f"{algo_name}", lw=3, alpha=0.5,  
        color=colors[algo_idx],
        edgecolor=colors[algo_idx],
    )


plt.xlabel('Stopping time', fontsize=13)
plt.ylabel('Number of Trials', fontsize=13)

plt.legend(fontsize=15)

# plt.savefig(f"fc_dsh_{algo_name}_{version}.png", format='png')
# plt.savefig(f"{algo_name}_{version}.pdf", format='pdf')

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


