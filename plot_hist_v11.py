
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

version = "v12"

# algo_names = ['tstci', 'fcsh-1.01']
algo_names = ['lucb', 'tstci', 'fcsh-1.01']
# algo_names = ['lucb', 'tstci', 'fcsh-1.01', 'fcsh-1.001', 'fcsh-1.01-d1.01']
# algo_names = ['lucb', 'tstci', 'fcsh-1.01-d3', 'fcsh-1.01-d4']

colors = ['g','r', 'y', 'b', 'orange']

max_iter = 999999
n_trials = 1000

for algo_idx, algo_name in enumerate(algo_names):
    filename = f"final_results/all_stop_times_{algo_name}_{n_trials}_{version}.txt"
    print(filename)
    all_stopping_times = np.loadtxt(filename)
    # all_stopping_times = all_stopping_times[:100000]
    
    if algo_name == 'lucb':
        algo_name = 'LUCB1'
    elif algo_name == 'tstci':
        algo_name = 'TS-TCI'
    elif algo_name == 'fcsh-1.01' or algo_name == 'fcsh-1.1':
        algo_name = 'FC-DSH'
    # elif algo_name == 'fcsh-1.01-noreuse' or algo_name == 'fcsh-2-noreuse':
    #     algo_name = 'FC-DSH-no-reuse'
    # print(all_stopping_times[:50])
    # print(len(all_stopping_times))
    # # print(all_stopping_times)
    
    print(f"max = {np.max(all_stopping_times):0.4f}")
    print(f"min = {np.min(all_stopping_times):0.4f}")
    num_fails = np.sum(all_stopping_times == max_iter)
    print(f"num fails = {num_fails} ({num_fails/n_trials:0.2f}%)")

    num_tails = np.sum(all_stopping_times > 5000)
    print(f"num tails = {num_tails} ({num_tails/n_trials*100:0.2f}%)")
    
    plt.hist(
        all_stopping_times, bins=50,
        label=f"{algo_name}", lw=3, alpha=0.7,  
        color=colors[algo_idx],
        edgecolor=colors[algo_idx],
    )


plt.xlabel('Stopping time', fontsize=13)
plt.ylabel('Number of Trials', fontsize=13)

plt.legend(fontsize=15)
plt.savefig(f"figures/plot_hist_{n_trials}_{version}.pdf", format='pdf')

plt.show()



