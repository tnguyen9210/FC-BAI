
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_palette("tab20")
colors = sns.color_palette("bright")

import numpy as np
from scipy.stats import kurtosis, norm 

np.set_printoptions(precision=4)

from empiricaldist import Cdf


version = "v12"

algo_names = ['se_orig', 'se_t4', 'lucb', 'tstci', 'fcsh-1.01',
              'fcsh-1.1', 'fcsh-2', ]
algo_names = ['lucb', 'tstci', 'fcsh-1.01', 'fcsh-1.1']
algo_names = ['lucb', 'tstci', 'fcsh-1.01']
algo_names = ['lucb_p', 'tstci', 'fcsh-1.01']
# algo_names = ['fcsh-1.01']
# algo_names = ['lucb']
# algo_names = ['tstci']
# algo_names = ['lucb', 'tstci', 'fcsh-1.1', 'se_t4']
# algo_names = ['fcsh-1.01', 'fcsh-1.1', 'fcsh-2']

colors = ['g','r', 'y', 'b', 'orange', 'purple']

max_iter = 999999
n_trials = 1000000

def plot_data(samples, dist_name, idx):
    samples -= np.mean(samples)
    # samples /= np.linalg.norm(samples)
    samples_sorted = np.sort(samples)
    samples_cdf = np.arange(1, len(samples_sorted) + 1) / len(samples_sorted)
    samples_tail = 1 - samples_cdf
    _samples_sorted = samples_sorted[samples_tail != -np.inf]
    _samples_cdf = samples_cdf[samples_tail != -np.inf]
    _samples_tail = samples_tail[samples_tail != -np.inf]

    _samples_sorted_log = np.log(_samples_sorted)
    _samples_tail_log = np.log(_samples_tail)

    plt.plot(_samples_sorted, _samples_tail_log,
             label=f"{dist_name}", color=colors[idx])

for algo_idx, algo_name in enumerate(algo_names):
    
    filename = f"final_results/all_stop_times_{algo_name}_{n_trials}_{version}.txt"
    print(filename)
    all_stopping_times = np.loadtxt(filename)
    # all_stopping_times = all_stopping_times[:10000]
    print(len(all_stopping_times))

    if algo_name == 'lucb' or algo_name == 'lucb_p':
        algo_name = 'LUCB1'
    elif algo_name == 'tstci':
        algo_name = 'TS-TCI'
    elif algo_name == 'fcsh-1.01' or algo_name == 'fcsh-1.1':
        algo_name = 'FC-DSH'

    plot_data(all_stopping_times, algo_name, algo_idx)
    
# plt.xlabel('log(Stopping time)', fontsize=13)
# plt.ylabel('log(1-CDF)', fontsize=13)
plt.ylabel('log(P(X > x))', fontsize=13)
plt.xlabel('Stopping time', fontsize=13)
# plt.ylabel('CDF', fontsize=13)
# plt.title(f'n_rigged = {n_rigged}', fontsize=13)
# plt.xticks(np.arange(6, 8))

plt.legend(fontsize=15)

# plt.savefig(f"plot_logcdf_plot_sep_{n_trials}_{version}.png", format='png')
plt.savefig(f"figures/plot_logcdf_plot_sep_{n_trials}_{version}.pdf", format='pdf')

plt.show()



