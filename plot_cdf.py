
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
# algo_names = ['fcsh-1.01']
# algo_names = ['lucb']
# algo_names = ['tstci']
# algo_names = ['lucb', 'tstci', 'fcsh-1.1', 'se_t4']
# algo_names = ['fcsh-1.01', 'fcsh-1.1', 'fcsh-2']

# colors = ['g','r', 'y', 'b', 'orange']
colors = ['g','r', 'y', 'b', 'orange']

max_iter = 999999
n_trials = 1000000


def make_model(sample, size=1000):
    mu = np.mean(sample)
    sigma = np.std(sample, ddof=1)
    model = norm(mu, sigma)

    xs = np.linspace(np.min(sample), np.max(sample), size)
    ys = model.cdf(xs)

    return xs, ys

for algo_idx, algo_name in enumerate(algo_names):
    
    filename = f"final_results/all_stop_times_{algo_name}_{n_trials}_{version}.txt"
    print(filename)
    all_stopping_times = np.loadtxt(filename)
    # all_stopping_times = all_stopping_times[:10000]
    print(len(all_stopping_times))

    if algo_name == 'lucb':
        algo_name = 'LUCB1'
    elif algo_name == 'tstci':
        algo_name = 'TS-TCI'
    elif algo_name == 'fcsh-1.01' or algo_name == 'fcsh-1.1':
        algo_name = 'FC-DSH'
        
    # all_stopping_times -= np.mean(all_stopping_times)
    std = np.std(all_stopping_times, ddof=1)
    sorted_samples = np.sort(all_stopping_times)
    cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    
    plt.plot(sorted_samples, cdf,
             label=f"{algo_name}", color=colors[algo_idx])

    print(f"max = {np.max(all_stopping_times):0.4f}")
    print(f"min = {np.min(all_stopping_times):0.4f}")
    num_fails = np.sum(all_stopping_times == max_iter)
    print(f"num fails = {num_fails} ({num_fails/n_trials:0.2f}%)")

plt.xlabel('Stopping time', fontsize=13)
plt.ylabel('CDF', fontsize=13)
# plt.xticks(np.arange(6, 8))

plt.legend(fontsize=15)

# plt.savefig(f"figures/plot_cdf_sep_centered_{n_trials}_{version}.png", format='png')
plt.savefig(f"figures/plot_cdf_sep_{n_trials}_{version}.pdf", format='pdf')
# plt.savefig(f"figures/plot_cdf_sep_centered_{n_trials}_{version}.pdf", format='pdf')

plt.show()



