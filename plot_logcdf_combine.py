
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_palette("tab20")
colors = sns.color_palette("bright")
colors = sns.color_palette("Paired", 12)


import numpy as np
from scipy.stats import kurtosis, norm 

np.set_printoptions(precision=4)

# from empiricaldist import Cdf


version = "v12"

algo_names = ['se_orig', 'se_t4', 'lucb', 'tstci', 'fcsh-1.01',
              'fcsh-1.1', 'fcsh-2', ]
algo_names = ['lucb', 'tstci', 'fcsh-1.01']
# algo_names = ['lucb_p', 'lucb_t4', 'tstci', 'fcsh-1.01']
# algo_names = ['lucb_t0']
# algo_names = ['lucb_p']
# algo_names = ['fcsh-1.01']
# algo_names = ['lucb']
# algo_names = ['tstci']
# algo_names = ['lucb', 'tstci', 'fcsh-1.1', 'se_t4']
# algo_names = ['fcsh-1.01', 'fcsh-1.1', 'fcsh-2']

# colors = ['g','r', 'y', 'b', 'orange']
# colors = ['g','r', 'y', 'b', 'orange', 'purple']

max_iter = 999999
n_trials = 1000000


def plot_data(samples, dist_name, idx):
    samples -= np.mean(samples)
    samples_sorted = np.sort(samples)
    samples_cdf = np.arange(1, len(samples_sorted) + 1) / len(samples_sorted)
    samples_tail = np.log(1 - samples_cdf)
    _samples_sorted = samples_sorted[samples_tail != -np.inf]
    _samples_cdf = samples_cdf[samples_tail != -np.inf]
    _samples_tail = samples_tail[samples_tail != -np.inf]

    # plt.plot(_samples_sorted, _samples_cdf,
    #          label=f"{dist_name}", color=colors[idx])
    
    plt.plot(_samples_sorted, _samples_tail,
             label=f"{dist_name}", color=colors[idx])
    
    # plt.hist(
    #     samples, bins=50,
    #     label=f"{dist_name}", lw=3, alpha=0.5,  
    #     color=colors[idx],
    #     edgecolor=colors[idx],
    # )

    # plt.legend()
    # plt.show()

    
algo_name = algo_names[0]
part_idxes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# part_idxes = [0, 1, 2, '2a', 3, 4, 5, 6, 7, 8, 9]
# part_idxes = [4, 5, 7, 9]
all_stop_times_combined = []
checks = []
for algo_idx, part_idx in enumerate(part_idxes):

    filename = f"final_results/all_stop_times_{algo_name}_{n_trials}_{version}_{part_idx}.txt"
    all_stop_times = np.loadtxt(filename)
    all_stop_times_combined.append(all_stop_times)
    checks.append(all_stop_times[0])
    # print(all_stop_times[:10])
    # all_stop_times = all_stop_times[:10000]
    print(len(all_stop_times))
     
    if algo_name == 'lucb':
        algo_name = 'LUCB1'
    elif algo_name == 'tstci':
        algo_name = 'TS-TCI'
    elif algo_name == 'fcsh-1.01' or algo_name == 'fcsh-1.1':
        algo_name = 'FC-DSH'

    
    plot_data(all_stop_times, part_idx, algo_idx)
    
    
print(checks)

all_stop_times_combined = np.concatenate(all_stop_times_combined)
plot_data(all_stop_times_combined, 'combined', 10)

plt.xlabel('Stopping time', fontsize=13)
plt.ylabel('CDF', fontsize=13)
# plt.title(f'n_rigged = {n_rigged}', fontsize=13)
# plt.xticks(np.arange(6, 8))

plt.legend(fontsize=15)

# plt.savefig(f"cdf_plot_sep_{algo_name}_{n_trials}_{version}.png", format='png')
plt.savefig(f"logcdf_plot_sep_{algo_name}_{n_trials}_{version}.png", format='png')
# plt.savefig(f"cdf_plot_sep_{algo_name}_{version}.pdf", format='pdf')

plt.show()



