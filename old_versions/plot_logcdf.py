
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

# colors = ['g','r', 'y', 'b', 'orange']
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

    # plt.plot(_samples_sorted, _samples_cdf,
    #          label=f"{dist_name}", color=colors[idx])
    
    plt.plot(_samples_sorted, _samples_tail_log,
             label=f"{dist_name}", color=colors[idx])
    
    # plt.plot(_samples_sorted_log, _samples_tail,
    #          label=f"{dist_name}", color=colors[idx])
    
    # plt.hist(
    #     samples, bins=50,
    #     label=f"{dist_name}", lw=3, alpha=0.5,  
    #     color=colors[idx],
    #     edgecolor=colors[idx],
    # )

    # plt.legend()
    # plt.show()

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
    # # print(all_stopping_times)
    # # stop
    # all_stopping_times -= np.mean(all_stopping_times)
    # std = np.std(all_stopping_times, ddof=1)
    # sorted_samples = np.sort(all_stopping_times)
    # # _norm = norm(loc=0, scale=1)
    # cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    # res = np.log(1 - cdf)
    # _sorted_samples = sorted_samples[res != -np.inf]
    # _cdf = cdf[res != -np.inf]
    # _res = res[res != -np.inf]
    # # print(res[-10:])
    # # print(res[res!=-np.inf][-10:])
    # # stop

    # # res = norm.logsf(sorted_samples, loc=0, scale=std)
    # # res = norm.cdf(sorted_samples)
    # # print(len(res))
    # _xlog = np.log(_sorted_samples)
    # # plt.plot(x[x> 6], res[x> 6], label=f"{algo_name}", color=colors[algo_idx])
    # # plt.plot(x, res, label=f"{algo_name}", color=colors[algo_idx])
    # # plt.plot(sorted_samples[res!=-np.inf], res[res!=-np.inf],
    # #          label=f"{algo_name}", color=colors[algo_idx])
    # # plt.plot(_xlog[_xlog > 9], _res[_xlog > 9],
    # #          label=f"{algo_name}", color=colors[algo_idx])
    # plt.plot(_xlog, _res,
    #          label=f"{algo_name}", color=colors[algo_idx])
    # # plt.plot(sorted_samples, cdf,
    # #          label=f"{algo_name}", color=colors[algo_idx])
    # # plt.show()
    # # stop

    # # cdf = Cdf.from_seq(all_stopping_times)
    # # print(cdf.shape)
    # # print()
    # # # print(cdf)
    # # print(len(np.array(cdf)))
    # # print(len(np.array(cdf[:10])))
    # # print(cdf[:10])
    # # cdf.plot(label=f"{algo_name}", color=colors[algo_idx])
    # # res = np.log(1-cdf)
    
    # # print(all_stopping_times[:50])
    # # print(len(all_stopping_times))
    # # # print(all_stopping_times)
    # # stop
    # print(f"max = {np.max(all_stopping_times):0.4f}")
    # print(f"min = {np.min(all_stopping_times):0.4f}")
    # num_fails = np.sum(all_stopping_times == max_iter)
    # print(f"num fails = {num_fails} ({num_fails/n_trials:0.2f}%)")

    # plt.hist(
    #     all_stopping_times, bins=50,
    #     label=f"{algo_name}", lw=3, alpha=0.5,  
    #     color=colors[algo_idx],
    #     edgecolor=colors[algo_idx],
    # )
    # plt.plot(sorted_samples, cdf, marker='.', linestyle='none',
    #          label=f"{algo_name}", color=colors[algo_idx])

# plt.xlabel('log(Stopping time)', fontsize=13)
# plt.ylabel('log(1-CDF)', fontsize=13)
plt.ylabel('log(P(X > x))', fontsize=13)
plt.xlabel('Stopping time', fontsize=13)
# plt.ylabel('CDF', fontsize=13)
# plt.title(f'n_rigged = {n_rigged}', fontsize=13)
# plt.xticks(np.arange(6, 8))

plt.legend(fontsize=15)

# plt.savefig(f"plot_logcdf_plot_sep_{n_trials}_{version}.png", format='png')
plt.savefig(f"plot_logcdf_plot_sep_{n_trials}_{version}.pdf", format='pdf')

plt.show()



