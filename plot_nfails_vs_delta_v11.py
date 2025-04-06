
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_palette("tab20")
colors = sns.color_palette("bright")

import numpy as np
from scipy.stats import kurtosis

np.set_printoptions(precision=4)


version = "v12"

algo_names = ['se_orig', 'se_t4', 'lucb', 'tstci', 'fcsh-1.01',
              'fcsh-1.1', 'fcsh-2', ]
algo_names = ['lucb', 'tstci', 'fcsh-1.01', 'fcsh-1.1']
algo_names = ['lucb', 'lucb_t2', 'fcsh-1.01']
# algo_names = ['lucb', 'tstci', 'fcsh-1.1', 'se_t4']
algo_names = ['se_orig']

colors = ['skyblue', 'g','r', 'y', 'b', 'orange']

max_iter = 999999
n_trials = 100000

delta_list = [0.001, 0.005, 0.01, 0.025, 0.05]
# delta_list = [0.01, 0.03, 0.05, 0.07, 0.09]
delta_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]

for algo_idx, algo_name in enumerate(algo_names):
    num_fails_list = []
    per_fails_list = []
    for (_, delta) in enumerate(delta_list):
        filename = f"final_results/all_stop_times_{algo_name}_{n_trials}_{delta}_{version}.txt"
        print(filename)
        all_stopping_times = np.loadtxt(filename)
        # print(all_stopping_times)
        # stop
        # all_stopping_times = all_stopping_times[:100000]
    
        # stop
        print(f"max = {np.max(all_stopping_times):0.4f}")
        print(f"min = {np.min(all_stopping_times):0.4f}")
        num_fails = np.sum(all_stopping_times == max_iter)
        print(f"num fails = {num_fails} ({num_fails/n_trials*100:0.2f}%)")
        num_fails_list.append(num_fails)
        per_fails_list.append(num_fails/n_trials*100)
    
    # plt.plot(delta_list, num_fails_list,
    #          label=f"{algo_name}", color=colors[algo_idx])
    plt.plot(np.log10(delta_list), per_fails_list,
             label=f"{algo_name}", color=colors[algo_idx])
        
plt.xlabel('Delta', fontsize=13)
plt.ylabel('Percentage of failed trials (in %)', fontsize=13)
plt.yticks(per_fails_list)
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
x_range = [5, 4, 3, 2, 1]
x_names = [f"10^{-x}" for x in x_range]
plt.xticks(np.log10(delta_list), x_names)

plt.legend(fontsize=15)
plt.savefig(f"se_orig_nfails_vs_delta_{n_trials}_{version}.png", format='png')
# plt.savefig(f"fc_bai_comparison_{n_trials}_{version}.pdf", format='pdf')

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


