
import numpy as np

algo_name = 'lucb_t0'
version = 'v12'
n_trials = 1000000

part_idxes = [3, 5, 6, 7]
all_stop_times = []
for part_idx in part_idxes:
    filename = f"final_results/all_stop_times_{algo_name}_{n_trials}_{version}_{part_idx}.txt"
    print(filename)
    all_stop_times.append(np.loadtxt(filename))

all_stop_times = np.concatenate(all_stop_times)
print(len(all_stop_times))

filename = f"final_results/all_stop_times_{algo_name}_{n_trials}_{version}_full.txt"
np.savetxt(filename, all_stop_times)
