
import time
import random 

import numpy as np
np.set_printoptions(precision=4)

import multiprocess as mp
from types import SimpleNamespace

from se_algo import se_orig, se_t4

def create_algo(algo_name):
    if algo_name == 'se_orig':
        algo = se_orig
    else:
        algo = se_t4
    return algo 


manager = mp.Manager()
all_stop_times = manager.list()

def run_trial(i_trial, K, algo, opt):
    random.seed(10000+i_trial)
    np.random.seed(10000+i_trial)
    
    output, stop_time = algo(K, opt.arm_mus, opt.sigma, opt.delta, opt.max_iter)
    all_stop_times.append(stop_time)

    return 


np.random.seed(1)

opt = SimpleNamespace()
    
opt.sigma = 1
opt.delta = 0.05
opt.max_iter = 1000000

version = "v12"
K = 4
mu_best = 1
mu_sub = 1 - 0.4
opt.arm_mus = [mu_best] + [mu_sub]*(K-1)
    
n_trials = 1000

print(f"K = {K}")
print(f"arm_mus = {opt.arm_mus}")
print(f"max_iter = {opt.max_iter}")
print(f"n_trials = {n_trials}")

algo_names = ["se_t4", "se_orig"]
algo_names = ["se_orig"]

delta_list = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
# delta_list = [delta for delta in np.linspace(0.00, 0.05, 11)]
# print(delta_list)
# stop
# delta_list = [0.02, 0.03, 0.05, 0.07, 0.09, 0.11, 0.]
for (i_algo, algo_name) in enumerate(algo_names):
    print(f"\n-> algo_name = {algo_name}")
    algo = create_algo(algo_name)
    for (_, delta) in enumerate(delta_list):
        opt.delta = delta
        print(f"delta = {opt.delta}")
        trial_args = [(i_trial, K, algo, opt) for i_trial in range(n_trials)]
        
        start_time = time.time()
        pool = mp.Pool()
        pool.starmap(run_trial, trial_args)
        pool.close()
    
        total_time = time.time() - start_time
        print(f"it takes {total_time:0.4f}s")
        print(f"it takes {total_time/(n_trials+1):0.4f}s per trial")

        # print(all_stop_times)
        filename = f"results/all_stop_times_{algo_name}_{n_trials}_{opt.delta}_{version}.txt"
        np.savetxt(filename, np.array(all_stop_times))

        all_stop_times[:] = []
