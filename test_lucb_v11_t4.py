
import time
import random 

import importlib
import kjbandits
importlib.reload(kjbandits)
from kjbandits import *

import numpy as np
np.set_printoptions(precision=4)

import multiprocess as mp


def algo_factory_fc(algo_name, K, seed, sigma_sq, beta, delta):
    if algo_name == 'fcsh-2':
        algo = FCDoublingSequentialHalving(K, seed=seed, factor=2.0)
    elif algo_name == 'fcsh-1.5':
        algo = FCDoublingSequentialHalving(K, seed=seed, factor=1.5)
    elif algo_name == 'fcsh-1.1':
        algo = FCDoublingSequentialHalving(K, seed=seed, factor=1.1, divisor=2)
    elif algo_name == 'fcsh-1.05':
        algo = FCDoublingSequentialHalving(K, seed=seed, factor=1.05, divisor=2)
    elif algo_name == 'fcsh-1.01':
        algo = FCDoublingSequentialHalving(K, seed=seed, factor=1.01, divisor=2)
    elif algo_name == 'fcsh-1.01-d1.01':
        algo = FCDoublingSequentialHalving(K, seed=seed, factor=1.01, divisor=1.01)
    elif algo_name == 'fcsh-1.01-d2.5':
        algo = FCDoublingSequentialHalving(K, seed=seed, factor=1.01, divisor=2.5)
    elif algo_name == 'fcsh-1.01-d3':
        algo = FCDoublingSequentialHalving(K, seed=seed, factor=1.01, divisor=3)
    elif algo_name == 'fcsh-1.01-d4':
        algo = FCDoublingSequentialHalving(K, seed=seed, factor=1.01, divisor=4)
    elif algo_name == 'fcsh-1.01-d5':
        algo = FCDoublingSequentialHalving(K, seed=seed, factor=1.01, divisor=5)
    elif algo_name == 'tstci':
        algo = FCTsTci(K, beta, sigma_sq, seed)
    elif algo_name == "lucb":
        algo = Lucb(K, sigma_sq, delta)
    else:
        raise ValueError()
    
    return algo


opt = SimpleNamespace()

opt.delta = .05
opt.dataseed = 103
opt.max_iter = 1000000
opt.sigma_sq = 1.0 ** 2
opt.algoseed = 29
opt.beta = .5

version = 'v11_t4'
K = 4
mu_opt = 1.0
mu_sub = 1.0 - 0.1
opt.mu = [mu_opt] + [mu_sub]*(K-1)

n_trials = 10000

print(f"mus = {opt.mu}")
print(f"num_trials = {n_trials}")

# algo_names = ['tstci', 'fcsh-2', 'fcsh-1.5', 'fcsh-1.01']
# algo_names = ['fcsh-1.01', 'fcsh-1.1', 'fcsh-2']
algo_names = ['lucb']

emax_mat = np.zeros((len(algo_names), n_trials))
best_arm_mat = np.zeros((len(algo_names), n_trials)) 
seed_ary = gen_seeds2(n_trials, seed=opt.dataseed)

num_cores = mp.cpu_count()
print(f"Number of CPU cores: {num_cores}")

manager = mp.Manager()
all_stop_times = manager.list()

def run_trial(
        i_trial, K, algo_name, opt):
    # random.seed()
    random.seed(10000)
    np.random.seed(10000)

    env = Gaussian(opt.mu, opt.sigma_sq, seed=seed_ary[i_trial])
    algo = algo_factory_fc(
        algo_name, K, opt.algoseed + i_trial, opt.sigma_sq, opt.beta, opt.delta)
    
    tau, is_top = run_bandit_lucb_t4(
        algo, env, opt.delta, opt.max_iter, opt.sigma_sq)
    # print(tau)
    
    all_stop_times.append(tau)

    return

K = len(opt.mu)

for (i_algo, algo_name) in enumerate(algo_names):
    print(f"\n-> algo_name = {algo_name}")

    trial_args = [(i_trial, K, algo_name, opt) for i_trial in range(n_trials)]
    
    start_time = time.time()
    pool = mp.Pool()
    pool.starmap(run_trial, trial_args)
    pool.close()
    
    total_time = time.time() - start_time
    print(f"it takes {total_time:0.4f}s")
    print(f"it takes {total_time/(n_trials+1):0.4f}s per trial")
    
    all_stop_times = np.array(all_stop_times)
    filename = f"results/all_stop_times_{algo_name}_{n_trials}_{version}.txt"
    np.savetxt(filename, all_stop_times)


