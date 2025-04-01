
import time  

import importlib
import kjbandits
importlib.reload(kjbandits)
from kjbandits import *

import numpy as np 
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)


opt = SimpleNamespace()
opt.n_try = 1000
#opt.mu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
opt.mu = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0]

K = 4
tmp = np.array([np.sqrt(i/(8+1)) for i in range(1,1+8)])
opt.mu = 1.0 - tmp[::-1]
opt.mu[-1] = 1.0
#1 - np.sqrt(8-i)
#opt.mu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
opt.delta = .05
opt.dataseed = 103
opt.max_iter = 100000
opt.sigma_sq = 1.0 ** 2
opt.algoseed = 29
opt.beta = .5

version = 'v13'
K = 4
mu_opt = 1.0
mu_sub = 1.0 - 0.2
opt.mu = [mu_opt] + [mu_sub]*(K-1)


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

#algo_names = ['ucb', 'sh']
#algo_names = ['ucb', 'sh', 'sh-reuse']
#algo_names = ['tstci', 'fcsh-2', 'fcsh-1.5', 'fcsh-1.01']
algo_names = ['tstci', 'fcsh-1.01', 'fcsh-1.01-d1.01', 'fcsh-1.01-d3', 'fcsh-1.01-d4', 'fcsh-1.01-d5']
algo_names = ['fcsh-1.01', 'fcsh-1.1', 'fcsh-2']
algo_names = ['tstci']
# algo_names = ['lucb']
#algo_names = ['tstci', 'fcsh-2', 'fcsh-1.5', 'fcsh-1.01', 'fcsh-1.01-d3', 'fcsh-1.01-d4']
opt.algo_names = algo_names

emax_mat = np.zeros((len(algo_names),opt.n_try))
best_arm_mat = np.zeros((len(algo_names),opt.n_try)) 
seed_ary = gen_seeds2(opt.n_try,seed=opt.dataseed)
tab = KjTable()

print(f"mus = {opt.mu}")
print(f"num_trials = {opt.n_try}")

K = len(opt.mu)
n_pulls = np.zeros((len(algo_names), opt.n_try, K))
for (i_algo, algo_name) in enumerate(algo_names):
    print(f"\n-> algo_name = {algo_name}")
    start_time = time.time()
    all_stopping_times = []
    for i_try in range(opt.n_try):
        env = Gaussian(opt.mu, opt.sigma_sq, seed=seed_ary[i_try])
        algo = algo_factory_fc(algo_name, K, opt.algoseed + i_try, opt.sigma_sq, opt.beta, opt.delta)
        tau, is_stop = run_bandit_pe(algo, env, opt.delta, opt.max_iter, opt.sigma_sq)
        # tau, is_stop = run_bandit_lucb(algo, env, opt.delta, opt.max_iter, opt.sigma_sq)

        # ext = res.extract()
        all_stopping_times.append(tau)

        # tab.update('tau', (i_algo, i_try), ext.tau[0])
        # n_pulls[i_algo, i_try, :] = ext.n_pulls[0]
        if (i_try == 0) or ((i_try + 1) % 50 == 0):
            print(f"trial {i_try}, stopping time = {tau}")
            total_time = time.time() - start_time
            print(f"it takes {total_time:0.4f}s")
            print(f"it takes {total_time/(i_try+1):0.4f}s per trial")
            np.savetxt(f"results/all_stopping_times_{algo_name}_{i_try+1}_{version}.txt", all_stopping_times)

    np.savetxt(f"results/all_stopping_times_{algo_name}_{i_try+1}_{version}.txt", all_stopping_times)


# #--------
# printExpr("opt")
# print("")
# print(tab)
# tab = tab.extract()
# print(tab)
# print(tab.tau.mean(1))
# print(tab.tau.std(1) / np.sqrt(opt.n_try))

# all_stopping_times = np.array(tab.tau)
# np.savetxt(f"all_stopping_time_{algo_names[0]}.txt", all_stopping_times)

# n_pulls = np.array(n_pulls)
# print(n_pulls)
# print(n_pulls.mean(1))

# color_list = ['skyblue','g','r']

# plt.hist(
#     all_stopping_times, bins=2, color=color_list[0],
#     alpha=0.5, edgecolor=color_list[0], label="FC-DSH", lw=3)

# plt.savefig(f"fc_dsh_kwang.png", format='png')
# plt.show()

# print("")
# n_algo = len(algo_names)
# print("%20s %15s%15s%15s%15s" %('algo', 'avg', 'mse', 'misid', 'sreg'))
# for i in range(n_algo):
#     emax_mat[i,:] = sort(emax_mat[i,:])
#     Delta = opt.mu.max() - opt.mu
#     sreg = Delta[best_arm_mat[i,:].astype(int)].mean()
#     print("%20s:%15.4f%15.4f%15.4f%15.4f" % (algo_names[i], emax_mat[i,:].mean(), ((emax_mat[i,:] - 0.5)**2).mean(), np.mean(best_arm_mat[i,:] != 0), sreg))
# 
# print("")
# 
# print("wrong answers")
# for i in range(n_algo):
#     print(best_arm_mat[i,best_arm_mat[i,:] != 0])
# 
# import matplotlib.pyplot as plt
# plt.ion()
# 
# for i in range(n_algo):
#     for j in range(i + 1, n_algo):
#         plt.figure()
#         plt.plot([0,1], [0,1])
#         plt.plot(emax_mat[i,:], emax_mat[j,:])
#         plt.xlabel(algo_names[i])
#         plt.ylabel(algo_names[j])
