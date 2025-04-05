import time 
import numpy as np
import matplotlib.pyplot as plt


def run_trials(algo_name, algo, n_trials, n_arms, arm_mus, sigma, delta, max_iter, version):
    all_outputs = []
    all_stop_times = []
    start_time = time.time()
    for i_try in range(n_trials):
        output, stop_time = algo(n_arms, arm_mus, sigma, delta, max_iter)
        all_outputs.append(output)
        all_stop_times.append(stop_time)
        # print(stop_time, output)
        
        if (i_try == 0) or ((i_try + 1) % 100 == 0):
            print(f"trial {i_try}, stopping time = {stop_time}")
            total_time = time.time() - start_time
            print(f"it takes {total_time:0.4f}s")
            print(f"it takes {total_time/(i_try+1):0.4f}s per trial")
            np.savetxt(f"results/all_stopping_times_{algo_name}_{i_try+1}_{version}.txt", all_stop_times)

    np.savetxt(f"results/all_stopping_times_{algo_name}_{i_try+1}_{version}.txt", all_stop_times)
    
    return all_outputs, all_stop_times


def se_t4(n_arms, arm_mus, sigma, delta, max_iter=100000):
    K = n_arms
    # print(K)
    # arm_mus = [mu_best] + [mu_sub]*(K-1)

    b_stopped = False 
    arm_idxes = [idx for idx in range(n_arms)]
    arm_idxes = np.arange(K)
    # print(arm_idxes)
    arm_muhats = np.zeros(K)
    arm_nsamples = np.zeros(K)
    arm_sum_rewards = np.zeros(K)
    for t in range(max_iter):
        # select arm
        _nsamples = arm_nsamples[arm_idxes]
        _idx = np.argmin(_nsamples)
        a_idx_to_pull = arm_idxes[_idx]

        # sample and get reward
        _sample = np.random.normal(arm_mus[a_idx_to_pull], sigma)
        arm_sum_rewards[a_idx_to_pull] += _sample
        arm_nsamples[a_idx_to_pull] += 1

        if (t < K):
            continue

        # compute and remove the arms 
        arm_muhats[arm_idxes] = arm_sum_rewards[arm_idxes]/arm_nsamples[arm_idxes]
        thres = np.sqrt(2 * np.log(3.3 * K * (t ** 4) / delta) / arm_nsamples)
        max_muhat = np.max(arm_muhats)

        for idx, a_idx in enumerate(arm_idxes):
            if (max_muhat - arm_muhats[a_idx] >= thres[a_idx]):
                # Means.remove(Means[j])
                arm_muhats[a_idx] = float('-inf')
                arm_idxes = np.delete(arm_idxes, idx)
                break
        
        if len(arm_idxes) <= 1:
            b_stopped = True 
            break

        
    return arm_idxes[0], t  


def se_orig(n_arms, arm_mus, sigma, delta, max_iter=100000):
    K = n_arms
    # print(K)

    b_stopped = False 
    arm_idxes = [idx for idx in range(n_arms)]
    arm_idxes = np.arange(K)
    # print(arm_idxes)
    arm_muhats = np.zeros(K)
    arm_nsamples = np.zeros(K)
    arm_sum_rewards = np.zeros(K)
    for t in range(max_iter):
        # select arm
        _nsamples = arm_nsamples[arm_idxes]
        _idx = np.argmin(_nsamples)
        a_idx_to_pull = arm_idxes[_idx]

        # sample and get reward
        _sample = np.random.normal(arm_mus[a_idx_to_pull], sigma)
        arm_sum_rewards[a_idx_to_pull] += _sample
        arm_nsamples[a_idx_to_pull] += 1

        if (t < K):
            continue

        # compute and remove the arms 
        arm_muhats[arm_idxes] = arm_sum_rewards[arm_idxes]/arm_nsamples[arm_idxes]
        thres = np.sqrt(2 * np.log(3.3 * K * (t ** 2) / delta) / t)
        max_muhat = np.max(arm_muhats)

        for idx, a_idx in enumerate(arm_idxes):
            if (max_muhat - arm_muhats[a_idx] >= thres):
                # Means.remove(Means[j])
                arm_muhats[a_idx] = float('-inf')
                arm_idxes = np.delete(arm_idxes, idx)
                break
        
        if len(arm_idxes) <= 1:
            b_stopped = True 
            break

        
    return arm_idxes[0], t  


def create_algo(algo_name):
    if algo_name == 'se_orig':
        algo = se_orig
    else:
        algo = se_t4

    return algo 


def main():
    color_list = ['skyblue','g','r']
    np.random.seed(1)
    
    sigma = 1
    delta = 0.05
    max_iter = 1000000

    version = "v13"
    n_arms = 4
    mu_best = 1
    mu_sub = 1 - 0.2
    arm_mus = [mu_best] + [mu_sub]*(n_arms-1)
    
    # version = "v14"
    # n_arms = 16
    # mu_best = 1
    # mu_sub = 1 - 0.4
    # arm_mus = [mu_best] + [mu_sub]*(n_arms-1)
    
    n_trials = 1000

    print(f"n_arms = {n_arms}")
    print(f"arm_mus = {arm_mus}")
    print(f"n_trials = {n_trials}")
    print(f"max_iter = {max_iter}")

    algo_names = ["FC-DSH", "FC-DSH-reuse"]
    algo_names = ["se_t4", "se_orig"]
    
    for idx, algo_name in enumerate(algo_names):
        print(f"\n-> algo_name = {algo_name}")
        algo = create_algo(algo_name)
        # start_time = time.time()
        trials_outputs, trials_stop_times = \
            run_trials(algo_name, algo, n_trials, n_arms,
                       arm_mus, sigma, delta, max_iter, version)
        # total_time = time.time() - start_time
        # logging.info(f"it takes {total_time:0.2f}")

        # num_corrects = np.sum(np.array(trials_outputs) == 0)
        # logging.debug(trials_outputs)
        # logging.info(f"accuracy = {num_corrects/n_trials:0.4f}")
    
        # plt.hist(
        #     trials_stop_times, bins=10, color=color_list[idx],
        #     alpha=0.5, edgecolor=color_list[idx], label=algo_name, lw=3)
  
    # plt.xticks(fontsize=13)
    # plt.yticks(fontsize=13)

    # plt.xlabel('Stopping time', fontsize=13)
    # plt.ylabel('Number of Trials', fontsize=13)
    # # plt.title('Histogram of stopping times')
    # plt.legend(fontsize=15)
    # plt.savefig(f"fc_dsh_compare.png", format='png')
    # # plt.savefig(method + '.pdf', format='pdf')
    # plt.show()

if __name__ == "__main__":
    main()





