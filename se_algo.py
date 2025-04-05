
import numpy as np

def se_t4(K, arm_mus, sigma, delta, max_iter=100000):
    # K = n_arms
    # print(K)

    b_stopped = False 
    arm_idxes = [idx for idx in range(K)]
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


def se_orig(K, arm_mus, sigma, delta, max_iter=100000):
    # K = n_arms
    # print(K)

    b_stopped = False 
    arm_idxes = [idx for idx in range(K)]
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
