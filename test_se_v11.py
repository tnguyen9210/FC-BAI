import time 
import numpy as np
import matplotlib.pyplot as plt

no_non_stops = 0
cum_non_stops = 0

no_non_stops2 = 0
cum_non_stops2 = 0

sigma = 1
mu_best = 1
mu_sub = 0.900
n_trials = 1000
delta = 0.05
max_iter = 30000
all_stopping_times = np.zeros(n_trials)

np.random.seed(123456)
big_trials = 1

eps_list = [0.00]
color_list = ['skyblue','g','r']

esp_count = 0

experiment_config = "SE_Original_longrun"

K = 16

for eps in eps_list:
    for k in range(big_trials):
        no_non_stops = 0
        no_non_stops2 = 0
        start_time = time.time()
        for trial_idx in range(n_trials):
            switch = True
            arm_idxes = [0, 1, 2, 3]
            arm_mus = [mu_best] + [mu_sub]*(K-1)
            
            arm_muhats = np.zeros(K)
            arm_sum = np.zeros(K)
            arm_nsamples = np.zeros(K)
            # for j in range(3):
            #     arm_muhats[j] = np.random.normal(arm_mus[j], sigma)
                
            for t in range(1, max_iter):
                # sample each in S once and update muhats
                for a_idx in arm_idxes:
                    arm_sample = np.random.normal(arm_mus[a_idx], sigma)
                    # arm_muhats[a_idx] += (arm_sample - arm_muhats[a_idx])/t
                    arm_sum[a_idx] += arm_sample
                    arm_nsamples[a_idx] += 1
                    # arm_muhats[a_idx] = \
                    #     (arm_muhats[a_idx] * (t - 1) + arm_sample) / t

                arm_muhats[arm_idxes] = arm_sum[arm_idxes]/arm_nsamples[arm_idxes]
                
                total_nsamples = np.sum(arm_nsamples)
                thres = np.sqrt(2 * np.log(3.3 * K * (total_nsamples ** 4) / delta) / arm_nsamples)
                max_muhat = np.max(arm_muhats)
                for a_idx in arm_idxes:
                    # print(np.max(samples) - samples[j])
                    # print( np.sqrt(2* np.log(3.3*(t**2)/delta)/t))
                    # print("end")
                    if (max_muhat - arm_muhats[a_idx] + eps >= thres[a_idx]):
                        # Means.remove(Means[j])
                        arm_muhats[a_idx] = float('-inf')
                        arm_idxes.remove(a_idx)
                        break
                    
                if len(arm_idxes) <= 1:
                    break
                
                if (t > 5000 and switch == True):
                    no_non_stops2 = no_non_stops2 + 1
                    switch = False
                    
                if (t > 30000):
                    no_non_stops = no_non_stops + 1
                    break
                
            # print(t-1)
            total_nsamples = np.sum(arm_nsamples)
            all_stopping_times[trial_idx] = total_nsamples
            

        total_time = time.time() - start_time
        print(f"it takes {total_time}")
        cum_non_stops = cum_non_stops + no_non_stops
        cum_non_stops2 = cum_non_stops2 + no_non_stops2

    # print(all_stopping_times)
    plt.hist(all_stopping_times, bins=100, color=color_list[esp_count], alpha=0.3, edgecolor=color_list[esp_count], label='eps = ' + str(eps), lw=3)
    # plt.hist(all_stopping_times, bins=10000, color=color_list[esp_count], alpha=0.5, edgecolor=color_list[esp_count],lw=3)

    esp_count = esp_count + 1

np.savetxt('all_stopping_times_se_v11.txt', all_stopping_times)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

plt.xlabel('Stopping time', fontsize=13)
plt.ylabel('Number of Trials', fontsize=13)
# plt.title('Histogram of stopping times')
#plt.legend(fontsize=15)
plt.savefig(experiment_config + '.png', format='png')
plt.savefig(experiment_config + '.pdf', format='pdf')
plt.show()
print(cum_non_stops / big_trials)
print(cum_non_stops2 / big_trials)






