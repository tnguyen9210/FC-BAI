
import numpy as np
import random 
import matplotlib.pyplot as plt

from empiricaldist import Cdf

colors = ['g','r', 'y', 'b', 'orange', 'purple']

np.random.seed(123)
random.seed(123)

def plot_data(samples, dist_name, idx):
    samples_sorted = np.sort(samples)
    samples_cdf = np.arange(1, len(samples_sorted) + 1) / len(samples_sorted)
    samples_tail = np.log(1 - samples_cdf)
    _samples_sorted = samples_sorted[samples_tail != -np.inf]
    _samples_cdf = samples_cdf[samples_tail != -np.inf]
    _samples_tail = samples_tail[samples_tail != -np.inf]

    plt.plot(_samples_sorted, _samples_cdf,
             label=f"{dist_name}", color=colors[idx])
    
    # plt.plot(_samples_sorted, _samples_tail,
    #          label=f"{dist_name}", color=colors[idx])
    
    # plt.hist(
    #     samples, bins=50,
    #     label=f"{dist_name}", lw=3, alpha=0.5,  
    #     color=colors[idx],
    #     edgecolor=colors[idx],
    # )

    # plt.legend()
    # plt.show()

n_trials = 100000
normal_samples = np.random.randn(n_trials)  # replace with your data array
# t_samples = np.random.standard_t(df=4, size=1000)

plot_data(normal_samples, 'normal', 4)
df_list = [5, 6]
for idx, df in enumerate(df_list):
    print(f'df = {df}')
    t_samples = np.random.standard_t(df=df, size=n_trials)
    plot_data(t_samples, f"student-t_df{df}", idx+1)

plt.legend()

plt.savefig(f"logcdf_plot_sep_student_t_{n_trials}.png", format='png')

plt.show()

# normal_sorted = np.sort(normal_samples)
# t_sorted = np.sort(t_samples)

# normal_cdf = np.arange(1, len(normal_sorted) + 1) / len(normal_sorted)
# normal_tail = np.log(1 - normal_cdf)
# _normal_sorted = normal_sorted[normal_tail != -np.inf]
# _normal_cdf = normal_cdf[normal_tail != -np.inf]
# _normal_tail = normal_tail[normal_tail != -np.inf]

# # plt.plot(normal_sorted, normal_cdf,
# #          label=f"normal", color=colors[0])
# plt.plot(normal_sorted, normal_tail,
#          label=f"normal", color=colors[0])

# normal_cdf = np.arange(1, len(normal_sorted) + 1) / len(normal_sorted)
# normal_tail = np.log(1 - normal_cdf)
# _normal_sorted = normal_sorted[normal_tail != -np.inf]
# _normal_cdf = normal_cdf[normal_tail != -np.inf]
# _normal_tail = normal_tail[normal_tail != -np.inf]

# plt.plot(normal_sorted, normal_cdf,
#          label=f"normal", color=colors[0])

# plt.show()

# Sort the samples
# sorted_samples = np.sort(samples)

# Compute the cumulative probabilities
# cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)

# # Plot the CDF
# plt.figure(figsize=(8, 5))
# plt.plot(sorted_samples, cdf, marker='.', linestyle='none')
# plt.xlabel('Value')
# plt.ylabel('CDF')
# plt.title('Empirical CDF')
# # plt.grid(True)
# plt.show()

