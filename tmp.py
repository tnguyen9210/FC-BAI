
import numpy as np
import matplotlib.pyplot as plt

from empiricaldist import Cdf


# Example: generate or load your samples
samples = np.random.randn(1000)  # replace with your data array

cdf = Cdf.from_seq(samples)

cdf.plot(label='data')

plt.show()

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

