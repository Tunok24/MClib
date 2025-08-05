import numpy as np
import matplotlib.pyplot as plt

# Define f(x) and g(x)
def f_x(x):
    return np.sqrt(x) * np.exp(-x)

def g_x(x):
    return np.exp(-x)  # Not normalized


# Rejection sampling to get x ~ g(x)
def rejection_sample_g(n_samples, x_min, x_max):
    accepted = []
    attempts = 0
    # M = max of g(x) over [x_min, x_max] → occurs at x = x_min
    M = g_x(x_min)
    while len(accepted) < n_samples:
        x = np.random.uniform(x_min, x_max)
        u = np.random.uniform(0, 1)
        if u < g_x(x) / M:  # Use unnormalized g(x)
            accepted.append(x)
        attempts += 1
    acceptance_rate = n_samples / attempts
    return np.array(accepted), acceptance_rate


# Main Monte Carlo integration with importance sampling
def monte_carlo_integrate_with_rejection(n_samples, x_min, x_max):
    x_samples, acc_rate = rejection_sample_g(n_samples, x_min, x_max)
    weights = f_x(x_samples) / g_x(x_samples)  # Unnormalized g(x) — totally fine!
    estimate = np.mean(weights)
    std_dev = np.std(weights) / np.sqrt(n_samples)
    return estimate, std_dev, acc_rate

# Run it!
n_samples = 2000000
x_min = 0.0001
x_max = 10.0
integral_estimate, uncertainty, acc_rate = monte_carlo_integrate_with_rejection(n_samples, x_min, x_max)

print(f"Estimated integral: {integral_estimate:.6f} ± {uncertainty:.6f}")
print(f"Rejection sampling acceptance rate: {acc_rate:.2f}")



# # Parameters
# max_samples = n_samples
# step = 2000
# sample_sizes = np.arange(step, max_samples + 1, step)

# estimates = []
# uncertainties = []
# acc_rates = []

# for n in sample_sizes:
#     estimate, std_err, acc = monte_carlo_integrate_with_rejection(n)
#     estimates.append(estimate)
#     uncertainties.append(std_err)
#     acc_rates.append(acc)

# # Plotting
# fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# axs[0].plot(sample_sizes, estimates, label='Estimate')
# axs[0].set_ylabel("Integral Estimate")
# axs[0].grid(True)
# axs[0].legend()

# axs[1].plot(sample_sizes, uncertainties, label='Uncertainty (Std. Error)', color='orange')
# axs[1].set_ylabel("Uncertainty")
# axs[1].grid(True)
# axs[1].legend()

# plt.suptitle("Monte Carlo Statistics vs. Number of Samples")
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig("./fig/mc_statistics_vs_samples.png", dpi=300)
# plt.close()
