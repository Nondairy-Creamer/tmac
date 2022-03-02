from matplotlib import pyplot as plt
import numpy as np
from scipy import stats, signal
import calcium_inference.models as cim
import calcium_inference.fourier as cif
import calcium_inference.preprocessing as cip


def generate_example_data(num_ind, num_neurons, mean_r, mean_g, variance_noise_r, variance_noise_g,
                          variance_a, variance_m, tau_a, tau_m):
    fourier_basis, frequency_vec = cif.get_fourier_basis(num_ind)

    frac_nan = 0.05

    # get the diagonal of radial basis kernel in fourier space
    c_diag_a = variance_a * tau_a * np.sqrt(2 * np.pi) * np.exp(-0.5 * frequency_vec**2 * tau_a**2)
    c_diag_m = variance_m * tau_m * np.sqrt(2 * np.pi) * np.exp(-0.5 * frequency_vec**2 * tau_m**2)

    a = fourier_basis @ (np.sqrt(c_diag_a[:, None]) * np.random.randn(num_ind, num_neurons))
    m = fourier_basis @ (np.sqrt(c_diag_m[:, None]) * np.random.randn(num_ind, num_neurons))

    noise_r = np.sqrt(variance_noise_r) * np.random.randn(num_ind, num_neurons)
    noise_g = np.sqrt(variance_noise_g) * np.random.randn(num_ind, num_neurons)

    red_true = mean_r * (m + noise_r + 1)
    green_true = mean_g * ((a + 1) * (m + 1) + noise_g)

    # add photobleaching
    photo_tau = num_ind / 2
    red_bleached = red_true * np.exp(-np.arange(num_ind)[:, None] / photo_tau)
    green_bleached = green_true * np.exp(-np.arange(num_ind)[:, None] / photo_tau)

    # nan a few values
    ind_to_nan = np.random.rand(num_ind) <= frac_nan
    red_bleached[ind_to_nan, :] = np.array('nan')
    green_bleached[ind_to_nan, :] = np.array('nan')

    return red_bleached, green_bleached, a, m


def col_corr(a_true, a_hat):
    corr = np.zeros(a_true.shape[1])

    for c in range(a_true.shape[1]):
        true_vec = a_true[:, c]
        hat_vec = a_hat[:, c]
        corr[c] = np.mean(true_vec * hat_vec) / np.std(true_vec) / np.std(hat_vec)

    return corr


num_ind = 5000
num_neurons = 100
mean_r = 20
mean_g = 30
variance_noise_r_true = 0.2**2
variance_noise_g_true = 0.2**2
variance_a_true = 0.5**2
variance_m_true = 0.5**2
tau_a_true = 4
tau_m_true = 4

# generate synthetic data
red_bleached, green_bleached, a_true, m_true = generate_example_data(num_ind, num_neurons, mean_r, mean_g,
                                                                     variance_noise_r_true, variance_noise_g_true,
                                                                     variance_a_true, variance_m_true,
                                                                     tau_a_true, tau_m_true)

red_interp = cip.interpolate_over_nans(red_bleached)[0]
green_interp = cip.interpolate_over_nans(green_bleached)[0]

red = cip.photobleach_correction(red_interp)
green = cip.photobleach_correction(green_interp)

# infer the model parameters
trained_variables = cim.additive_calcium_inference_model(red, green, verbose=True)

## Plotting ##
# convert both channels into fold change from the mean and subtract off the mean
red = red / np.mean(red, axis=0) - 1
green = green / np.mean(green, axis=0) - 1

# pull out the trained variables
a_trained = trained_variables['a']
m_trained = trained_variables['m']
variance_r_noise_trained = trained_variables['variance_r_noise']
variance_g_noise_trained = trained_variables['variance_g_noise']
variance_a_trained = trained_variables['variance_a']
variance_m_trained = trained_variables['variance_m']
length_scale_a_trained = trained_variables['length_scale_a']
length_scale_m_trained = trained_variables['length_scale_m']

# calculate the prediction from the ratio model
num_std = 3
filter_std = tau_a_true
filter_x = np.arange(filter_std * num_std * 2) - filter_std * num_std
filter_shape = stats.norm.pdf(filter_x / filter_std) / filter_std
green_filtered = signal.convolve2d(green, filter_shape[:, None], 'same')
red_filtered = signal.convolve2d(red, filter_shape[:, None], 'same')
ratio = (green_filtered + 1) / (red_filtered + 1) - 1

# choose which neuron to plot
plot_ind = 2
plot_start = 150
plot_time = 100

plt.figure()
# plot the true activity against the inferred activity
axes = plt.subplot(3, 1, 1)
plt.plot(a_true[plot_start:plot_start+plot_time, plot_ind])
plt.plot(a_trained[plot_start:plot_start+plot_time, plot_ind])
plt.plot([0, plot_time], [0, 0])
lims = axes.get_ylim()
lim_to_use = np.max(np.abs(lims))
axes.set_ylim([-lim_to_use, lim_to_use])
plt.legend(['a_true', 'a_trained'])

# plot the true activity against the ratio model
axes = plt.subplot(3, 1, 2)
plt.plot(a_true[plot_start:plot_start+plot_time, plot_ind])
plt.plot(ratio[plot_start:plot_start+plot_time, plot_ind])
plt.plot([0, plot_time], [0, 0])
lims = axes.get_ylim()
lim_to_use = np.max(np.abs(lims))
axes.set_ylim([-lim_to_use, lim_to_use])
plt.legend(['a_true', 'ratio'])
plt.xlabel('time')
plt.ylabel('neural activity')

# plot the true motion artifact against the inferred motion artifact
axes = plt.subplot(3, 1, 3)
plt.plot(m_true[plot_start:plot_start+plot_time, plot_ind])
plt.plot(m_trained[plot_start:plot_start+plot_time, plot_ind])
plt.plot([0, plot_time], [0, 0])
lims = axes.get_ylim()
lim_to_use = np.max(np.abs(lims))
axes.set_ylim([-lim_to_use, lim_to_use])
plt.legend(['m_true', 'm_trained'])
plt.xlabel('time')
plt.ylabel('activity')
plt.show()

# ratio vs AIM performance
ratio_corelation_squared = col_corr(a_true, ratio) ** 2
a_corelation_squared = col_corr(a_true, a_trained) ** 2

plt.figure()
plt.violinplot([ratio_corelation_squared, a_corelation_squared])
axes = plt.gca()
axes.set_ylim([0, 1])
plt.ylabel('correlation squared')
axes.set_xticks([1, 2])
axes.set_xticklabels(['ratio', 'inference'])
plt.show()

# Plot scatter plot of true activity against inferred activity
# get y=x line
lower = np.min(a_true[:, plot_ind])
upper = np.max(a_true[:, plot_ind])
plt.figure()
plt.scatter(a_true[:, plot_ind], a_trained[:, plot_ind])
plt.plot([lower, upper], [lower, upper], 'r')
plt.xlabel('true value of a(t)')
plt.ylabel('inferred value of a(t)')
plt.show()

# Violin plot of hyperparameter fits over neurons
plt.figure()
axes = plt.subplot(1, 2, 1)
plt.violinplot([variance_a_trained, variance_m_trained, variance_g_noise_trained, variance_r_noise_trained])
plt.scatter(np.arange(1, 5), [variance_a_true, variance_m_true, variance_noise_g_true, variance_noise_r_true], color='g', marker='x')
ylim = axes.get_ylim()
axes.set_ylim([0, ylim[1]])

axes = plt.subplot(1, 2, 2)
plt.violinplot([length_scale_a_trained, length_scale_m_trained])
plt.scatter(np.arange(1, 3), [tau_a_true, tau_m_true], color='g', marker='x')
ylim = axes.get_ylim()
axes.set_ylim([0, ylim[1]])
plt.show()

