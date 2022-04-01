from matplotlib import pyplot as plt
import numpy as np
import tmac.models as tm
import tmac.preprocessing as tp
from tmac.synthetic_data import generate_synthetic_data, col_corr, ratio_model


# set the parameters of the synthetic data
num_ind = 5000
num_neurons = 100
mean_r = 20
mean_g = 30
variance_noise_r_true = 0.005
variance_noise_g_true = 0.005
variance_a_true = 0.03
variance_m_true = 0.04
tau_a_true = 4
tau_m_true = 2
frac_nan = 0.05
beta = 200

# generate synthetic data
red_bleached, green_bleached, a_true, m_true = generate_synthetic_data(num_ind, num_neurons, mean_r, mean_g,
                                                                       variance_noise_r_true, variance_noise_g_true,
                                                                       variance_a_true, variance_m_true,
                                                                       tau_a_true, tau_m_true,
                                                                       frac_nan=frac_nan, beta=beta,
                                                                       multiplicative=False)
# interpolate out the nans in the data
red_interp = tp.interpolate_over_nans(red_bleached)[0]
green_interp = tp.interpolate_over_nans(green_bleached)[0]

# divide out the photobleaching
red = tp.photobleach_correction(red_interp)
green = tp.photobleach_correction(green_interp)

# infer the model parameters
trained_variables = tm.tmac_ac(red, green, verbose=True)

## Plotting ##
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
ratio = ratio_model(red, green, tau_a_true / 2)

# choose which neuron to plot and at what time indicies
plot_ind = 0
plot_start = 150
plot_time = 100

plt.figure()
green_fold_change = green / np.mean(green, axis=0) - 1
red_fold_change = red / np.mean(red, axis=0) - 1
# plot the green fluorescence
axes = plt.subplot(3, 1, 1)
plt.plot(green_fold_change[plot_start:plot_start+plot_time, plot_ind])
plt.plot(red_fold_change[plot_start:plot_start+plot_time, plot_ind])
plt.plot([0, plot_time], [0, 0])
lims = axes.get_ylim()
lim_to_use = np.max(np.abs(lims))
axes.set_ylim([-lim_to_use, lim_to_use])
plt.legend(['green'])

# plot the true activity against the inferred activity
axes = plt.subplot(3, 1, 2)
plt.plot(a_true[plot_start:plot_start+plot_time, plot_ind])
plt.plot(a_trained[plot_start:plot_start+plot_time, plot_ind])
plt.plot([0, plot_time], [0, 0])
lims = axes.get_ylim()
lim_to_use = np.max(np.abs(lims))
axes.set_ylim([-lim_to_use, lim_to_use])
plt.legend(['a_true', 'a_trained'])

# plot the motion artifact
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

# ratio vs TMAC performance
ratio_corelation_squared = col_corr(a_true, ratio)**2
tmac_corelation_squared = col_corr(a_true, a_trained) ** 2

plt.figure()
axes = plt.subplot(1, 2, 1)
plt.violinplot([ratio_corelation_squared, tmac_corelation_squared])
axes.set_ylim([0, 1])
plt.ylabel('correlation squared')
axes.set_xticks([1, 2])
axes.set_xticklabels(['ratio', 'inference'])
axes = plt.subplot(1, 2, 2)
plt.violinplot(tmac_corelation_squared - ratio_corelation_squared)
lims = np.array(axes.get_ylim())
lim_to_use = np.max(np.abs(lims))
axes.set_ylim([-lim_to_use, lim_to_use])
plt.plot([0.5, 1.5], [0, 0], '-k')
axes.set_xticks([1])
axes.set_xticklabels(['inference - ratio score'])
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
plt.scatter(np.arange(1, 5), [variance_a_true, variance_m_true, variance_noise_g_true, variance_noise_r_true], color='g', marker='o')
ylim = axes.get_ylim()
axes.set_ylim([0, ylim[1]])
axes.set_xticks(np.arange(4) + 1)
axes.set_xticklabels(['var_a', 'var_m', 'var_g', 'var_r'])
plt.ylabel('parameter value')
plt.legend(['inferred', 'true'])

axes = plt.subplot(1, 2, 2)
plt.violinplot([length_scale_a_trained, length_scale_m_trained])
plt.scatter(np.arange(1, 3), [tau_a_true, tau_m_true], color='g', marker='o')
ylim = axes.get_ylim()
axes.set_ylim([0, ylim[1]])
axes.set_xticks(np.arange(2) + 1)
axes.set_xticklabels(['tau_a', 'tau_m'])
plt.show()

