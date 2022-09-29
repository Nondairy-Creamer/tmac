import numpy as np
import tmac.fourier as tfo
from scipy import stats, signal


def softplus(x, beta=50):
    return np.log(1 + np.exp(beta * x)) / beta


def generate_synthetic_data(num_ind, num_neurons, mean_r, mean_g, variance_noise_r, variance_noise_g,
                            variance_a, variance_m, tau_a, tau_m,
                            frac_nan=0.0, beta=20, multiplicative=False, rng_seed=None):
    """ Function that generates synthetic two channel imaging data

    Args:
        num_ind: number of measurements in time
        num_neurons: number of neurons recorded
        mean_r: mean fluoresence of the red channel
        mean_g: mean fluoresence of the green channel
        variance_noise_r: variance of the gaussian noise in the red channel
        variance_noise_g: variance of the gaussian noise in the green channel
        variance_a: variance of the calcium activity
        variance_m: variance of the motion artifact
        tau_a: timescale of the calcium activity
        tau_m: timescale of the motion artifact
        frac_nan: fraction of time points to set to NaN
        beta: parameter of softplus to keep values from going negative
        multiplicative: generate data assuming a multiplicative interaction between a and m
                        if false, generate data with additive interaction
    Returns:
        red_bleached: synthetic red channel data (motion + noise)
        green_bleached: synthetic green channel data (activity + motion + noise)
        a: activity Gaussian process
        m: motion artifact Gaussian process
    """
    rng = np.random.default_rng(rng_seed)

    fourier_basis, frequency_vec = tfo.get_fourier_basis(num_ind)

    # get the diagonal of radial basis kernel in fourier space
    c_diag_a = variance_a * tau_a * np.sqrt(2 * np.pi) * np.exp(-0.5 * frequency_vec**2 * tau_a**2)
    c_diag_m = variance_m * tau_m * np.sqrt(2 * np.pi) * np.exp(-0.5 * frequency_vec**2 * tau_m**2)

    a = fourier_basis @ (np.sqrt(c_diag_a[:, None]) * rng.standard_normal((num_ind, num_neurons)))
    m = fourier_basis @ (np.sqrt(c_diag_m[:, None]) * rng.standard_normal((num_ind, num_neurons)))

    # keep a and m from being negative for the multiplicative model
    # a has mean 1, m has mean 0
    a = softplus(a + 1, beta=beta)
    m = softplus(m + 1, beta=beta) - 1

    noise_r = np.sqrt(variance_noise_r) * rng.standard_normal((num_ind, num_neurons))
    noise_g = np.sqrt(variance_noise_g) * rng.standard_normal((num_ind, num_neurons))

    red_true = mean_r * softplus(m + 1 + noise_r, beta=beta)

    if multiplicative:
        green_true = mean_g * softplus(a * (m + 1) + noise_g, beta=beta)
    else:
        green_true = mean_g * softplus(a + m + noise_g, beta=beta)

    # add photobleaching
    photo_tau = num_ind / 3
    red_bleached = red_true * np.exp(-np.arange(num_ind)[:, None] / photo_tau)
    green_bleached = green_true * np.exp(-np.arange(num_ind)[:, None] / photo_tau)

    # nan a few values
    ind_to_nan = rng.random(num_ind) <= frac_nan
    red_bleached[ind_to_nan, :] = np.array('nan')
    green_bleached[ind_to_nan, :] = np.array('nan')

    return red_bleached, green_bleached, a, m


def col_corr(a_true, a_hat):
    """Calculate pearson correlation coefficient between each column of a_true and a_hat"""
    corr = np.zeros(a_true.shape[1])

    for c in range(a_true.shape[1]):
        true_vec = a_true[:, c] - np.mean(a_true[:, c])
        hat_vec = a_hat[:, c] - np.mean(a_hat[:, c])
        corr[c] = np.mean(true_vec * hat_vec) / np.std(true_vec) / np.std(hat_vec)

    return corr


def ratio_model(red, green, tau):
    # calculate the prediction from the ratio model
    # assumes red
    red = red / np.mean(red, axis=0)
    green = green / np.mean(green, axis=0)

    num_std = 3
    num_filter_ind = np.round(tau * num_std) * 2 + 1
    filter_x = np.arange(num_filter_ind) - (num_filter_ind - 1) / 2
    filter_shape = stats.norm.pdf(filter_x / tau) / tau
    green_filtered = signal.convolve2d(green, filter_shape[:, None], 'same')
    red_filtered = signal.convolve2d(red, filter_shape[:, None], 'same')
    ratio = green_filtered / red_filtered - 1

    return ratio
