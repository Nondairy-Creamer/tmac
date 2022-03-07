import numpy as np
import calcium_inference.fourier as cif


def generate_synthetic_data(num_ind, num_neurons, mean_r, mean_g, variance_noise_r, variance_noise_g,
                            variance_a, variance_m, tau_a, tau_m, frac_nan=0.0):
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

    Returns:
        red_bleached: synthetic red channel data (motion + noise)
        green_bleached: synthetic green channel data (activity + motion + noise)
        a: activity Gaussian process
        m: motion artifact Gaussian process
    """
    fourier_basis, frequency_vec = cif.get_fourier_basis(num_ind)

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