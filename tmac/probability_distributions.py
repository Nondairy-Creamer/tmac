import torch
import numpy as np
import tmac.fourier as tfo


def tmac_evidence_and_posterior(r, fourier_r, log_variance_r_noise, g, fourier_g, log_variance_g_noise,
                                log_variance_a, log_tau_a, log_variance_m, log_tau_m,
                                threshold=1e8, calculate_posterior=False):
    """ Two-channel motion artifact correction (TMAC) evidence and posterior distribution

    Args:
        r: red channel
        fourier_r: fourier transform of the red channel
        log_variance_r_noise: log of the variance of the red channel Gaussian noise
        g: green channel
        fourier_g: fourier transform of the green channel
        log_variance_g_noise: log of the variance of the green channel Gaussian noise
        log_variance_a: log of the variance of the activity
        log_tau_a: log of the timescale of the activity
        log_variance_m: log of the variance of the motion artifact
        log_tau_m: log of the timescale of the motion artifact
        threshold: maximum condition number of the radial basis function kernel for the Gaussian process
        calculate_posterior: boolean, whether to calculate the posterior

    if calculate_posterior:
        Returns: a_hat, m_hat
    else:
        Returns: log probability of the evidence
    """

    # exponentiate the log variances and invert the noise variances
    variance_r_noise = torch.exp(log_variance_r_noise)
    variance_g_noise = torch.exp(log_variance_g_noise)
    variance_a = torch.exp(log_variance_a)
    length_scale_a = torch.exp(log_tau_a)
    variance_m = torch.exp(log_variance_m)
    length_scale_m = torch.exp(log_tau_m)

    variance_r_noise_inv = 1 / variance_r_noise
    variance_g_noise_inv = 1 / variance_g_noise

    device = r.device
    dtype = r.dtype

    # calculate the gaussian process components in fourier space
    t_max = r.shape[0]

    all_freq = tfo.get_fourier_freq(t_max)
    all_freq = torch.tensor(all_freq, device=device, dtype=dtype)
    # smallest length scale (longest in fourier space)
    min_length = torch.min(length_scale_a.detach(), length_scale_m.detach())
    max_freq = 2*np.log(threshold) / min_length**2
    frequencies_to_keep = all_freq**2 < max_freq
    freq = all_freq[frequencies_to_keep]
    n_freq = len(freq)
    cutoff = torch.tensor(1 / threshold, device=device, dtype=dtype)

    # compute the diagonals of the covariances in fourier space
    covariance_a_fft = torch.maximum(torch.exp(-0.5 * freq**2 * length_scale_a**2), cutoff)
    covariance_a_fft = variance_a * (length_scale_a * np.sqrt(2 * np.pi)) * covariance_a_fft
    covariance_m_fft = torch.maximum(torch.exp(-0.5 * freq**2 * length_scale_m**2), cutoff)
    covariance_m_fft = variance_m * (length_scale_m * np.sqrt(2 * np.pi)) * covariance_m_fft

    f11 = 1 / covariance_a_fft + variance_g_noise_inv
    f22 = 1 / covariance_m_fft + variance_r_noise_inv + variance_g_noise_inv
    f12 = torch.tile(variance_g_noise_inv, f11.shape)

    f_det = f11 * f22 - f12**2

    k = f11 - f12**2 / f22

    f11_inv = 1 / k
    f22_inv = 1 / f22 + f12**2 / f22**2 / k
    f12_inv = -f12 / f22 / k

    log_det_term = torch.log(f_det).sum() + torch.log(covariance_a_fft * covariance_m_fft).sum() +\
                   t_max*torch.log(variance_g_noise*variance_r_noise)

    # compute the quadratic term
    fourier_r_trimmed = fourier_r[frequencies_to_keep]
    fourier_g_trimmed = fourier_g[frequencies_to_keep]

    auto_corr_term = (variance_r_noise_inv * r**2 + variance_g_noise_inv * g**2).sum()

    normalized_r_fft = variance_r_noise_inv * fourier_r_trimmed
    normalized_g_fft = variance_g_noise_inv * fourier_g_trimmed

    f_quad_mult_1 = normalized_g_fft
    f_quad_mult_2 = normalized_r_fft + normalized_g_fft

    f_quad = (f11_inv * f_quad_mult_1**2).sum() + (f22_inv * f_quad_mult_2**2).sum() + 2 * (f12_inv * f_quad_mult_1 * f_quad_mult_2).sum()

    quad_term = auto_corr_term - f_quad

    # define the prior over the hyperparameters which is a gamma distribution
    # we want this to be a low information prior with set parameters
    # treat it like its the distribution of sample variances for a low sampled gaussian variable with half the variance
    # of one of the fluorescent channels
    n = 5
    alpha = (n - 1) / 2
    beta_var_a = (n - 1) / (2 * torch.var(g))
    beta_var_m = (n - 1) / (2 * torch.var(r))
    beta_var_r = (n - 1) / (2 * torch.var(r))
    beta_var_g = (n - 1) / (2 * torch.var(g))

    log_gamma_var_a = (alpha - 1) * torch.log(variance_a) - beta_var_a * variance_a
    log_gamma_var_m = (alpha - 1) * torch.log(variance_m) - beta_var_m * variance_m
    log_gamma_var_r = (alpha - 1) * torch.log(variance_r_noise) - beta_var_r * variance_r_noise
    log_gamma_var_g = (alpha - 1) * torch.log(variance_g_noise) - beta_var_g * variance_g_noise

    hyperprior_term = log_gamma_var_a + log_gamma_var_m + log_gamma_var_r + log_gamma_var_g

    obj = -log_det_term - quad_term + hyperprior_term

    if calculate_posterior:
        a_fft = f11_inv * f_quad_mult_1 + f12_inv * f_quad_mult_2
        m_fft = f22_inv * f_quad_mult_2 + f12_inv * f_quad_mult_1

        m_padded = torch.zeros_like(r, device=device, dtype=dtype)
        a_padded = torch.zeros_like(r, device=device, dtype=dtype)

        m_padded[frequencies_to_keep] = m_fft
        a_padded[frequencies_to_keep] = a_fft

        m_hat = tfo.real_ifft(m_padded)
        a_hat = tfo.real_ifft(a_padded)

        return a_hat, m_hat

    else:
        return torch.mean(obj)

