import torch
import numpy as np
import tmac.fourier as tfo


def tmac_evidence_and_posterior(r, fourier_r, log_variance_r_noise, g, fourier_g, log_variance_g_noise,
                                log_variance_a, log_tau_a, log_variance_m, log_tau_m,
                                threshold=1e8, calculate_posterior=False, truncate_freq=False):
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
        truncate_freq: boolean, if true truncates low amplitude frequencies in Fourier domain. This should give the same
            results but may give sensitivity to the initial conditions
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
    variance_noise_inv_sum = variance_r_noise_inv + variance_g_noise_inv

    device = r.device
    dtype = r.dtype

    # calculate the gaussian process components in fourier space
    # kernel = fourier_basis @ S @ fourier_basis.T
    # since S is diagonal, we just return the diagonal compnents for the activity (a) and motion (m)
    t_max = r.shape[0]

    all_freq = tfo.get_fourier_freq(t_max)
    all_freq = torch.tensor(all_freq, device=device, dtype=dtype)
    # smallest length scale (longest in fourier space)
    min_length = torch.min(length_scale_a.detach(), length_scale_m.detach())

    if truncate_freq:
        max_freq = 2*np.log(threshold) / min_length**2
        frequencies_to_keep = all_freq**2 < max_freq
    else:
        frequencies_to_keep = np.full(all_freq.shape, True)

    freq = all_freq[frequencies_to_keep]
    n_freq = len(freq)
    cutoff = torch.tensor(1 / threshold, device=device, dtype=dtype)

    # compute the diagonals of the covariances in fourier space
    covariance_a_fft = torch.maximum(torch.exp(-0.5 * freq ** 2 * length_scale_a ** 2), cutoff)
    covariance_a_fft = variance_a * (length_scale_a * np.sqrt(2 * np.pi)) * covariance_a_fft
    covariance_m_fft = torch.maximum(torch.exp(-0.5 * freq ** 2 * length_scale_m ** 2), cutoff)
    covariance_m_fft = variance_m * (length_scale_m * np.sqrt(2 * np.pi)) * covariance_m_fft

    D11 = 1 / covariance_m_fft + variance_noise_inv_sum
    D22 = 1 / covariance_a_fft + variance_g_noise_inv
    D12 = torch.ones(n_freq, device=device, dtype=dtype) * variance_g_noise_inv

    d_det = D11 * D22 - D12 ** 2

    D11_inv = (1 + D12 ** 2 / d_det) / D11
    D22_inv = D11 / d_det
    D12_inv = -D12 / d_det

    log_det_term = torch.log(d_det).sum() + torch.log(covariance_a_fft * covariance_m_fft).sum() + \
                   t_max * torch.log(variance_g_noise * variance_r_noise)
    #
    # alpha = covariance_m_fft + variance_r_noise
    # beta = covariance_m_fft
    # delta = covariance_m_fft + covariance_a_fft + variance_g_noise
    #
    # # log_det_term = torch.log(alpha * delta - beta ** 2).sum()
    # # log_det_term_2 = torch.log(delta).sum() + torch.log(alpha - beta**2 / delta).sum()

    # compute the quadratic term
    fourier_r_trimmed = fourier_r[frequencies_to_keep]
    fourier_g_trimmed = fourier_g[frequencies_to_keep]

    bxy1 = fourier_g_trimmed * variance_g_noise_inv + fourier_r_trimmed * variance_r_noise_inv
    bxy2 = fourier_g_trimmed * variance_g_noise_inv

    xy_bd_inv_bxy = (D11_inv * bxy1 ** 2).sum() + (D22_inv * bxy2 ** 2).sum() + 2 * (D12_inv * bxy1 * bxy2).sum()

    quad_term = (r ** 2).sum() * variance_r_noise_inv + (g ** 2).sum() * variance_g_noise_inv - xy_bd_inv_bxy

    obj = -log_det_term - quad_term

    if calculate_posterior:
        m_fft = D11_inv * bxy1 + D12_inv * bxy2
        a_fft = D22_inv * bxy2 + D12_inv * bxy1

        m_padded = torch.zeros_like(r, device=device, dtype=dtype)
        a_padded = torch.zeros_like(r, device=device, dtype=dtype)

        m_padded[frequencies_to_keep] = m_fft
        a_padded[frequencies_to_keep] = a_fft

        m_hat = tfo.real_ifft(m_padded)
        a_hat = tfo.real_ifft(a_padded)

        return a_hat, m_hat

    else:
        return torch.mean(obj)
