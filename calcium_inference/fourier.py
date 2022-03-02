import numpy as np
import torch


def get_fourier_freq(t_max):
    """Returns the frequencies for a vector of length t_max"""
    n_cos = np.ceil((t_max + 1) / 2)  # number of cosine terms (positive freqs)
    n_sin = np.floor((t_max - 1) / 2)  # number of sine terms (negative freqs)
    w_cos = np.arange(0, n_cos)  # cosine freqs
    w_sin = np.arange(-n_sin, 0)  # sine freqs
    w_vec = np.concatenate((w_cos, w_sin), axis=0)
    frequencies = 2 * np.pi / t_max * w_vec

    return frequencies


def real_fft(x, n=None):
    """Performs a real discrete 1D Fourier transform of the columns of x"""

    single_vec = False
    if len(x.shape) == 1:
        single_vec = True
        x = x[:, None]

    if n is None:
        n = x.shape[0]

    x_fft = torch.fft.fft(x, n, dim=0) / np.sqrt(n / 2)
    x_fft[0, :] = x_fft[0, :] / np.sqrt(2)

    if np.mod(n, 2) == 0:
        imx = int(np.ceil((n - 1) / 2))
        x_fft[imx, :] = x_fft[imx, :] / np.sqrt(2)

    x_hat = x_fft.real
    isin = int(np.ceil((n + 1) / 2))
    x_hat[isin:, :] = -x_fft[isin:, :].imag

    if single_vec:
        x_hat = x_hat[:, 0]

    return x_hat


def real_ifft(x_hat, n=None):
    """Performs an inverse of a real discrete 1D Fourier transform of the columns of x"""

    single_vec = False
    if len(x_hat.shape) == 1:
        single_vec = True
        x_hat = x_hat[:, None]

    if n is None:
        n = x_hat.shape[0]

    nxh = x_hat.shape[0]
    n_cos = int(np.ceil((nxh + 1) / 2))
    n_sin = int(np.floor((nxh - 1) / 2))

    x_hat[0, :] = x_hat[0, :] * np.sqrt(2)

    if np.mod(nxh, 2) == 0:
        x_hat[n_cos-1, :] = x_hat[n_cos-1, :] * np.sqrt(2)

    xfft = torch.zeros_like(x_hat, dtype=torch.complex128, device=x_hat.device)
    xfft[:] = x_hat[:]
    xfft[n_cos:, :] = torch.flip(x_hat[1:n_sin+1, :], dims=[0])
    xfft[1:n_sin+1, :] = xfft[1:n_sin+1, :] + 1j * torch.flip(x_hat[n_cos:, :], dims=[0])
    xfft[n_cos:] = xfft[n_cos:, :] - 1j * x_hat[n_cos:, :]

    x = torch.fft.ifft(xfft, dim=0).real * np.sqrt(nxh / 2)
    x = x[:n, :]

    if single_vec:
        x = x[:, 0]

    return x


def get_fourier_basis(n_ind):
    """Returns an orthonormal real Fourier basis for a vector of length n_ind"""

    n_cos = (np.ceil((n_ind + 1) / 2))
    n_sin = (np.floor((n_ind - 1) / 2))

    cos_freq = 2 * np.pi / n_ind * np.arange(n_cos)
    sin_freq = 2 * np.pi / n_ind * np.arange(-n_sin, 0)
    frequency_vec = np.concatenate((cos_freq, sin_freq), axis=0)  # frequency vector

    x = np.arange(n_ind)
    cos_basis = np.cos(cos_freq[:, None] * x[None, :]) / np.sqrt(n_ind / 2)
    sin_basis = np.sin(sin_freq[:, None] * x[None, :]) / np.sqrt(n_ind / 2)
    fourier_basis = np.concatenate((cos_basis, sin_basis), axis=0)

    fourier_basis[0, :] = fourier_basis[0, :] / np.sqrt(2)

    if np.mod(n_ind, 2) == 0:
        fourier_basis[int(n_cos-1), :] = fourier_basis[int(n_cos-1), :] / np.sqrt(2)

    return fourier_basis, frequency_vec
