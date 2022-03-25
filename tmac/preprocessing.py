import numpy as np
from scipy import interpolate, optimize
import torch


def interpolate_over_nans(input_mat, t=None):
    """ Function to interpolate over NaN values along the first dimension of a matrix

    Args:
        input_mat: numpy array, [time, neurons]
        t: optional time vector, only useful if input_mat is not sampled regularly in time

    Returns: Interpolated input_mat, interpolated time
    """

    # if t is not specified, assume it has been sampled at regular intervals
    if t is None:
        t = np.arange(input_mat.shape[0])

    output_mat = np.zeros(input_mat.shape)

    # calculate the average sample rate and uses this to create an interpolated t
    sample_rate = 1 / np.mean(np.diff(t, axis=0))
    t_interp = np.arange(input_mat.shape[0]) / sample_rate

    # loop through each column of the data and interpolate them separately
    for c in range(input_mat.shape[1]):
        # check if all the data is nan and skip if it is
        if np.all(np.isnan(input_mat[:, c])):
            print('column ' + str(c) + ' is all NaN, skipping')
            continue

        # find the location of all nan values
        no_nan_ind = ~np.isnan(input_mat[:, c])

        # remove nans from t and the data
        no_nan_t = t[no_nan_ind]
        no_nan_data_mat = input_mat[no_nan_ind, c]

        # interpolate values linearly
        interp_obj = interpolate.interp1d(no_nan_t, no_nan_data_mat, kind='linear', fill_value='extrapolate')
        output_mat[:, c] = interp_obj(t_interp)

    return output_mat, t_interp


def photobleach_correction(time_by_neurons, t=None):
    """ Function to fit an exponential with a shared tau to all the columns of time_by_neurons

    This function fits the function A*exp(-t / tau) to the matrix time_by_neurons. Tau is a single time constant shared
    between every column in time_by_neurons. A is an amplitude vector that is fit separately for each column. The
    correction is time_by_neurons / exp(-t / tau), preserving the amplitude of the data.

    Args:
        time_by_neurons: numpy array [time, neurons]
        t: optional, only important if time_by_neurons is not sampled evenly in time

    Returns: time_by_neurons divided by the exponential
    """

    if np.any(np.isnan(time_by_neurons)):
        raise Exception('Photobleach correction cannot be performed with NaNs in data')

    if t is None:
        t = np.arange(time_by_neurons.shape[0])
    device = 'cpu'
    dtype = torch.float64

    # convert inputs to tensors
    t_torch = torch.tensor(t, dtype=dtype, device=device)
    time_by_neurons_torch = torch.tensor(time_by_neurons, dtype=dtype, device=device)

    tau_0 = t[-1, None]/2
    a_0 = np.mean(time_by_neurons, axis=0)
    p_0 = np.concatenate((tau_0, a_0), axis=0)

    def loss_fn(p):
        exponential = p[None, 1:] * torch.exp(-t_torch[:, None] / p[0])
        return ((exponential - time_by_neurons_torch)**2).sum()

    def loss_fn_np(p_in):
        p = torch.tensor(p_in, dtype=dtype, device=device)
        return loss_fn(p).numpy()

    def loss_fn_jacobian_np(p_in):
        p = torch.tensor(p_in, dtype=dtype, device=device, requires_grad=True)
        loss = loss_fn(p)
        return torch.autograd.grad(loss, p, create_graph=False)[0].numpy()

    # optimization function with jacobian from pytorch
    p_hat = optimize.minimize(loss_fn_np, p_0,
                              jac=loss_fn_jacobian_np,
                              method='BFGS')

    time_by_neurons_corrected = time_by_neurons_torch / torch.exp(-t_torch[:, None] / p_hat.x[0])

    return time_by_neurons_corrected.numpy()
