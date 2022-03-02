import numpy as np
import torch
import time
from scipy import optimize
from scipy.stats import norm
import calcium_inference.probability_distributions as cipd
import calcium_inference.fourier as cif


def additive_calcium_inference_model(red_np, green_np, optimizer='BFGS', verbose=False):
    device = 'cpu'
    dtype = torch.float64

    # convert data to units of fold mean and subtract mean
    red_np = red_np / np.mean(red_np, axis=0) - 1
    green_np = green_np / np.mean(green_np, axis=0) - 1

    # convert to tensors and fourier transform
    red = torch.tensor(red_np, device=device, dtype=dtype)
    green = torch.tensor(green_np, device=device, dtype=dtype)
    red_fft = cif.real_fft(red)
    green_fft = cif.real_fft(green)

    # estimate all model parameters from the data
    variance_r_noise_init = np.var(red_np, axis=0)
    variance_g_noise_init = np.var(green_np, axis=0)
    variance_a_init = np.var(green_np, axis=0)
    variance_m_init = np.var(red_np, axis=0)

    # initialize length scale using the autocorrelation of the data
    length_scale_a_init = np.zeros(red_np.shape[1])
    length_scale_m_init = np.zeros(red_np.shape[1])

    for n in range(green_np.shape[1]):
        # approximate as the standard deviation of a gaussian fit to the autocorrelation function
        length_scale_m_init[n] = initialize_length_scale(red_np[:, n])
        length_scale_a_init[n] = initialize_length_scale(green_np[:, n])

    # preallocate space for all the training variables
    a_trained = np.zeros(red_np.shape)
    m_trained = np.zeros(red_np.shape)
    variance_r_noise_trained = np.zeros(variance_r_noise_init.shape)
    variance_g_noise_trained = np.zeros(variance_g_noise_init.shape)
    variance_a_trained = np.zeros(variance_a_init.shape)
    length_scale_a_trained = np.zeros(length_scale_a_init.shape)
    variance_m_trained = np.zeros(variance_m_init.shape)
    length_scale_m_trained = np.zeros(length_scale_m_init.shape)

    # loop through each neuron and perform inference
    start = time.time()
    for n in range(red_np.shape[1]):
        # get the initial values for the hyperparameters of this neuron
        # All hyperparameters are positive so we fit them in log space
        evidence_training_variables = np.log([variance_r_noise_init[n], variance_g_noise_init[n], variance_a_init[n],
                                              length_scale_a_init[n], variance_m_init[n], length_scale_m_init[n]])

        # define the evidence loss function. This function takes in and returns pytorch tensors
        def evidence_loss_fn(training_variables):
            return -cipd.gp_additive_fft_circ(red[:, n], red_fft[:, n], training_variables[0],
                                            green[:, n], green_fft[:, n], training_variables[1],
                                              training_variables[2], training_variables[3],
                                              training_variables[4], training_variables[5])[0]

        # a wrapper function of evidence that takes in and returns numpy variables
        def evidence_loss_fn_np(training_variables_in):
            training_variables = torch.tensor(training_variables_in, dtype=dtype, device=device)
            return evidence_loss_fn(training_variables).numpy()

        # wrapper function of for jacobian of the evidence that takes in and returns numpy variables
        def evidence_loss_jacobian_np(training_variables_in):
            training_variables = torch.tensor(training_variables_in, dtype=dtype, device=device, requires_grad=True)
            loss = evidence_loss_fn(training_variables)
            return torch.autograd.grad(loss, training_variables, create_graph=False)[0].numpy()

        # optimization function with jacobian from pytorch
        trained_variances = optimize.minimize(evidence_loss_fn_np, evidence_training_variables,
                                              jac=evidence_loss_jacobian_np,
                                              method=optimizer)

        # calculate the posterior values
        # The posterior is gaussian so we don't need to optimize, we find a and m in one step
        trained_variance_torch = torch.tensor(trained_variances.x, dtype=dtype, device=device)
        a, m = cipd.gp_additive_fft_circ(red[:, n], red_fft[:, n], trained_variance_torch[0], green[:, n], green_fft[:, n], trained_variance_torch[1],
                                         trained_variance_torch[2], trained_variance_torch[3], trained_variance_torch[4], trained_variance_torch[5],
                                         calculate_posterior=True)[1:]

        a_trained[:, n] = a.numpy()
        m_trained[:, n] = m.numpy()
        variance_r_noise_trained[n] = np.exp(trained_variances.x[0])
        variance_g_noise_trained[n] = np.exp(trained_variances.x[1])
        variance_a_trained[n] = np.exp(trained_variances.x[2])
        length_scale_a_trained[n] = np.exp(trained_variances.x[3])
        variance_m_trained[n] = np.exp(trained_variances.x[4])
        length_scale_m_trained[n] = np.exp(trained_variances.x[5])

        if verbose:
            decimals = 1e3
            # print out timing stuff
            elapsed = time.time() - start
            remaining = elapsed / (n + 1) * (red_np.shape[1] - (n + 1))
            elapsed_truncated = np.round(elapsed * decimals) / decimals
            remaining_truncated = np.round(remaining * decimals) / decimals
            print(str(n + 1) + '/' + str(red_np.shape[1]) + ' neurons complete')
            print(str(elapsed_truncated) + 's elapsed, estimated ' + str(remaining_truncated) + 's remaining')

    trained_variables = {'a': a_trained,
                         'm': m_trained,
                         'variance_r_noise': variance_r_noise_trained,
                         'variance_g_noise': variance_g_noise_trained,
                         'variance_a': variance_a_trained,
                         'length_scale_a': length_scale_a_trained,
                         'variance_m': variance_m_trained,
                         'length_scale_m': length_scale_m_trained,
                         }

    return trained_variables


def initialize_length_scale(y):
    x = np.arange(-len(y)/2, len(y)/2) + 0.5
    y_corr = np.correlate(y, y, mode='same')

    # fit the std of a gaussian to the correlation function

    def loss(p):
        return p[0] * norm.pdf(x, 0, p[1]) - y_corr

    p_init = np.array((np.max(y_corr), 1.0))
    s_hat = optimize.leastsq(loss, p_init)[0]

    # return the standard deviation
    return s_hat[1]

