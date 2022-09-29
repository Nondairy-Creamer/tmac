import torch
from scipy import optimize


def scipy_minimize_with_grad(loss_fn_torch, variables_np, optimizer='BFGS', device='cpu', dtype=torch.float64):
    """ Minimizes a loss function using BFGS using scipy.optimize with a pytorch gradient

    This function takes advantage of the fact that scipy.optimize implements BFGS and chooses a learning rate and
    criteria so that the user doesn't need to. To speed this up, we provide the loss function written with pytorch.
    and send the Jacobian of the loss to the optimizer

    Args:
        loss_fn_torch: function which takes in torch version of variables_np and returns a scalar loss
        variables_np: numpy array, [number_of_variables,], initial values of the variables to perform descent on
        optimizer: string, scipy optimizer
        device: string, 'cpu' or 'cuda'. 'cuda' runs optimization on the GPU
        dtype: data type of the torch variables

    Returns: trained_variables: numpy array, [number_of_variables,], the optimized variables
    """

    # a wrapper function of evidence that takes in and returns numpy variables
    def loss_fn_np(variables_np_in):
        training_variables = torch.tensor(variables_np_in, device=device, dtype=dtype)
        return loss_fn_torch(training_variables).numpy()

    # wrapper function of for Jacobian of the evidence that takes in and returns numpy variables
    def loss_jacobian_np(variables_np_in):
        variables_torch = torch.tensor(variables_np_in, dtype=dtype, device=device, requires_grad=True)
        loss = loss_fn_torch(variables_torch)
        return torch.autograd.grad(loss, variables_torch, create_graph=False)[0].numpy()

    # optimization function with Jacobian from pytorch
    trained_variables = optimize.minimize(loss_fn_np, variables_np,
                                          jac=loss_jacobian_np,
                                          method=optimizer)

    return trained_variables