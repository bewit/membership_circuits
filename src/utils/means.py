import torch
from scipy.stats import hmean as scipy_hmean


def generalized_mean(log_x: torch.Tensor, log_weights: torch.Tensor, p: torch.Tensor, dim: int, complex: bool = False) -> torch.Tensor:
    if p.dim() == log_x.dim()-1:
        xp = p.unsqueeze(2) * log_x
    else:
        xp = p * log_x # this is equal to log(x^p)
    if complex:
        wxp = complex_logsumexp(xp + log_weights, dim=dim) - complex_logsumexp(log_weights, dim=dim)
    else:
        wxp = torch.logsumexp(xp + log_weights, dim=dim) - torch.logsumexp(log_weights, dim=dim)
    gen_mean = wxp / p
    return gen_mean


def complex_logsumexp(input: torch.Tensor, dim: int = None, keepdim: bool = False) -> torch.Tensor:
    """ https://scicomp.stackexchange.com/questions/34273/log-sum-exp-trick-for-signed-complex-numbers
    """
    magnitude_real = torch.amax(input.real, dim=dim, keepdims=True) # magnitude w.r.t. to the real part is sufficient, as the magnitude of the imaginary part is constant and does not influence the result

    output = torch.log(torch.sum(torch.exp(input - magnitude_real), dim=dim, keepdims=keepdim))

    if not keepdim:
        magnitude_real = torch.squeeze(magnitude_real, dim=dim)
        
    output = output + magnitude_real

    return output


def harmonic_mean_2d(data: torch.Tensor, weights: torch.Tensor = None, dim: int = -1):
    """ Harmonic Mean for matrices (2-dim tensors)
    """
    if weights is None:
        weight_shape = (1, data.shape[1])
        weights = torch.ones(weight_shape) / data.shape[1]
    else: 
        weights = weights.view(1, -1)

    assert torch.isclose(torch.sum(weights), torch.tensor(1.0)), weights

    # hm = torch.sum(weights / data, dim=dim, keepdim=True) ** (-1)
    hmean = torch.tensor(scipy_hmean(data.detach().cpu().numpy(), axis=dim, weights=weights.detach().cpu().numpy())).view(-1, 1)

    return hmean