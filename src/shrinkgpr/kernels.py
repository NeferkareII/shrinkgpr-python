"""Kernel functions for Gaussian and t processes (port of R/kernel_funcs.R).

Each kernel takes ``thetas`` (n_latent x d), ``tau`` (n_latent), ``x`` (N x d)
and an optional ``x_star`` (N_new x d). It returns a batch of covariance
matrices of shape n_latent x N x N, or n_latent x N_new x N when ``x_star``
is given. No input checks are performed.
"""

import math

import torch


def sqdist(x, thetas, x_star=None):
    """Squared distances between the theta-scaled inputs."""
    scale = torch.sqrt(thetas.unsqueeze(1))
    x_scaled = x.unsqueeze(0) * scale
    if x_star is None:
        return torch.cdist(x_scaled, x_scaled, p=2.0).pow(2)
    x_star_scaled = x_star.unsqueeze(0) * scale
    return torch.cdist(x_star_scaled, x_scaled, p=2.0).pow(2)


def _inv_tau(tau):
    return 1.0 / tau.unsqueeze(1).unsqueeze(2)


def kernel_se(thetas, tau, x, x_star=None):
    """Squared exponential (RBF) kernel."""
    D = sqdist(x, thetas, x_star)
    return _inv_tau(tau) * torch.exp(-0.5 * D)


def kernel_matern_12(thetas, tau, x, x_star=None):
    """Matérn kernel with smoothness 1/2 (absolute exponential)."""
    D = torch.sqrt(sqdist(x, thetas, x_star) + 1e-4)
    return _inv_tau(tau) * torch.exp(-D)


def kernel_matern_32(thetas, tau, x, x_star=None):
    """Matérn kernel with smoothness 3/2."""
    D = torch.sqrt(sqdist(x, thetas, x_star) + 1e-4)
    sqrt3 = math.sqrt(3.0)
    return _inv_tau(tau) * (1 + sqrt3 * D) * torch.exp(-sqrt3 * D)


def kernel_matern_52(thetas, tau, x, x_star=None):
    """Matérn kernel with smoothness 5/2."""
    D = torch.sqrt(sqdist(x, thetas, x_star) + 1e-4)
    sqrt5 = math.sqrt(5.0)
    return _inv_tau(tau) * (1 + sqrt5 * D + (5.0 / 3.0) * D**2) * torch.exp(-sqrt5 * D)
