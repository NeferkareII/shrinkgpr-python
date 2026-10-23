r"""Kernel functions for Gaussian and t processes (port of R/kernel_funcs.R).

Each kernel takes ``thetas`` (n_latent x d), ``tau`` (n_latent), ``x`` (N x d)
and an optional ``x_star`` (N_new x d). It returns a batch of covariance
matrices of shape n_latent x N x N, or n_latent x N_new x N when ``x_star``
is given. No input checks are performed.

All kernels work on the theta-scaled distance
$r^2 = \sum_{j=1}^d \theta_j (x_j - x'_j)^2$, where $\theta_j \ge 0$ are
the inverse squared length scales. $\tau > 0$ is the global shrinkage
parameter of the prior on the $\theta_j$; the kernels' output variance is
$1/\tau$. The Matérn kernels use $D = \sqrt{r^2 + 10^{-4}}$, which keeps the gradient finite
at $r = 0$.

A custom kernel passed as ``kernel_func`` must follow the same signature,
return positive semi-definite matrices, and use torch operations only, so it
runs on the CPU and the GPU.
"""

import math

import torch


def sqdist(x, thetas, x_star=None):
    r"""Squared distances $r^2$ between the theta-scaled inputs.

    Note the argument order, ``(x, thetas, x_star)``, which differs from the
    kernels'. Returns n_latent x N x N, or n_latent x N_new x N when ``x_star``
    is given.
    """
    scale = torch.sqrt(thetas.unsqueeze(1))
    x_scaled = x.unsqueeze(0) * scale
    if x_star is None:
        return torch.cdist(x_scaled, x_scaled, p=2.0).pow(2)
    x_star_scaled = x_star.unsqueeze(0) * scale
    return torch.cdist(x_star_scaled, x_scaled, p=2.0).pow(2)


def _inv_tau(tau):
    return 1.0 / tau.unsqueeze(1).unsqueeze(2)


def kernel_se(thetas, tau, x, x_star=None):
    r"""Squared exponential (RBF) kernel, for smooth functions.

    $$k(x, x') = \frac{1}{\tau} \exp\left(-\frac{r^2}{2}\right)$$
    """
    D = sqdist(x, thetas, x_star)
    return _inv_tau(tau) * torch.exp(-0.5 * D)


def kernel_matern_12(thetas, tau, x, x_star=None):
    r"""Matérn kernel with smoothness 1/2 (absolute exponential).

    $$k(x, x') = \frac{1}{\tau} \exp(-D)$$
    """
    D = torch.sqrt(sqdist(x, thetas, x_star) + 1e-4)
    return _inv_tau(tau) * torch.exp(-D)


def kernel_matern_32(thetas, tau, x, x_star=None):
    r"""Matérn kernel with smoothness 3/2.

    $$k(x, x') = \frac{1}{\tau} \left(1 + \sqrt{3} D\right) \exp\left(-\sqrt{3} D\right)$$
    """
    D = torch.sqrt(sqdist(x, thetas, x_star) + 1e-4)
    sqrt3 = math.sqrt(3.0)
    return _inv_tau(tau) * (1 + sqrt3 * D) * torch.exp(-sqrt3 * D)


def kernel_matern_52(thetas, tau, x, x_star=None):
    r"""Matérn kernel with smoothness 5/2.

    $$k(x, x') = \frac{1}{\tau} \left(1 + \sqrt{5} D + \frac{5}{3} D^2\right)
    \exp\left(-\sqrt{5} D\right)$$
    """
    D = torch.sqrt(sqdist(x, thetas, x_star) + 1e-4)
    sqrt5 = math.sqrt(5.0)
    return _inv_tau(tau) * (1 + sqrt5 * D + (5.0 / 3.0) * D**2) * torch.exp(-sqrt5 * D)
