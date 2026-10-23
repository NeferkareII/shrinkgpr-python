"""Simulated data for Gaussian process regression (port of R/simGPR.R)."""

import numbers

import numpy as np
import pandas as pd
import torch
from torch.distributions import Gamma, MultivariateNormal

from ._base import _default_device
from .kernels import kernel_se
from .sim_mvgpr import _columns, _f64


def sim_gpr(N=200, d=3, d_mean=0, sigma2=0.1, tau=2, kernel_func=kernel_se, perc_spars=0.5,
            rho=0, theta=None, beta=None, device=None):
    """Simulate data from a Gaussian process regression model.

    The response is drawn as ``y ~ N(x_mean beta, K(x; theta, tau) + sigma2 I)``
    with standard normal covariates (equicorrelated with correlation ``rho``)
    and, for ``d_mean > 0``, standard normal mean covariates. All random
    numbers come from torch's generator, so ``torch.manual_seed`` makes the
    result reproducible.

    Parameters
    ----------
    N, d : int
        Number of observations and of covariates.
    d_mean : int
        Number of covariates of the linear mean; 0 for a zero mean.
    sigma2 : float
        Noise variance.
    tau : float
        Global shrinkage parameter; the kernel's output variance is ``1/tau``.
    kernel_func : callable
        Kernel function.
    perc_spars : float
        Share of the ``theta`` (and ``beta``) set to zero, in [0, 1].
    rho : float
        Correlation between the covariates, in [0, 1].
    theta : array-like of shape (d,), optional
        Inverse length scales; drawn from Gamma(6, rate 24) and sparsified if
        not given.
    beta : array-like of shape (d_mean,), optional
        Mean coefficients; standard normal and sparsified if not given.
    device : str or torch.device, optional
        Defaults to CUDA when available.

    Returns
    -------
    dict
        ``data``: a DataFrame with columns ``y``, ``x.1..x.d`` and
        ``x_mean.1..x_mean.d_mean`` (``x`` and ``x_mean`` for a single column,
        as in R); ``true_vals``: a dict with ``theta``, ``sigma2``, ``tau``
        and, for ``d_mean > 0``, ``beta``.
    """
    for name, v in {"N": N, "d": d}.items():
        if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 1:
            raise ValueError(f"The argument '{name}' must be a positive integer.")
    if isinstance(d_mean, bool) or not isinstance(d_mean, numbers.Integral) or d_mean < 0:
        raise ValueError("The argument 'd_mean' must be a non-negative integer.")
    for name, v in {"sigma2": sigma2, "tau": tau}.items():
        if not isinstance(v, numbers.Real) or not v > 0:
            raise ValueError(f"The argument '{name}' must be a positive numeric value.")
    if not callable(kernel_func):
        raise TypeError("The argument 'kernel_func' must be a valid function.")
    if not 0 <= perc_spars <= 1:
        raise ValueError("The argument 'perc_spars' must be a numeric value between 0 and 1.")
    if not 0 <= rho <= 1:
        raise ValueError("The argument 'rho' must be a numeric value between 0 and 1.")
    device = _default_device() if device is None else torch.device(device)

    # Drawn in R's order: beta, theta, covariates, mean covariates, response
    if beta is None:
        beta = torch.randn(d_mean, dtype=torch.float64).numpy()
        beta[torch.randperm(d_mean)[:round(d_mean * perc_spars)].numpy()] = 0.0
    else:
        beta = np.asarray(beta, dtype=float)
        if beta.shape != (d_mean,):
            raise ValueError("The argument 'beta', if provided, must be a numeric vector of "
                             "length 'd_mean'.")

    if theta is None:
        theta = Gamma(_f64(6), _f64(24)).sample((d,)).numpy()
        theta[torch.randperm(d)[:round(d * perc_spars)].numpy()] = 0.0
    else:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (d,):
            raise ValueError("The argument 'theta', if provided, must be a numeric vector of "
                             "length 'd'.")

    theta_t = torch.tensor(theta, dtype=torch.float32, device=device).unsqueeze(0)
    tau_t = torch.tensor([float(tau)], dtype=torch.float32, device=device)

    if rho > 0:
        Sigma = torch.full((d, d), float(rho), device=device)
        Sigma.fill_diagonal_(1.0)
        L = torch.linalg.cholesky(Sigma)
        x = torch.randn(N, d, device=device) @ L.t()
    else:
        x = torch.randn(N, d, device=device)

    K = kernel_func(theta_t, tau_t, x)[0] + sigma2 * torch.eye(N, device=device)
    if d_mean > 0:
        x_mean = torch.randn(N, d_mean, device=device)
        mu = x_mean @ torch.tensor(beta, dtype=torch.float32, device=device)
    else:
        mu = torch.zeros(N, device=device)
    y = MultivariateNormal(mu, K).sample()

    columns = [y.cpu().double().numpy()[:, None], x.cpu().double().numpy()]
    names = ["y"] + _columns("x", d)
    if d_mean > 0:
        columns.append(x_mean.cpu().double().numpy())
        names += _columns("x_mean", d_mean)
    data = pd.DataFrame(np.column_stack(columns), columns=names)

    true_vals = {"theta": theta, "sigma2": sigma2, "tau": tau}
    if d_mean > 0:
        true_vals["beta"] = beta
    return {"data": data, "true_vals": true_vals}
