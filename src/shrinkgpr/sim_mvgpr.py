"""Simulated data for multivariate Gaussian process regression (port of R/simMVGPR.R)."""

import numbers

import numpy as np
import pandas as pd
import torch
from torch.distributions import Beta, Gamma

from ._base import _default_device
from .kernels import kernel_se


def _f64(v):
    return torch.tensor(float(v), dtype=torch.float64)


def _columns(prefix, k):
    """Column names of a k-column matrix in an R data.frame: ``x``, or ``x.1..x.k``."""
    return [prefix] if k == 1 else [f"{prefix}.{j + 1}" for j in range(k)]


def rlkjcorr(M, eta=1.0):
    """One M x M correlation matrix from the LKJ(eta) distribution (onion method).

    Port of the internal ``rlkjcorr`` of the R package, drawing from torch's
    random number generator.
    """
    if M < 2:
        raise ValueError("'M' must be at least 2")

    def rbeta(a, b):
        return Beta(_f64(a), _f64(b)).sample().item()

    alpha = eta + (M - 2) / 2
    R = np.zeros((M, M))  # upper triangular Cholesky factor
    r12 = 2 * rbeta(alpha, alpha) - 1
    R[0, 0] = 1.0
    R[0, 1] = r12
    R[1, 1] = np.sqrt(1 - r12**2)
    for m in range(2, M):
        alpha -= 0.5
        y = rbeta(m / 2, alpha)
        z = torch.randn(m, dtype=torch.float64).numpy()
        z = z / np.sqrt(np.sum(z**2))
        R[:m, m] = z * np.sqrt(y)
        R[m, m] = np.sqrt(1 - y)
    return R.T @ R


def sim_mvgpr(N=200, M=2, d=3, sigma2=0.1, tau=2, kernel_func=kernel_se, perc_spars=0.5,
              rho=0, theta=None, Omega=None, device=None):
    """Simulate data from a multivariate Gaussian process regression model.

    The response is drawn as ``Y ~ MN(0, K(x; theta, tau) + sigma2 I, Omega)``,
    i.e. ``vec(Y) ~ N(0, Omega kron (K + sigma2 I))``, with standard normal
    covariates (equicorrelated with correlation ``rho``). All random numbers
    come from torch's generator, so ``torch.manual_seed`` makes the result
    reproducible.

    Parameters
    ----------
    N, M, d : int
        Number of observations, responses and covariates.
    sigma2 : float
        Noise variance.
    tau : float
        Global shrinkage parameter; the kernel's output variance is ``1/tau``.
    kernel_func : callable
        Kernel function.
    perc_spars : float
        Share of the ``theta`` set to zero, in [0, 1].
    rho : float
        Correlation between the covariates, in [0, 1].
    theta : array-like of shape (d,), optional
        Inverse length scales; drawn from Gamma(6, rate 24) and sparsified if
        not given.
    Omega : array-like of shape (M, M), optional
        Symmetric positive definite between-response covariance; ``S D S``
        with ``D ~ LKJ(1)`` and ``diag(S) ~ Gamma(1, 1)`` if not given.
    device : str or torch.device, optional
        Defaults to CUDA when available.

    Returns
    -------
    dict
        ``data``: a DataFrame with columns ``y.1..y.M`` and ``x.1..x.d``
        (``x`` for ``d = 1``, as in R);
        ``true_vals``: a dict with ``theta``, ``Omega``, ``sigma2`` and ``tau``.
    """
    for name, v in {"N": N, "M": M, "d": d}.items():
        if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 1:
            raise ValueError(f"The argument '{name}' must be a positive integer.")
    if M < 2:
        raise ValueError("The argument 'M' must be at least 2.")
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

    if theta is None:
        theta = Gamma(_f64(6), _f64(24)).sample((d,)).numpy()
        n_zero = round(d * perc_spars)
        theta[torch.randperm(d)[:n_zero].numpy()] = 0.0
    else:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (d,):
            raise ValueError("The argument 'theta', if provided, must be a numeric vector of "
                             "length 'd'.")

    if Omega is None:
        D = rlkjcorr(M, 1.0)
        S = np.diag(Gamma(_f64(1), _f64(1)).sample((M,)).numpy())
        Omega = S @ D @ S
    else:
        Omega = np.asarray(Omega, dtype=float)
        if Omega.shape != (M, M):
            raise ValueError("The argument 'Omega', if provided, must be a numeric symmetric "
                             "positive definite matrix of size M x M.")
        if not np.allclose(Omega, Omega.T) or np.any(np.linalg.eigvalsh(Omega) <= 0):
            raise ValueError("The argument 'Omega', if provided, must be a symmetric positive "
                             "definite matrix.")

    theta_t = torch.tensor(theta, dtype=torch.float32, device=device).unsqueeze(0)
    tau_t = torch.tensor([float(tau)], dtype=torch.float32, device=device)
    Omega_t = torch.tensor(Omega, dtype=torch.float32, device=device)

    if rho > 0:
        Sigma = torch.full((d, d), float(rho), device=device)
        Sigma.fill_diagonal_(1.0)
        L = torch.linalg.cholesky(Sigma)
        x = torch.randn(N, d, device=device) @ L.t()
    else:
        x = torch.randn(N, d, device=device)

    K = kernel_func(theta_t, tau_t, x)[0]
    L_S = torch.linalg.cholesky(K + sigma2 * torch.eye(N, device=device))
    L_Om = torch.linalg.cholesky(Omega_t)
    Z = torch.randn(N, M, device=device)
    y = L_S @ Z @ L_Om.t()

    y_np, x_np = y.cpu().double().numpy(), x.cpu().double().numpy()
    data = pd.DataFrame(np.column_stack([y_np, x_np]),
                        columns=_columns("y", M) + _columns("x", d))
    return {"data": data,
            "true_vals": {"theta": theta, "Omega": Omega, "sigma2": sigma2, "tau": tau}}
