"""Internal numerical helpers (port of R/internal_utils.R)."""

import warnings

import numpy as np
import torch
from scipy import stats


def lightweight_ols(y, x):
    """Two-sided p-value of the slope in a simple linear regression of y on x."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = y.size
    ym = y.mean()
    xm = x.mean()

    sxx = np.sum((x - xm) ** 2)
    beta_hat = np.sum((x - xm) * (y - ym)) / sxx
    alpha_hat = ym - beta_hat * xm

    resids = y - alpha_hat - beta_hat * x
    sigma2_hat = np.sum(resids**2) / (n - 2)
    beta_hat_var = sigma2_hat / sxx

    return 2 * stats.t.cdf(-abs(beta_hat / np.sqrt(beta_hat_var)), df=n - 2)


def robust_chol(A, tol=1e-6, upper=False):
    """Cholesky factor of a batch of symmetric matrices (B x N x N).

    Falls back on increasing diagonal jitter and then an eigenvalue shift when
    the plain decomposition fails.
    """
    A = 0.5 * (A + A.transpose(1, 2))
    B, N, _ = A.shape

    def result(L):
        return L.transpose(1, 2) if upper else L

    # Fast path for well-conditioned matrices
    L, info = torch.linalg.cholesky_ex(A)
    if not info.any():
        return result(L)

    # Quick jitter: 3 attempts with aggressive jumps
    eye = torch.eye(N, device=A.device, dtype=A.dtype).expand(B, N, N)
    diag_mean = torch.diagonal(A, dim1=1, dim2=2).mean(dim=1)
    jitter = tol

    for _ in range(3):
        L, info = torch.linalg.cholesky_ex(A + eye * (jitter * diag_mean).view(-1, 1, 1))
        if not info.any():
            return result(L)
        jitter *= 100

    # Eigenvalue fallback with tight safety margin
    with torch.no_grad():
        min_e = torch.linalg.eigvalsh(A).min(dim=1).values
        shift = torch.clamp(-min_e + tol, min=0.0)
    shift = 1.1 * shift + tol
    L, info = torch.linalg.cholesky_ex(A + eye * shift.view(-1, 1, 1))

    # Float32 eigenvalues may be imprecise, one more aggressive attempt
    if info.any():
        shift = 2.0 * shift + 1e-3
        L, info = torch.linalg.cholesky_ex(A + eye * shift.view(-1, 1, 1))

    if info.any():
        raise RuntimeError("Cholesky decomposition failed")
    return result(L)


def res_protector(x, tol=1e-6):
    """Push values away from zero (from either side) to at least ``tol`` in magnitude."""
    s = 2 * (x >= 0).to(dtype=x.dtype) - 1
    return s * torch.clamp(torch.abs(x), min=tol)


def merge_args(defaults, user, name):
    """Merge user-supplied keyword arguments into defaults.

    Unknown names are ignored with a warning, as in the R package.
    """
    if user is None:
        return dict(defaults)
    if not isinstance(user, dict):
        raise TypeError(f"'{name}' has to be a dict")

    unknown = [k for k in user if k not in defaults]
    if unknown:
        verb = "has" if len(unknown) == 1 else "have"
        warnings.warn(
            f"{', '.join(unknown)} {verb} been incorrectly named in '{name}' and will be ignored",
            stacklevel=3,
        )

    return {k: user.get(k, v) for k, v in defaults.items()}
