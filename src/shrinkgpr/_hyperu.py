"""Log of the confluent hypergeometric function U with a custom gradient (port of R/log_hyperu.R)."""

import math

import numpy as np
import torch
from scipy import special

EULER_GAMMA = 0.57721566490153286060651209008240243

# Below this threshold U is evaluated through its small-z expansion
SMALL_Z = 1e-9


def robust_hyperu(a, b, z):
    """Elementwise U(a, b, z) for scalar a, b and a tensor z > 0.

    Small z use the leading terms of the expansion around zero; everything
    else is evaluated by scipy on the CPU.
    """
    a = float(a)
    b = float(b)
    res = torch.full_like(z, float("nan"))
    small = z.abs() < SMALL_Z

    if small.any():
        zs = z[small]
        lg_a = math.lgamma(a)
        if b > 2:
            res[small] = torch.exp(math.lgamma(b - 1) - lg_a + (1 - b) * torch.log(zs))
        elif b == 2:
            res[small] = torch.exp(-lg_a - torch.log(zs))
        elif b > 1:
            # Gamma(1 - b) is negative here, so the constant term is negative
            res[small] = torch.exp(math.lgamma(b - 1) - lg_a + (1 - b) * torch.log(zs)) + (
                math.gamma(1 - b) / math.gamma(a - b + 1)
            )
        elif b == 1:
            res[small] = -math.exp(-lg_a) * (torch.log(zs) + special.digamma(a) + 2 * EULER_GAMMA)
        elif b == 0:
            res[small] = math.exp(-math.lgamma(a + 1))
        else:
            res[small] = math.exp(math.lgamma(1 - b) - math.lgamma(a - b + 1))

    rest = ~small
    if rest.any():
        z_np = z[rest].detach().cpu().numpy().astype(np.float64)
        if b == 0:
            # scipy is inaccurate for b = 0 and z < 1e-3; use Kummer's transformation
            # U(a, 0, z) = z U(a + 1, 2, z)
            u = z_np * special.hyperu(a + 1, 2.0, z_np)
        else:
            u = special.hyperu(a, b, z_np)
        res[rest] = torch.as_tensor(u, dtype=z.dtype, device=z.device)

    return res


class LogHyperU(torch.autograd.Function):
    """log U(a, b, x), differentiable in x only."""

    @staticmethod
    def forward(ctx, a, b, x):
        u = robust_hyperu(a, b, x)
        ctx.save_for_backward(a, b, x, u)
        return torch.log(u)

    @staticmethod
    def backward(ctx, grad_output):
        a, b, x, u = ctx.saved_tensors
        # dU/dz = -a U(a + 1, b + 1, z)
        grad_x = grad_output * (-a * robust_hyperu(a + 1, b + 1, x) / u)
        return None, None, grad_x


def log_hyperu(a, b, x):
    """log U(a, b, x) for scalar tensors a, b and a tensor x > 0."""
    return LogHyperU.apply(a, b, x)
