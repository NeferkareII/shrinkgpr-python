"""Coupling-layer neural spline flow (port of R/NSF.R)."""

import torch
from torch import nn
from torch.nn import functional as F


def _coupling_net(d_in, hidden_dim, d_out):
    return nn.Sequential(
        nn.Linear(d_in, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, d_out),
    )


class NSFCL(nn.Module):
    """Neural spline flow with rational-quadratic spline couplings (Durkan et al., 2019).

    Each layer applies an invertible linear map ``W = P L U`` followed by two
    coupling steps, each transforming one half of the inputs conditioned on
    the other half.

    Parameters
    ----------
    d : int
        Latent dimension (at least 2).
    K : int
        Number of spline bins.
    B : float
        Tail bound; the spline acts on ``[-B, B]`` and is the identity outside.
    hidden_dim : int, optional
        Width of the coupling networks. Defaults to ``max(64, 2 * d)``.
    """

    def __init__(self, d, K=10, B=10, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(64, 2 * d)

        self.d = d
        self.K = K
        self.B = B
        self.n_params = 3 * K - 1
        self.d_lower = d // 2
        self.d_upper = d - self.d_lower

        # W = P L U with P a fixed permutation; at init L U = I
        P = torch.zeros(d, d)
        P[torch.arange(d), torch.randperm(d)] = 1
        self.register_buffer("P", P)
        self.L_raw = nn.Parameter(torch.zeros(d, d))
        self.U_raw = nn.Parameter(torch.zeros(d, d))
        self.log_diag_U = nn.Parameter(torch.zeros(d))

        self.f1 = _coupling_net(self.d_lower, hidden_dim, self.n_params * self.d_upper)
        self.f2 = _coupling_net(self.d_upper, hidden_dim, self.n_params * self.d_lower)

    def get_L(self):
        """Unit lower triangular factor ``L`` of the linear map ``W = P L U``."""
        eye = torch.eye(self.d, device=self.L_raw.device, dtype=self.L_raw.dtype)
        return torch.tril(self.L_raw, diagonal=-1) + eye

    def get_U(self):
        """Upper triangular factor ``U`` of ``W = P L U``, with a positive diagonal."""
        return torch.triu(self.U_raw, diagonal=1) + torch.diag(torch.exp(self.log_diag_U))

    def forward(self, z):
        batch_size = z.shape[0]

        # Invertible linear transform; log|det W| = sum(log diag U)
        W = self.P @ self.get_L() @ self.get_U()
        z = z @ W.t()
        log_diag_j = self.log_diag_U.sum().expand(batch_size)

        lower = z[:, : self.d_lower]
        upper = z[:, self.d_lower :]

        params1 = self.f1(lower).reshape(batch_size, self.d_upper, self.n_params)
        upper, ld1 = self._parse_and_transform(upper, params1)

        params2 = self.f2(upper).reshape(batch_size, self.d_lower, self.n_params)
        lower, ld2 = self._parse_and_transform(lower, params2)

        log_diag_j = log_diag_j + ld1.sum(dim=1) + ld2.sum(dim=1)
        return torch.cat([lower, upper], dim=1), log_diag_j

    def _parse_and_transform(self, x, params):
        W, H, D = torch.split(params, self.K, dim=-1)
        W = F.softmax(W, dim=-1) * 2 * self.B
        H = F.softmax(H, dim=-1) * 2 * self.B
        D = F.softplus(D)

        # Boundary derivatives fixed to 1 for identity tails
        ones = torch.ones(*x.shape, 1, device=x.device, dtype=x.dtype)
        D = torch.cat([ones, D, ones], dim=-1)
        return self._rqs_forward(x, W, H, D)

    def _rqs_forward(self, x, W, H, D):
        B = self.B
        inside = (x >= -B) & (x <= B)

        # Knot positions on [-B, B], K + 1 per dimension
        start = torch.full((*x.shape, 1), -B, device=x.device, dtype=x.dtype)
        cumW = torch.cat([start, torch.cumsum(W, dim=-1) - B], dim=-1)
        cumH = torch.cat([start, torch.cumsum(H, dim=-1) - B], dim=-1)

        x_safe = torch.clamp(x, min=-B, max=B)

        # Locate the bin via the K - 1 internal knots
        idx = torch.searchsorted(cumW[..., 1:-1].contiguous(), x_safe.unsqueeze(-1))
        idx = idx.clamp(min=0, max=self.K - 1)

        def at(t, i):
            return t.gather(-1, i).squeeze(-1)

        w_k, w_k1 = at(cumW, idx), at(cumW, idx + 1)
        h_k, h_k1 = at(cumH, idx), at(cumH, idx + 1)
        d_k, d_k1 = at(D, idx), at(D, idx + 1)

        delta_k = (h_k1 - h_k) / (w_k1 - w_k + 1e-8)
        xi = torch.clamp((x_safe - w_k) / (w_k1 - w_k + 1e-8), min=0, max=1)

        num = (h_k1 - h_k) * (delta_k * xi**2 + d_k * xi * (1 - xi))
        den = delta_k + (d_k + d_k1 - 2 * delta_k) * xi * (1 - xi)
        y_spline = h_k + num / (torch.abs(den) + 1e-8)

        dnum = delta_k**2 * (d_k1 * xi**2 + 2 * delta_k * xi * (1 - xi) + d_k * (1 - xi) ** 2)
        log_deriv = torch.log(torch.clamp(dnum, min=1e-8)) - 2 * torch.log(torch.abs(den) + 1e-8)

        # Spline inside [-B, B], identity in the tails
        y = torch.where(inside, y_spline, x)
        ld = torch.where(inside, log_deriv, torch.zeros_like(x))
        return y, ld
