"""Sylvester normalizing flow (port of R/sylvester.R and the TorchScript ``sylvester_full``)."""

import torch
from torch import nn


class Sylvester(nn.Module):
    """Sylvester normalizing flow (van den Berg et al., 2018).

    Transforms ``z`` as ``z + Q R1 h(R2 Q^T z + b)`` with ``Q`` built from
    ``n_householder`` Householder reflections, upper triangular ``R1``, ``R2``
    and ``h = tanh``.

    Parameters
    ----------
    d : int
        Latent dimension.
    n_householder : int, optional
        Number of Householder reflections. Defaults to ``min(5, d - 1)``.
    """

    def __init__(self, d, n_householder=None):
        super().__init__()
        if n_householder is None:
            n_householder = min(5, d - 1)

        self.d = d
        self.n_householder = n_householder

        reg = 1 / d**0.5
        self.both_R = nn.Parameter(torch.zeros(d, d).uniform_(-reg, reg))
        self.diag1 = nn.Parameter(torch.zeros(d))
        self.diag2 = nn.Parameter(torch.zeros(d))
        self.Q = nn.Parameter(torch.zeros(n_householder, d).uniform_(-reg, reg))
        self.b = nn.Parameter(torch.zeros(d))

        self.register_buffer("triu_mask", torch.triu(torch.ones(d, d), diagonal=1))

    def _orthogonal_q(self):
        V = self.Q.t() / torch.norm(self.Q.t(), p=2, dim=0, keepdim=True).clamp_min(1e-8)
        eye = torch.eye(self.d, device=V.device, dtype=V.dtype)
        Q = eye
        for i in range(self.n_householder):
            v = V[:, i : i + 1]
            Q = Q @ (eye - 2 * v @ v.t())
        return Q

    def forward(self, z):
        diag1 = torch.tanh(self.diag1)
        diag2 = torch.tanh(self.diag2)
        r1 = self.both_R * self.triu_mask + torch.diag_embed(diag1)
        r2 = self.both_R.t() * self.triu_mask + torch.diag_embed(diag2)

        Q = self._orthogonal_q()
        rqzb = r2 @ (Q.t() @ z.t()) + self.b.unsqueeze(1)
        h_rqzb = torch.tanh(rqzb)
        zk = z + (Q @ (r1 @ h_rqzb)).t()

        # R2 R1 is upper triangular, so det J = prod(1 + h' r1_ii r2_ii)
        diag_j = (1.0 - h_rqzb.pow(2)).t() * (diag1 * diag2) + 1.0
        log_diag_j = torch.log(torch.abs(diag_j)).sum(dim=1)
        return zk, log_diag_j
