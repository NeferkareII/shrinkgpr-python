"""Inverse autoregressive flow with a MADE network (port of R/IAF.R)."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class MaskedLinear(nn.Module):
    """Linear layer whose weight is multiplied elementwise by a fixed binary mask."""

    def __init__(self, in_features, out_features, mask, zero_init=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        if zero_init:
            nn.init.zeros_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


def _hidden_degrees(H, d):
    # Cycle through 1..d-1 so that every representable degree appears
    if d <= 1:
        return torch.ones(H, dtype=torch.long)
    return torch.arange(H) % (d - 1) + 1


class IAF(nn.Module):
    """Inverse autoregressive flow (Kingma et al., 2016) with an LSTM-style update.

    A MADE network maps ``z`` to a shift ``m`` and gate pre-activation ``s``;
    the update is ``sigma * z + (1 - sigma) * m`` with
    ``sigma = sigmoid(s + forget_bias)``.

    Parameters
    ----------
    d : int
        Latent dimension.
    hidden_dim : int, optional
        Width of the hidden layers. Defaults to ``max(64, 2 * d)``.
    n_hidden : int
        Number of hidden layers.
    forget_bias : float
        Added to the gate pre-activation; larger values start closer to the identity.
    perm : sequence of int, optional
        Autoregressive ordering, a permutation of ``range(d)`` (0-based). Drawn
        from torch's RNG if not given.
    """

    def __init__(self, d, hidden_dim=None, n_hidden=1, forget_bias=2, perm=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(64, 2 * d)

        self.d = d
        self.forget_bias = forget_bias

        perm = torch.randperm(d) if perm is None else torch.as_tensor(perm, dtype=torch.long)
        if sorted(perm.tolist()) != list(range(d)):
            raise ValueError("'perm' has to be a permutation of range(d)")
        self.register_buffer("perm", perm)

        # Degrees (1-based as in MADE): inputs and outputs share the ordering
        deg_io = perm + 1
        deg_hidden = [_hidden_degrees(hidden_dim, d) for _ in range(n_hidden)]

        # Keep a connection when the later degree is >= the earlier one
        m_in = (deg_hidden[0].unsqueeze(1) >= deg_io.unsqueeze(0)).float()
        layers = [MaskedLinear(d, hidden_dim, m_in)]
        for l in range(1, n_hidden):
            m_hh = (deg_hidden[l].unsqueeze(1) >= deg_hidden[l - 1].unsqueeze(0)).float()
            layers.append(MaskedLinear(hidden_dim, hidden_dim, m_hh))
        self.hidden_layers = nn.ModuleList(layers)

        # Strict inequality so no output depends on its own input; shift and
        # gate blocks share a mask
        m_out = (deg_io.unsqueeze(1) > deg_hidden[-1].unsqueeze(0)).float()
        # Zero-init so that m = s = 0 and z' = sigmoid(forget_bias) * z at the start
        self.out_layer = MaskedLinear(hidden_dim, 2 * d, torch.cat([m_out, m_out]), zero_init=True)

    def forward(self, z):
        h = z
        for layer in self.hidden_layers:
            h = torch.tanh(layer(h))
        out = self.out_layer(h)
        m = out[:, : self.d]
        s = out[:, self.d :]

        # Triangular Jacobian with sigma on the diagonal
        sigma = torch.sigmoid(s + self.forget_bias)
        zk = sigma * z + (1 - sigma) * m
        log_diag_j = torch.log(sigma).sum(dim=1)
        return zk, log_diag_j
