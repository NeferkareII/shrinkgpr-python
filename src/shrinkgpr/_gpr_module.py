"""Internal model for Gaussian process regression (port of R/GPR_class.R)."""

import math

import torch
from torch import nn

from ._hyperu import log_hyperu
from ._utils import res_protector, robust_chol


def ldnorm(K, L, sigma2, y, x_mean=None, beta=None):
    """Unnormalized Gaussian log-likelihood, batched over the latent samples.

    Returns ``-0.5 logdet(K + sigma2 I) - 0.5 y' (K + sigma2 I)^{-1} y`` with y
    de-meaned by ``x_mean @ beta`` if a mean is given. ``L`` is the Cholesky
    factor of ``K + sigma2 I``; it is computed if ``None``.
    """
    B, N, _ = K.shape
    if L is None:
        eye = torch.eye(N, device=K.device, dtype=K.dtype)
        L = torch.linalg.cholesky(K + eye * sigma2.view(B, 1, 1))

    slogdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=1)

    if beta is not None and x_mean is not None:
        y_batch = (y - x_mean @ beta.t()).t().unsqueeze(-1)
    else:
        y_batch = y.unsqueeze(0).expand(B, N, 1)

    alpha = torch.cholesky_solve(y_batch, L)
    quad = (y_batch.transpose(1, 2) @ alpha).squeeze(-1).squeeze(-1)
    return -0.5 * slogdet - 0.5 * quad


class GPRModule(nn.Module):
    """Variational GPR model: flows over the latent parameters plus the ELBO.

    The latent vector is laid out as ``[theta_1..theta_p, sigma2, tau]``,
    followed by ``[beta_1..beta_q, tau_mean]`` when there is a linear mean and
    then by ``n_extra`` further positive parameters (used by subclasses). All
    columns except the betas are mapped through a softplus.
    """

    n_extra = 0

    def __init__(self, y, x, x_mean=None, a=0.5, c=0.5, a_mean=0.5, c_mean=0.5,
                 sigma2_rate=10, n_layers=10, flow_func=None, flow_args=None,
                 kernel_func=None):
        super().__init__()
        self.n_cov = x.shape[1]
        self.mean_zero = x_mean is None
        self.n_mean = 0 if self.mean_zero else x_mean.shape[1]
        self.d = self.n_cov + 2 + self.n_extra + (0 if self.mean_zero else self.n_mean + 1)
        self.N = x.shape[0]

        self.kernel_func = kernel_func

        self.beta_sp = 0.7
        self.softplus = nn.Softplus(beta=self.beta_sp, threshold=20)

        flow_args = {} if flow_args is None else flow_args
        self.layers = nn.ModuleList(flow_func(self.d, **flow_args) for _ in range(n_layers))

        self.register_buffer("y", y.reshape(-1, 1))
        self.register_buffer("x", x)
        self.register_buffer("x_mean", x_mean)

        def scalar(v):
            return torch.tensor(float(v), dtype=x.dtype)

        self.register_buffer("prior_a", scalar(a))
        self.register_buffer("prior_c", scalar(c))
        self.register_buffer("prior_a_mean", scalar(a_mean))
        self.register_buffer("prior_c_mean", scalar(c_mean))
        self.register_buffer("prior_rate", scalar(sigma2_rate))

        positive = torch.ones(self.d, dtype=torch.bool)
        positive[self.n_cov + 2 : self.n_cov + 2 + self.n_mean] = False
        self.register_buffer("positive", positive)

    # Prior log densities (unnormalized)

    @staticmethod
    def ltg(x, a, c, tau):
        """Triple gamma prior on theta given the global scale tau."""
        tau = tau.unsqueeze(1)
        return (-0.5 * torch.log(tau) - 0.5 * torch.log(x)
                + log_hyperu(c + 0.5, 1.5 - a, a * x / (2.0 * c * tau)))

    @staticmethod
    def ngg(x, a, c, tau):
        """Normal-gamma-gamma prior on beta given the global scale tau."""
        tau = tau.unsqueeze(1)
        return -0.5 * torch.log(tau) + log_hyperu(c + 0.5, 1.5 - a, a * x**2 / (2.0 * c * tau))

    @staticmethod
    def lexp(x, rate):
        return torch.log(rate) - rate * x

    @staticmethod
    def ldf(x, d1, d2):
        """F(d1, d2) distribution."""
        return (d1 * 0.5 - 1.0) * torch.log(x) - (d1 + d2) * 0.5 * torch.log1p(d1 / d2 * x)

    # Flow and variational distribution

    def gen_batch(self, n_latent):
        return torch.randn(n_latent, self.d, device=self.y.device, dtype=self.y.dtype)

    def forward(self, zk):
        """Push base samples through the flows and the positivity transform.

        Returns the transformed samples and the log|det J| of the whole map.
        """
        log_det_J = 0
        for layer in self.layers:
            zk, log_diag_j = layer(zk)
            log_det_J = log_det_J + log_diag_j

        sp = self.softplus(zk)
        log_det_J = log_det_J + self.beta_sp * ((zk - sp) * self.positive).sum(dim=1)
        zk = torch.where(self.positive, sp, zk)
        return zk, log_det_J

    def unpack(self, zk_pos):
        """Split transformed samples into named parameters, guarded away from zero."""
        p, q = self.n_cov, self.n_mean
        params = {
            "thetas": res_protector(zk_pos[:, :p]),
            "sigma2": res_protector(zk_pos[:, p]),
            "tau": res_protector(zk_pos[:, p + 1], tol=1e-4),
            "betas": None,
        }
        if not self.mean_zero:
            params["betas"] = zk_pos[:, p + 2 : p + 2 + q]
            params["tau_mean"] = zk_pos[:, p + 2 + q]
        return params

    # ELBO

    def log_likelihood(self, K, L, params):
        return ldnorm(K, L, params["sigma2"], self.y, self.x_mean, params["betas"])

    def log_prior(self, params):
        thetas, tau = params["thetas"], params["tau"]
        prior = (self.ltg(thetas, self.prior_a, self.prior_c, tau).sum(dim=1).mean()
                 + self.ldf(tau, 2 * self.prior_c, 2 * self.prior_a).mean()
                 + self.lexp(params["sigma2"], self.prior_rate).mean())
        if not self.mean_zero:
            tau_mean = params["tau_mean"]
            prior = (prior
                     + self.ngg(params["betas"], self.prior_a_mean, self.prior_c_mean, tau_mean)
                     .sum(dim=1).mean()
                     + self.ldf(tau_mean, 2 * self.prior_c_mean, 2 * self.prior_a_mean).mean())
        return prior

    def noisy_cov(self, K, sigma2):
        eye = torch.eye(self.N, device=K.device, dtype=K.dtype)
        return K + eye * sigma2.view(-1, 1, 1)

    def elbo(self, zk_pos, log_det_J):
        params = self.unpack(zk_pos)
        K = self.kernel_func(params["thetas"], params["tau"], self.x)

        try:
            likelihood = self.log_likelihood(K, None, params).mean()
        except torch.linalg.LinAlgError:
            L = robust_chol(self.noisy_cov(K, params["sigma2"]))
            likelihood = self.log_likelihood(K, L, params).mean()

        elbo = likelihood + self.log_prior(params) + log_det_J.mean()
        if torch.isnan(elbo):
            raise RuntimeError("ELBO is NaN")
        return elbo

    # Prediction

    @torch.no_grad()
    def calc_pred_moments(self, x_new, nsamp, x_mean_new=None):
        """Predictive means (nsamp x N_new) and covariances (nsamp x N_new x N_new).

        Each of the ``nsamp`` posterior draws of the kernel parameters gives one
        Gaussian predictive distribution.
        """
        zk_pos, _ = self(self.gen_batch(nsamp))
        params = self.unpack(zk_pos)
        thetas, tau, sigma2, betas = (params["thetas"], params["tau"], params["sigma2"],
                                      params["betas"])

        # L L' = K + sigma2 I, alpha = (K + sigma2 I)^{-1} y
        K = self.kernel_func(thetas, tau, self.x)
        L = robust_chol(self.noisy_cov(K, sigma2))
        if self.mean_zero:
            y = self.y.expand(nsamp, self.N, 1)
        else:
            y = (self.y - self.x_mean @ betas.t()).t().unsqueeze(-1)
        alpha = torch.cholesky_solve(y, L)

        K_star_star = self.kernel_func(thetas, tau, x_new)
        K_star_t = self.kernel_func(thetas, tau, self.x, x_new)

        pred_mean = (K_star_t @ alpha).squeeze(-1)
        if not self.mean_zero:
            pred_mean = pred_mean + (x_mean_new @ betas.t()).t()

        eye_new = torch.eye(x_new.shape[0], device=x_new.device, dtype=x_new.dtype)
        v = torch.linalg.solve_triangular(L, K_star_t.transpose(1, 2), upper=False)
        pred_var = K_star_star - v.transpose(1, 2) @ v + eye_new * sigma2.view(-1, 1, 1)
        return pred_mean, pred_var

    @torch.no_grad()
    def predict(self, x_new, nsamp, x_mean_new=None):
        """Draws from the posterior predictive distribution (nsamp x N_new)."""
        pred_mean, pred_var = self.calc_pred_moments(x_new, nsamp, x_mean_new)
        chol = robust_chol(pred_var)
        eps = torch.randn(nsamp, x_new.shape[0], 1, device=x_new.device, dtype=x_new.dtype)
        return pred_mean + (chol @ eps).squeeze(-1)

    @torch.no_grad()
    def eval_pred_dens(self, y_new, x_new, nsamp, x_mean_new=None, log=False):
        """Posterior predictive density at a single test point, for each value in ``y_new``.

        The density is the average of the ``nsamp`` Gaussian predictive
        densities, computed with log-sum-exp.
        """
        pred_mean, pred_var = self.calc_pred_moments(x_new, nsamp, x_mean_new)
        dist = torch.distributions.Normal(pred_mean[:, 0], torch.sqrt(pred_var[:, 0, 0]))
        log_dens = dist.log_prob(y_new.unsqueeze(1))  # len(y_new) x nsamp
        res = torch.logsumexp(log_dens, dim=1) - math.log(nsamp)
        return res if log else torch.exp(res)
