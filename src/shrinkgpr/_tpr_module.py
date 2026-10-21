"""Internal model for Student-t process regression (port of R/TPR_class.R)."""

import math

import torch

from ._gpr_module import GPRModule
from ._utils import res_protector, robust_chol


def ldt(K, L, sigma2, y, nu, x_mean=None, beta=None):
    """Unnormalized multivariate Student-t log-likelihood, batched over the latent samples.

    The t distribution is parameterized by its covariance ``K + sigma2 I``
    (scale ``(nu - 2) / nu`` times that), so ``nu > 2``. y is de-meaned by
    ``x_mean @ beta`` if a mean is given. ``L`` is the Cholesky factor of
    ``K + sigma2 I``; it is computed if ``None``.
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
    # Same expression as the TorchScript ldt in R
    return (torch.lgamma((nu + N) * 0.5) - 0.5 * N * torch.log(nu - 2)
            - torch.lgamma(0.5 * nu) - 0.5 * slogdet
            - 0.5 * (nu + N) * torch.log(1 + 1 / (nu - 2) * quad))


def ldt_univariate(y, mean, scale, nu):
    """Log density of the univariate Student-t with variance ``scale`` and ``nu > 2`` df."""
    return (torch.lgamma((nu + 1) * 0.5)
            - 0.5 * (torch.log(scale) + torch.log(nu - 2) + math.log(math.pi))
            - torch.lgamma(nu * 0.5)
            - ((nu + 1) * 0.5) * torch.log(1 + (y - mean) ** 2 / ((nu - 2) * scale)))


class TPRModule(GPRModule):
    """Variational TPR model: the GPR model plus the degrees of freedom ``nu``.

    The latent vector ends with one more positive column, ``nu - 2``. Its prior
    is ``nu - 2 ~ Gamma(nu_alpha, nu_beta)`` (shape, rate).

    As in R, the ELBO does not guard the parameters away from zero with
    :func:`res_protector`; prediction guards every column with ``tol=1e-6``.
    """

    n_extra = 1

    def __init__(self, y, x, x_mean=None, a=0.5, c=0.5, a_mean=0.5, c_mean=0.5,
                 sigma2_rate=10, nu_alpha=0.5, nu_beta=2, n_layers=10, flow_func=None,
                 flow_args=None, kernel_func=None):
        super().__init__(y, x, x_mean, a=a, c=c, a_mean=a_mean, c_mean=c_mean,
                         sigma2_rate=sigma2_rate, n_layers=n_layers, flow_func=flow_func,
                         flow_args=flow_args, kernel_func=kernel_func)
        # Same buffer names as in R
        self.register_buffer("nu_alpha", torch.tensor(float(nu_alpha), dtype=x.dtype))
        self.register_buffer("nu_beta", torch.tensor(float(nu_beta), dtype=x.dtype))

    @staticmethod
    def ldg(x, alpha, beta):
        """Gamma(alpha, beta) distribution (shape, rate), unnormalized."""
        return (alpha - 1.0) * torch.log(x) - beta * x

    def unpack(self, zk_pos):
        """Split transformed samples into named parameters; ``nu`` is shifted by 2."""
        p, q = self.n_cov, self.n_mean
        params = {
            "thetas": zk_pos[:, :p],
            "sigma2": zk_pos[:, p],
            "tau": zk_pos[:, p + 1],
            "betas": None,
            "nu": zk_pos[:, -1] + 2,
        }
        if not self.mean_zero:
            params["betas"] = zk_pos[:, p + 2 : p + 2 + q]
            params["tau_mean"] = zk_pos[:, -2]
        return params

    # ELBO

    def log_likelihood(self, K, L, params):
        return ldt(K, L, params["sigma2"], self.y, params["nu"], self.x_mean, params["betas"])

    def log_prior(self, params):
        return (super().log_prior(params)
                + self.ldg(params["nu"] - 2, self.nu_alpha, self.nu_beta).mean())

    # Prediction

    @torch.no_grad()
    def calc_pred_moments(self, x_new, nsamp, x_mean_new=None):
        """Predictive means, covariances and degrees of freedom.

        Each of the ``nsamp`` posterior draws gives one multivariate Student-t
        predictive distribution with mean (nsamp x N_new), covariance
        (nsamp x N_new x N_new) and ``nu + N`` degrees of freedom (nsamp).
        """
        zk_pos, _ = self(self.gen_batch(nsamp))
        params = self.unpack(res_protector(zk_pos))
        thetas, tau, sigma2, betas, nu = (params["thetas"], params["tau"], params["sigma2"],
                                          params["betas"], params["nu"])

        # L L' = K + sigma2 I, alpha = (K + sigma2 I)^{-1} y
        K = self.kernel_func(thetas, tau, self.x)
        L = robust_chol(self.noisy_cov(K, sigma2))
        if self.mean_zero:
            y = self.y.expand(nsamp, self.N, 1)
        else:
            y = (self.y - self.x_mean @ betas.t()).t().unsqueeze(-1)
        alpha = torch.cholesky_solve(y, L)

        eye_new = torch.eye(x_new.shape[0], device=x_new.device, dtype=x_new.dtype)
        K_star_star = self.kernel_func(thetas, tau, x_new) + eye_new * sigma2.view(-1, 1, 1)
        K_star_t = self.kernel_func(thetas, tau, self.x, x_new)

        pred_mean = (K_star_t @ alpha).squeeze(-1)
        if not self.mean_zero:
            pred_mean = pred_mean + (x_mean_new @ betas.t()).t()

        # The Gaussian conditional covariance, rescaled by the fit of the training data
        v = torch.linalg.solve_triangular(L, K_star_t.transpose(1, 2), upper=False)
        quad = (alpha.transpose(1, 2) @ y).view(-1)
        factor = (nu + quad - 2) / (nu + self.N - 2)
        pred_scale = (K_star_star - v.transpose(1, 2) @ v) * factor.view(-1, 1, 1)
        return pred_mean, pred_scale, nu + self.N

    def _draw_chi2(self, df):
        """Chi-squared draws with ``df`` degrees of freedom (R uses ``rgamma(n, df/2, 1/2)``)."""
        return torch.distributions.Chi2(df).sample()

    @torch.no_grad()
    def predict(self, x_new, nsamp, x_mean_new=None):
        """Draws from the posterior predictive distribution (nsamp x N_new)."""
        pred_mean, pred_scale, pred_nu = self.calc_pred_moments(x_new, nsamp, x_mean_new)
        chol = robust_chol(pred_scale)
        eps = torch.randn(nsamp, x_new.shape[0], 1, device=x_new.device, dtype=x_new.dtype)
        r = (1 / self._draw_chi2(pred_nu)).view(nsamp, 1)
        return pred_mean + torch.sqrt(r * (pred_nu.view(nsamp, 1) - 2)) * (chol @ eps).squeeze(-1)

    @torch.no_grad()
    def eval_pred_dens(self, y_new, x_new, nsamp, x_mean_new=None, log=False):
        """Posterior predictive density at a single test point, for each value in ``y_new``.

        The density is the average of the ``nsamp`` Student-t predictive
        densities, computed with log-sum-exp.
        """
        pred_mean, pred_scale, pred_nu = self.calc_pred_moments(x_new, nsamp, x_mean_new)
        log_dens = ldt_univariate(y_new.unsqueeze(1), pred_mean[:, 0], pred_scale[:, 0, 0],
                                  pred_nu)  # len(y_new) x nsamp
        res = torch.logsumexp(log_dens, dim=1) - math.log(nsamp)
        return res if log else torch.exp(res)
