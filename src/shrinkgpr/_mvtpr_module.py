"""Internal model for multivariate Student-t process regression (port of R/MVTPR_class.R).

The N x M response matrix is matrix-variate t,
``Y ~ MT(nu, 0, K + sigma2 I, Omega)``. The latent layout is that of
:class:`~shrinkgpr._mvgpr_module.MVGPRModule` with one trailing positive
column for the degrees of freedom:
``[corr (P), log-scales (M-1), theta_1..theta_p, tau, tau_Om, sigma2, nu - 2]``,
so ``d_latent = P + M - 1 + p + 4`` with ``P = M(M-1)/2``. As in the
univariate TPR, ``nu`` is the softplus output plus 2.

The matrix t follows Shah et al. (2014): ``Y | V ~ MN(0, K + sigma2 I, V)``
with ``V ~ InvWishart(Omega, nu + M - 1)``, so every row is multivariate t
with ``nu`` degrees of freedom.
"""

import math

import torch

from ._mvgpr_module import MVGPRModule
from ._utils import robust_chol


def ldt_multi(L_K, L_Om, y, nu):
    """Unnormalized matrix-t log-likelihood, batched over the latent samples.

    ``L_K`` (B x N x N) and ``L_Om`` (B x M x M) are the Cholesky factors of
    the row covariance ``K + sigma2 I`` and the column covariance ``Omega``;
    ``nu`` (B) are the degrees of freedom. Returns

    ``-(nu+N+M-1)/2 logdet(I_M + C) - M/2 logdet(K + sigma2 I) - N/2 logdet(Omega)
    - log Gamma_N((nu+N-1)/2) + log Gamma_N((nu+N+M-1)/2)``

    with ``C = L_Om^{-1} Y' (K + sigma2 I)^{-1} Y L_Om^{-T}``.
    """
    n_latent = L_K.shape[0]
    N, M = y.shape
    Y = y.expand(n_latent, N, M)
    alpha = torch.cholesky_solve(Y, L_K)
    B = Y.transpose(-2, -1) @ alpha

    X = torch.linalg.solve_triangular(L_Om, B, upper=False, left=True)
    C = torch.linalg.solve_triangular(L_Om.transpose(-2, -1), X, upper=True, left=False)

    eye_M = torch.eye(M, device=y.device, dtype=y.dtype).expand(n_latent, M, M)
    L_Iplus = robust_chol(eye_M + C)
    ld_Iplus = 2 * torch.log(torch.diagonal(L_Iplus, dim1=-2, dim2=-1)).sum(dim=1)

    ld_K = 2 * torch.log(torch.diagonal(L_K, dim1=-2, dim2=-1)).sum(dim=1)
    ld_Om = 2 * torch.log(torch.diagonal(L_Om, dim1=-2, dim2=-1)).sum(dim=1)

    nll = (0.5 * (nu + N + M - 1) * ld_Iplus
           + 0.5 * M * ld_K
           + 0.5 * N * ld_Om
           + (torch.mvlgamma(0.5 * (nu + N - 1), N)
              - torch.mvlgamma(0.5 * (nu + N + M - 1), N)))
    return -nll


def sample_matrix_t(means, row_cov, col_scale, nu):
    """Matrix-t draws in the parametrization of ``mniw::rMT``.

    One draw per batch element: ``V ~ InvWishart(col_scale, nu)`` and
    ``X | V ~ MN(means, row_cov, V)``, i.e. ``X = means + chol(row_cov) Z
    chol(V)'`` with standard normal ``Z``. Shapes are B x n x m, B x n x n,
    B x m x m and B.

    With ``col_scale = L L'`` and the Bartlett factor ``A`` (lower triangular,
    ``A_ii^2 ~ chi2(nu - i)``, standard normal below the diagonal), ``V^{-1} =
    L^{-T} A A' L^{-1}`` is Wishart(col_scale^{-1}, nu), so
    ``V = (L A^{-T})(L A^{-T})'`` and no Wishart draw needs to be inverted.
    """
    B, _, m = means.shape
    dtype, device = means.dtype, means.device
    idx = torch.arange(m, device=device, dtype=dtype)
    chi2 = 2 * torch.distributions.Gamma(0.5 * (nu.unsqueeze(1) - idx), 1.0).sample()
    A = torch.diag_embed(torch.sqrt(chi2))
    rows, cols = torch.tril_indices(m, m, offset=-1, device=device)
    A[:, rows, cols] = torch.randn(B, rows.numel(), device=device, dtype=dtype)

    L_col = robust_chol(col_scale)
    L_R = robust_chol(row_cov)
    Z = torch.randn(means.shape, device=device, dtype=dtype)
    # Z A^{-1}: rows of Z A^{-1} L' have covariance V
    Z_A = torch.linalg.solve_triangular(A, L_R @ Z, upper=False, left=False)
    return means + Z_A @ L_col.transpose(1, 2)


class MVTPRModule(MVGPRModule):
    """Variational MVTPR model: the MVGPR model with a matrix-t likelihood.

    ``nu - 2`` has a gamma prior with shape ``nu_alpha`` and rate ``nu_beta``.
    """

    n_extra = 1
    _pred_sigma2_tol = 1e-4

    def __init__(self, y, x, a=0.5, c=0.5, eta=4, a_Om=0.5, c_Om=0.5, sigma2_rate=10,
                 nu_alpha=0.5, nu_beta=2, n_layers=10, flow_func=None, flow_args=None,
                 kernel_func=None):
        super().__init__(y, x, a=a, c=c, eta=eta, a_Om=a_Om, c_Om=c_Om,
                         sigma2_rate=sigma2_rate, n_layers=n_layers, flow_func=flow_func,
                         flow_args=flow_args, kernel_func=kernel_func)
        self.register_buffer("nu_alpha", torch.tensor(float(nu_alpha), dtype=x.dtype))
        self.register_buffer("nu_beta", torch.tensor(float(nu_beta), dtype=x.dtype))

    @staticmethod
    def ldg(x, alpha, beta):
        """Gamma(alpha, rate beta) distribution (unnormalized)."""
        return (alpha - 1.0) * torch.log(x) - beta * x

    def unpack(self, zk_pos, sigma2_tol=1e-4):
        params = super().unpack(zk_pos, sigma2_tol)
        params["nu"] = zk_pos[:, self.n_omega + self.n_cov + 3] + 2
        return params

    def log_likelihood(self, K, L_Om, params):
        L_K = robust_chol(self.noisy_cov(K, params["sigma2"]))
        return ldt_multi(L_K, L_Om, self.y, params["nu"])

    def log_prior(self, params, omega):
        return (super().log_prior(params, omega)
                + self.ldg(params["nu"] - 2, self.nu_alpha, self.nu_beta).mean())

    # Prediction

    @torch.no_grad()
    def calc_pred_moments(self, x_new, nsamp):
        """Parameters of the matrix-t predictive distribution per posterior draw.

        Returns a dict with ``means`` (nsamp x N_new x M), the row scale ``K``
        (nsamp x N_new x N_new), the posterior column scale
        ``Omega = Omega + Y'(K + sigma2 I)^{-1} Y`` (nsamp x M x M) and the
        posterior degrees of freedom ``nu = nu + N`` (nsamp).
        """
        draws = self._pred_draws(x_new, nsamp)
        Omega_hat = draws["Omega"] + self.y.t().expand(nsamp, self.M, self.N) @ draws["alpha"]
        return {"means": draws["means"], "K": draws["K"], "Omega": Omega_hat,
                "nu": draws["params"]["nu"] + self.N}

    @torch.no_grad()
    def predict(self, x_new, nsamp):
        """Draws from the posterior predictive distribution (nsamp x N_new x M).

        Like R, which calls ``mniw::rMT`` with ``nu + M - 1``: the column
        covariance of each draw is ``InvWishart(Omega, nu + M - 1)``, so the
        draws follow the density of :meth:`eval_pred_dens`.
        """
        moments = self.calc_pred_moments(x_new, nsamp)
        return sample_matrix_t(moments["means"], moments["K"], moments["Omega"],
                               moments["nu"] + self.M - 1)

    @torch.no_grad()
    def eval_pred_dens(self, y_new, x_new, nsamp, log=False):
        """Posterior predictive density at a single test point, for each row of ``y_new``.

        ``y_new`` is n_eval x M. The density averages the ``nsamp`` multivariate
        t densities with ``nu`` degrees of freedom and scale ``K Omega / nu``.
        """
        moments = self.calc_pred_moments(x_new, nsamp)
        M = y_new.shape[1]
        nu = moments["nu"]
        diff = y_new.unsqueeze(0) - moments["means"]  # nsamp x n_eval x M

        L_Om = robust_chol(moments["Omega"])
        k_s = moments["K"][:, 0, 0]

        X = torch.linalg.solve_triangular(L_Om, diff.transpose(1, 2), upper=False)
        maha_scaled = X.pow(2).sum(dim=1) / k_s.unsqueeze(1)

        ld_Om = 2 * torch.log(torch.diagonal(L_Om, dim1=-2, dim2=-1)).sum(dim=1)
        log_norm = (torch.mvlgamma(0.5 * (nu + M), 1) - torch.mvlgamma(0.5 * nu, 1)
                    - 0.5 * M * math.log(math.pi) - 0.5 * M * torch.log(k_s) - 0.5 * ld_Om)
        log_comp = (log_norm.unsqueeze(1)
                    - 0.5 * (nu + M).unsqueeze(1) * torch.log1p(maha_scaled))

        res = self._log_mean_exp(log_comp)
        return res if log else torch.exp(res)
