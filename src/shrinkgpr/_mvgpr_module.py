"""Internal model for multivariate Gaussian process regression (port of R/MVGPR_class.R).

The N x M response matrix is matrix normal, ``Y ~ MN(0, K + sigma2 I, Omega)``.
The between-response covariance is ``Omega = L_Om L_Om'`` with
``L_Om = S L_D``: ``L_D`` is the Cholesky factor of a correlation matrix
``D`` and ``S = diag(s_1..s_M)`` holds scales whose product is 1.

Latent vector layout (0-based), with ``P = M(M-1)/2`` and ``Q = P + M - 1``:

- ``[0, P)``: unconstrained correlation parameters of ``D``;
- ``[P, Q)``: unconstrained log-scales of ``s_1..s_{M-1}`` (``log s_M`` is
  minus their sum);
- ``[Q, Q + p)``: ``theta_1..theta_p``;
- then ``tau``, ``tau_Om`` and ``sigma2``, followed by ``n_extra`` further
  positive parameters (``nu`` for the t process).

So ``d_latent = P + M - 1 + p + 3 (+ n_extra)``. The first ``Q`` columns stay
unconstrained; all others go through the softplus (``beta=0.7``).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from ._gpr_module import GPRModule
from ._utils import res_protector, robust_chol

# Smooth bound b * tanh(s / b) on the unconstrained log-scales
S_BOUND = 4.0
# Diagonal of L_Om is pushed above EPS_DIAG with a softplus of this sharpness
EPS_DIAG = 0.03
DIAG_BETA = 10.0


def ldnorm_multi(L_K, L_Om, y):
    """Unnormalized matrix-normal log-likelihood, batched over the latent samples.

    ``L_K`` (B x N x N) and ``L_Om`` (B x M x M) are the Cholesky factors of the
    row covariance ``K + sigma2 I`` and the column covariance ``Omega``. Returns
    ``-M/2 logdet(K + sigma2 I) - N/2 logdet(Omega)
    - 1/2 tr(Omega^{-1} Y' (K + sigma2 I)^{-1} Y)``.
    """
    n_latent = L_K.shape[0]
    N, M = y.shape
    alpha = torch.cholesky_solve(y.expand(n_latent, N, M), L_K)
    B = y.t().expand(n_latent, M, N) @ alpha
    om_inv_B = torch.cholesky_solve(B, L_Om)
    tr = -0.5 * torch.diagonal(om_inv_B, dim1=-2, dim2=-1).sum(dim=1)

    slogdet_K = 2.0 * torch.log(torch.diagonal(L_K, dim1=-2, dim2=-1)).sum(dim=1)
    slogdet_Om = 2.0 * torch.log(torch.diagonal(L_Om, dim1=-2, dim2=-1)).sum(dim=1)
    return -0.5 * M * slogdet_K - 0.5 * N * slogdet_Om + tr


def make_corr_chol(omega_uncons, M):
    """Cholesky factor of a correlation matrix from unconstrained parameters.

    Stan-style transform: the ``M(M-1)/2`` parameters are scaled, mapped into
    (-1, 1) by tanh and filled row by row into the strictly lower triangle;
    stick breaking then turns them into the rows of a unit-norm Cholesky
    factor. Returns ``L`` (B x M x M) and the log|det J| of the map (B).
    """
    n_latent = omega_uncons.shape[0]
    dtype, device = omega_uncons.dtype, omega_uncons.device
    p_omega = M * (M - 1) // 2
    temp = 2 + 4 * math.sqrt(math.log(p_omega))
    u = omega_uncons / temp
    z_vec = torch.tanh(u)

    # Strictly lower triangle, filled by row
    rows, cols = torch.tril_indices(M, M, offset=-1, device=device)
    z_mat = torch.zeros(n_latent, M, M, dtype=dtype, device=device)
    z_mat[:, rows, cols] = z_vec[:, :p_omega]

    # log|d tanh(u)/d omega| = -2 log cosh(u) - log(temp), with a stable log cosh
    logcosh = torch.logaddexp(u, -u) - math.log(2.0)
    logJ_tanh = -2.0 * logcosh.sum(dim=1) - p_omega * math.log(temp)

    # Stick breaking: 0.5 * sum_{i>j} log(remaining length before entry (i, j))
    logJ_sb = torch.zeros(n_latent, dtype=dtype, device=device)
    zeros = torch.zeros(n_latent, dtype=dtype, device=device)
    L_rows = [torch.stack([torch.ones_like(zeros)] + [zeros] * (M - 1), dim=1)]
    for i in range(1, M):
        rem = torch.ones_like(zeros)
        row = []
        for j in range(i):
            rem_before = torch.clamp(rem, min=1e-3)
            val = z_mat[:, i, j] * torch.sqrt(rem_before)
            row.append(val)
            rem = rem_before - val.pow(2)
            logJ_sb = logJ_sb + 0.5 * torch.log(rem_before)
        row.append(torch.sqrt(torch.clamp(rem, min=1e-3)))
        row.extend([zeros] * (M - 1 - i))
        L_rows.append(torch.stack(row, dim=1))

    L = torch.stack(L_rows, dim=1)
    return L, logJ_tanh + logJ_sb


def bias_diag(L):
    """Replace the diagonal of ``L`` by ``EPS_DIAG + softplus(diag - EPS_DIAG)``.

    Keeps the diagonal of the Cholesky factor of Omega away from zero.
    Returns the new factor and the original diagonal.
    """
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    diag2 = EPS_DIAG + F.softplus(diag - EPS_DIAG, beta=DIAG_BETA)
    eye = torch.eye(L.shape[-1], dtype=torch.bool, device=L.device)
    return torch.where(eye, torch.diag_embed(diag2), L), diag


class MVGPRModule(nn.Module):
    """Variational MVGPR model: flows over the latent parameters plus the ELBO.

    See the module docstring for the latent layout. Subclasses set
    ``n_extra`` for trailing positive parameters and override the hooks
    ``unpack``, ``log_likelihood`` and ``log_prior``.
    """

    n_extra = 0

    def __init__(self, y, x, a=0.5, c=0.5, eta=4, a_Om=0.5, c_Om=0.5, sigma2_rate=10,
                 n_layers=10, flow_func=None, flow_args=None, kernel_func=None):
        super().__init__()
        self.N, self.M = y.shape
        self.n_cov = x.shape[1]
        self.n_corr = self.M * (self.M - 1) // 2
        self.n_omega = self.n_corr + self.M - 1
        self.d = self.n_omega + self.n_cov + 3 + self.n_extra

        self.kernel_func = kernel_func

        self.beta_sp = 0.7
        self.softplus = nn.Softplus(beta=self.beta_sp, threshold=20)

        flow_args = {} if flow_args is None else flow_args
        self.layers = nn.ModuleList(flow_func(self.d, **flow_args) for _ in range(n_layers))

        self.register_buffer("y", y)
        self.register_buffer("x", x)

        def scalar(v):
            return torch.tensor(float(v), dtype=x.dtype)

        self.register_buffer("prior_a", scalar(a))
        self.register_buffer("prior_c", scalar(c))
        self.register_buffer("prior_eta", scalar(eta))
        self.register_buffer("prior_a_Om", scalar(a_Om))
        self.register_buffer("prior_c_Om", scalar(c_Om))
        self.register_buffer("prior_rate", scalar(sigma2_rate))

        positive = torch.ones(self.d, dtype=torch.bool)
        positive[: self.n_omega] = False
        self.register_buffer("positive", positive)

    # Prior log densities (unnormalized), shared with the univariate model
    ltg = staticmethod(GPRModule.ltg)
    ngg = staticmethod(GPRModule.ngg)
    lexp = staticmethod(GPRModule.lexp)
    ldf = staticmethod(GPRModule.ldf)

    # Flow and variational distribution

    def gen_batch(self, n_latent):
        """Base samples, with the Omega block scaled down and the theta block shifted.

        Starting near Omega = I and small theta stabilizes early training.
        """
        z = torch.randn(n_latent, self.d, device=self.y.device, dtype=self.y.dtype)
        q, p = self.n_omega, self.n_cov
        scale_omega = 0.2 / math.sqrt(q)
        return torch.cat([scale_omega * z[:, :q], z[:, q : q + p] - 2, z[:, q + p :]], dim=1)

    # Same flow pass and softplus on the masked columns as the univariate model
    forward = GPRModule.forward

    def unpack(self, zk_pos, sigma2_tol=1e-4):
        """Split transformed samples into named parameters.

        theta, tau and sigma2 are guarded away from zero; R uses tol 1e-4 for
        sigma2 in the ELBO and ``_pred_sigma2_tol`` in prediction.
        """
        q, p, k = self.n_corr, self.n_omega, self.n_omega + self.n_cov
        return {
            "D_uncons": zk_pos[:, :q],
            "S_uncons": zk_pos[:, q:p],
            "thetas": res_protector(zk_pos[:, p:k]),
            "tau": res_protector(zk_pos[:, k], tol=1e-4),
            "tau_Om": zk_pos[:, k + 1],
            "sigma2": res_protector(zk_pos[:, k + 2], tol=sigma2_tol),
        }

    def omega_factor(self, params):
        """Cholesky factor of Omega and the pieces the ELBO needs.

        Returns a dict with ``L_D`` and ``logJ_D`` (correlation factor and its
        log-Jacobian), ``S_diag`` (scales), ``log_det_S`` (log-Jacobian of the
        bounded scales), ``diag`` (diagonal of ``S L_D``) and ``L_Om`` (``S L_D``
        with its diagonal biased away from zero).
        """
        L_D, logJ_D = make_corr_chol(params["D_uncons"], self.M)

        s = params["S_uncons"]
        s_c = S_BOUND * torch.tanh(s / S_BOUND)
        S_logdiag = torch.cat([s_c, -s_c.sum(dim=1, keepdim=True)], dim=1)
        S_diag = torch.exp(S_logdiag)

        log_det_exp = s_c.sum(dim=1)
        log_det_squash = torch.log(torch.clamp(1 / torch.cosh(s / S_BOUND).pow(2),
                                               min=1e-12)).sum(dim=1)

        L_Om, diag = bias_diag(torch.diag_embed(S_diag) @ L_D)
        return {"L_D": L_D, "logJ_D": logJ_D, "S_diag": S_diag,
                "log_det_S": log_det_exp + log_det_squash, "diag": diag, "L_Om": L_Om}

    # ELBO

    def noisy_cov(self, K, sigma2):
        eye = torch.eye(self.N, device=K.device, dtype=K.dtype)
        return K + eye * sigma2.view(-1, 1, 1)

    def log_likelihood(self, K, L_Om, params):
        L_K = robust_chol(self.noisy_cov(K, params["sigma2"]))
        return ldnorm_multi(L_K, L_Om, self.y)

    def log_prior(self, params, omega):
        thetas, tau, tau_Om = params["thetas"], params["tau"], params["tau_Om"]
        logdet_D = 2 * torch.log(torch.diagonal(omega["L_D"], dim1=-2, dim2=-1)).sum(dim=1)
        lkj_term = (self.prior_eta - 1) * logdet_D
        return (self.ltg(thetas, self.prior_a, self.prior_c, tau).sum(dim=1).mean()
                + self.ldf(tau, 2 * self.prior_c, 2 * self.prior_a).mean()
                + lkj_term.mean()
                + self.ltg(omega["S_diag"], self.prior_a_Om, self.prior_c_Om, tau_Om)
                .sum(dim=1).mean()
                + self.ldf(tau_Om, 2 * self.prior_c_Om, 2 * self.prior_a_Om).mean()
                + self.lexp(params["sigma2"], self.prior_rate).mean())

    def elbo(self, zk_pos, log_det_J):
        params = self.unpack(zk_pos)
        K = self.kernel_func(params["thetas"], params["tau"], self.x)
        omega = self.omega_factor(params)

        likelihood = self.log_likelihood(K, omega["L_Om"], params).mean()
        if torch.isnan(likelihood):
            raise RuntimeError("Likelihood is NaN")

        prior = self.log_prior(params, omega)
        diag_biasing_logJ = torch.log(torch.sigmoid(DIAG_BETA * (omega["diag"] - EPS_DIAG))
                                      ).sum(dim=1)
        var_dens = (log_det_J.mean() + omega["logJ_D"].mean() + omega["log_det_S"].mean()
                    + diag_biasing_logJ.mean())

        elbo = likelihood + prior + var_dens
        if torch.isnan(elbo):
            raise RuntimeError("ELBO is NaN")
        return elbo

    # Prediction

    # Zero-guard of sigma2 in prediction (R's MVGPR uses 1e-6, its MVTPR 1e-4)
    _pred_sigma2_tol = 1e-6

    def _pred_draws(self, x_new, nsamp):
        """Posterior draws and the predictive pieces shared with the t process."""
        zk_pos, _ = self(self.gen_batch(nsamp))
        params = self.unpack(zk_pos, sigma2_tol=self._pred_sigma2_tol)
        thetas, tau, sigma2 = params["thetas"], params["tau"], params["sigma2"]
        L_Om = self.omega_factor(params)["L_Om"]
        Omega = L_Om @ L_Om.transpose(1, 2)

        # L L' = K + sigma2 I, alpha = (K + sigma2 I)^{-1} Y
        K = self.kernel_func(thetas, tau, self.x)
        L = robust_chol(self.noisy_cov(K, sigma2))
        alpha = torch.cholesky_solve(self.y.expand(nsamp, self.N, self.M), L)

        K_star_star = self.kernel_func(thetas, tau, x_new)
        K_star_t = self.kernel_func(thetas, tau, self.x, x_new)
        pred_mean = K_star_t @ alpha

        eye_new = torch.eye(x_new.shape[0], device=x_new.device, dtype=x_new.dtype)
        v = torch.linalg.solve_triangular(L, K_star_t.transpose(1, 2), upper=False)
        K_post = K_star_star - v.transpose(1, 2) @ v + eye_new * sigma2.view(-1, 1, 1)
        return {"means": pred_mean, "K": K_post, "Omega": Omega, "alpha": alpha,
                "params": params}

    @torch.no_grad()
    def calc_pred_moments(self, x_new, nsamp):
        """Parameters of the matrix-normal predictive distribution per posterior draw.

        Returns a dict with ``means`` (nsamp x N_new x M), the row covariance
        ``K`` (nsamp x N_new x N_new) and the column covariance ``Omega``
        (nsamp x M x M).
        """
        draws = self._pred_draws(x_new, nsamp)
        return {k: draws[k] for k in ("means", "K", "Omega")}

    @torch.no_grad()
    def predict(self, x_new, nsamp):
        """Draws from the posterior predictive distribution (nsamp x N_new x M)."""
        moments = self.calc_pred_moments(x_new, nsamp)
        L_S = robust_chol(moments["K"])
        # As in R, the diagonal of the factor of Omega is biased a second time
        L_Om, _ = bias_diag(robust_chol(moments["Omega"]))
        Z = torch.randn(nsamp, x_new.shape[0], self.M, device=x_new.device, dtype=x_new.dtype)
        return moments["means"] + L_S @ Z @ L_Om.transpose(1, 2)

    @staticmethod
    def _log_mean_exp(log_comp):
        """Log of the mean over the samples (dim 0), computed as in R."""
        m = log_comp.max(dim=0).values
        return m + torch.log(torch.exp(log_comp - m.unsqueeze(0)).mean(dim=0))

    @torch.no_grad()
    def eval_pred_dens(self, y_new, x_new, nsamp, log=False):
        """Posterior predictive density at a single test point, for each row of ``y_new``.

        ``y_new`` is n_eval x M. The density averages the ``nsamp`` Gaussian
        predictive densities ``N(mean, K Omega)``.
        """
        moments = self.calc_pred_moments(x_new, nsamp)
        M = y_new.shape[1]
        diff = y_new - moments["means"]  # nsamp x n_eval x M
        k_s = moments["K"][:, :, 0]  # nsamp x 1

        L_Om = robust_chol(moments["Omega"])
        X = torch.linalg.solve_triangular(L_Om, diff.transpose(1, 2), upper=False)
        quad = X.pow(2).sum(dim=1) / k_s

        logdet_Om = 2 * torch.log(torch.diagonal(L_Om, dim1=1, dim2=2)).sum(dim=1)
        logdet_K = M * torch.log(k_s)
        log_comp = -0.5 * (quad + M * math.log(2 * math.pi) + logdet_K
                           + logdet_Om.unsqueeze(1))

        res = self._log_mean_exp(log_comp)
        return res if log else torch.exp(res)
