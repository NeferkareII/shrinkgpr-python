"""User-facing multivariate Student-t process regression (port of R/shrinkMVTPR.R)."""

from ._data import check_positive_scalar
from ._mvtpr_module import MVTPRModule
from .flows import sylvester
from .kernels import kernel_se
from .mvgpr import ShrinkMVGPR


class ShrinkMVTPR(ShrinkMVGPR):
    """Multivariate Student-t process regression with hierarchical shrinkage priors.

    The N x M response matrix is matrix-variate t,
    ``Y ~ MT(nu, 0, K + sigma2 I, Omega)``, which has heavier tails than the
    matrix normal of [`ShrinkMVGPR`][shrinkgpr.ShrinkMVGPR] and so is more robust to outliers.
    The priors are those of [`ShrinkMVGPR`][shrinkgpr.ShrinkMVGPR] plus
    ``nu - 2 ~ Gamma(nu_alpha, nu_beta)`` (shape, rate) on the degrees of
    freedom.

    Parameters
    ----------
    y, x, a, c, eta, a_Om, c_Om, sigma2_rate
        As for [`ShrinkMVGPR`][shrinkgpr.ShrinkMVGPR].
    nu_alpha, nu_beta : float
        Shape and rate of the gamma prior on ``nu - 2``.
    kernel_func, n_layers, flow_func, flow_args, device
        As for [`ShrinkMVGPR`][shrinkgpr.ShrinkMVGPR].

    Attributes
    ----------
    model_, last_model_ : MVTPRModule
        Best and last model, as for [`ShrinkMVGPR`][shrinkgpr.ShrinkMVGPR].
    loss_, loss_stor_, optimizer_, stop_reason_
        As for [`ShrinkMVGPR`][shrinkgpr.ShrinkMVGPR].
    """

    _module_class = MVTPRModule

    def __init__(self, y, x, *, a=0.5, c=0.5, eta=4, a_Om=0.5, c_Om=0.5, sigma2_rate=10,
                 nu_alpha=0.5, nu_beta=2, kernel_func=kernel_se, n_layers=10,
                 flow_func=sylvester, flow_args=None, device=None):
        check_positive_scalar(nu_alpha=nu_alpha, nu_beta=nu_beta)
        self.nu_alpha, self.nu_beta = nu_alpha, nu_beta
        super().__init__(y, x, a=a, c=c, eta=eta, a_Om=a_Om, c_Om=c_Om,
                         sigma2_rate=sigma2_rate, kernel_func=kernel_func, n_layers=n_layers,
                         flow_func=flow_func, flow_args=flow_args, device=device)

    def _module_kwargs(self):
        return {**super()._module_kwargs(), "nu_alpha": self.nu_alpha, "nu_beta": self.nu_beta}

    def calc_pred_moments(self, x_new=None, nsamp=100):
        """Parameters of the predictive distribution per posterior draw.

        For each of ``nsamp`` posterior draws, the predictive distribution at
        ``x_new`` is matrix-variate t with row scale ``K``, column scale
        ``Omega`` and ``nu`` degrees of freedom.

        Parameters
        ----------
        x_new : array-like of shape (N_new, p), optional
            Test covariates. Defaults to the training data.
        nsamp : int
            Number of posterior draws.

        Returns
        -------
        dict
            ``means`` of shape (nsamp, N_new, M), ``K`` of shape
            (nsamp, N_new, N_new), the posterior column scales ``Omega`` of
            shape (nsamp, M, M) and the posterior degrees of freedom ``nu`` of
            shape (nsamp,).
        """
        return super().calc_pred_moments(x_new, nsamp)

    def gen_posterior_samples(self, nsamp=1000):
        """Draws from the variational approximation of the posterior.

        As for [`ShrinkMVGPR.gen_posterior_samples`][shrinkgpr.ShrinkMVGPR.gen_posterior_samples], plus the degrees of
        freedom ``nu`` (array of shape (nsamp,)).
        """
        return super().gen_posterior_samples(nsamp)

    def _posterior_dict(self, params):
        return {**super()._posterior_dict(params), "nu": params["nu"]}
