"""User-facing Student-t process regression with shrinkage priors (port of R/shrinkTPR.R)."""

from ._data import check_positive_int, check_positive_scalar
from ._tpr_module import TPRModule
from .flows import sylvester
from .gpr import ShrinkGPR
from .kernels import kernel_se


class ShrinkTPR(ShrinkGPR):
    """Student-t process regression with hierarchical shrinkage priors.

    The Student-t process (Shah et al., 2014) replaces the Gaussian process of
    :class:`ShrinkGPR`; its degrees of freedom ``nu`` give heavier tails and
    more robustness to outliers. The noise is part of the process, so
    ``y ~ t_N(nu, mean, K + sigma2 I)``, parameterized by its covariance. The
    prior on the degrees of freedom is ``nu - 2 ~ Gamma(nu_alpha, nu_beta)``
    (shape, rate), so ``nu > 2``. Everything else is as in :class:`ShrinkGPR`.

    Parameters
    ----------
    y, x, x_mean, a, c, a_mean, c_mean, sigma2_rate
        As for :class:`ShrinkGPR`.
    nu_alpha : float
        Shape of the shifted gamma prior on ``nu``.
    nu_beta : float
        Rate of the shifted gamma prior on ``nu``.
    kernel_func, n_layers, flow_func, flow_args, device
        As for :class:`ShrinkGPR`.

    Attributes
    ----------
    model_, last_model_ : TPRModule
        Best and last model, as for :class:`ShrinkGPR`.
    loss_, loss_stor_, optimizer_, stop_reason_
        As for :class:`ShrinkGPR`.
    """

    _module_class = TPRModule

    def __init__(self, y, x, x_mean=None, *, a=0.5, c=0.5, a_mean=0.5, c_mean=0.5,
                 sigma2_rate=10, nu_alpha=0.5, nu_beta=2, kernel_func=kernel_se, n_layers=10,
                 flow_func=sylvester, flow_args=None, device=None):
        check_positive_scalar(nu_alpha=nu_alpha, nu_beta=nu_beta)
        self.nu_alpha, self.nu_beta = nu_alpha, nu_beta
        super().__init__(y, x, x_mean, a=a, c=c, a_mean=a_mean, c_mean=c_mean,
                         sigma2_rate=sigma2_rate, kernel_func=kernel_func, n_layers=n_layers,
                         flow_func=flow_func, flow_args=flow_args, device=device)

    def _module_kwargs(self):
        return {**super()._module_kwargs(), "nu_alpha": self.nu_alpha, "nu_beta": self.nu_beta}

    # Prediction

    def calc_pred_moments(self, x_new=None, nsamp=100, *, x_mean_new=None):
        """Moments of the posterior predictive distribution.

        For each of ``nsamp`` posterior draws of the parameters, the predictive
        distribution at ``x_new`` is a multivariate Student-t; its means,
        covariances and degrees of freedom are returned.

        Parameters
        ----------
        x_new : array-like of shape (N_new, p), optional
            Test covariates. Defaults to the training data.
        nsamp : int
            Number of posterior draws.
        x_mean_new : array-like of shape (N_new, q), optional
            Test covariates of the mean; required with ``x_new`` for models
            with a mean.

        Returns
        -------
        dict
            ``means`` of shape (nsamp, N_new), ``vars`` of shape
            (nsamp, N_new, N_new) and the degrees of freedom ``nu`` of shape
            (nsamp,), which are the posterior ``nu`` plus N.
        """
        self._check_fitted()
        check_positive_int(nsamp=nsamp)
        x_t, xm_t = self._new_data(x_new, x_mean_new)
        means, covs, nu = self.model_.calc_pred_moments(x_t, nsamp, xm_t)
        return {"means": means.cpu().numpy(), "vars": covs.cpu().numpy(),
                "nu": nu.cpu().numpy()}

    # Posterior

    def gen_posterior_samples(self, nsamp=1000):
        """Draws from the variational approximation of the posterior.

        As for :meth:`ShrinkGPR.gen_posterior_samples`, plus the degrees of
        freedom ``nu`` (array of shape (nsamp,)).
        """
        return super().gen_posterior_samples(nsamp)

    def _posterior_dict(self, zk):
        return {**super()._posterior_dict(zk), "nu": zk[:, -1] + 2}
