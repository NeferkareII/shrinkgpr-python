"""User-facing multivariate Gaussian process regression (port of R/shrinkMVGPR.R)."""

import numpy as np
import torch

from ._base import ADAM_DEFAULTS, ShrinkBase
from ._data import as_matrix, check_bool, check_positive_int, check_positive_scalar
from ._mvgpr_module import MVGPRModule
from .flows import sylvester
from .kernels import kernel_se

# R's shrinkMVGPR and shrinkMVTPR default to a smaller learning rate
MV_ADAM_DEFAULTS = {**ADAM_DEFAULTS, "lr": 1e-4}


class ShrinkMVGPR(ShrinkBase):
    """Multivariate Gaussian process regression with hierarchical shrinkage priors.

    The N x M response matrix is matrix normal,
    ``Y ~ MN(0, K + sigma2 I, Omega)``: the kernel matrix ``K`` has triple
    gamma priors on the inverse length scales ``theta``, and the M x M
    between-response covariance ``Omega = S D S`` has an LKJ prior on the
    correlation matrix ``D`` and triple gamma priors on the scales ``S``
    (whose product is 1). The posterior is approximated by normalizing flows
    trained through variational inference. There is no linear mean.

    Parameters
    ----------
    y : array-like of shape (N, M)
        Responses, M >= 2. DataFrame column names are kept as labels.
    x : array-like of shape (N, p)
        Covariates of the covariance function. DataFrame column names are kept
        as labels.
    a, c : float
        Triple gamma prior hyperparameters for ``theta``.
    eta : float
        Concentration of the LKJ prior on the correlations of ``Omega``;
        larger values push it towards the identity.
    a_Om, c_Om : float
        Triple gamma prior hyperparameters for the scales of ``Omega``.
    sigma2_rate : float
        Rate of the exponential prior on the noise variance.
    kernel_func : callable
        Kernel function, e.g. :func:`shrinkgpr.kernels.kernel_se`.
    n_layers : int
        Number of flow layers.
    flow_func : callable
        Flow class called as ``flow_func(d, **flow_args)``.
    flow_args : dict, optional
        Arguments for ``flow_func``; unknown names are ignored with a warning.
    device : str or torch.device, optional
        Defaults to CUDA when available.

    Attributes
    ----------
    model_ : MVGPRModule
        Model with the lowest loss seen during training.
    loss_ : float
        That lowest loss (negative ELBO).
    loss_stor_ : numpy.ndarray
        Loss of every epoch, across all calls to :meth:`fit`.
    last_model_ : MVGPRModule
        Model after the last epoch; training continues from it.
    optimizer_ : torch.optim.Adam
        Optimizer, with its state kept for continued training.
    stop_reason_ : str
        How the last call to :meth:`fit` ended: ``"max_iterations"``,
        ``"auto_stop"``, ``"interrupted"`` or ``"error"``.

    Notes
    -----
    As in R, training uses Adam with ``lr=1e-4`` by default and clips the
    gradient norm to 0.5 in every step.
    """

    _module_class = MVGPRModule
    _adam_defaults = MV_ADAM_DEFAULTS
    _max_grad_norm = 0.5

    def __init__(self, y, x, *, a=0.5, c=0.5, eta=4, a_Om=0.5, c_Om=0.5, sigma2_rate=10,
                 kernel_func=kernel_se, n_layers=10, flow_func=sylvester, flow_args=None,
                 device=None):
        check_positive_scalar(a=a, c=c, eta=eta, a_Om=a_Om, c_Om=c_Om,
                              sigma2_rate=sigma2_rate)
        self._init_data(y, x, kernel_func=kernel_func, n_layers=n_layers,
                        flow_func=flow_func, flow_args=flow_args, device=device)
        self.a, self.c, self.eta, self.a_Om, self.c_Om = a, c, eta, a_Om, c_Om
        self.sigma2_rate = sigma2_rate
        self._build_module()

    def _init_data(self, y, x, **setup):
        self._setup(**setup)
        y_np, self.y_names_ = as_matrix(y, "y", "y")
        if y_np.shape[1] < 2:
            raise ValueError("'y' must have at least 2 columns; use ShrinkGPR for a single "
                             "response")
        x_np, self.x_names_ = as_matrix(x, "x", "x")
        if x_np.shape[0] != y_np.shape[0]:
            raise ValueError("'y' and 'x' must have the same number of rows")
        self._y_np, self._x_np = y_np, x_np

    def _build_module(self):
        def tensor(arr):
            return torch.tensor(arr, dtype=torch.float32)

        self._module = self._module_class(
            tensor(self._y_np), tensor(self._x_np), **self._module_kwargs()
        ).to(self.device)
        del self._y_np, self._x_np

    def _module_kwargs(self):
        return {"a": self.a, "c": self.c, "eta": self.eta, "a_Om": self.a_Om,
                "c_Om": self.c_Om, "sigma2_rate": self.sigma2_rate,
                "n_layers": self.n_layers, "flow_func": self.flow_func,
                "flow_args": self.flow_args, "kernel_func": self.kernel_func}

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return (f"{type(self).__name__}(N={self._module.N}, responses=[{', '.join(self.y_names_)}],"
                f" covariates=[{', '.join(self.x_names_)}],"
                f" flow={getattr(self.flow_func, '__name__', self.flow_func)},"
                f" kernel={getattr(self.kernel_func, '__name__', self.kernel_func)}, {status})")

    # Prediction

    def _new_x(self, x_new):
        return self.model_.x if x_new is None else self._design(x_new, "x_new", self.x_names_)

    def calc_pred_moments(self, x_new=None, nsamp=100):
        """Parameters of the predictive distribution per posterior draw.

        For each of ``nsamp`` posterior draws of the parameters, the predictive
        distribution at ``x_new`` is matrix normal ``MN(means, K, Omega)``.

        Parameters
        ----------
        x_new : array-like of shape (N_new, p), optional
            Test covariates. Defaults to the training data.
        nsamp : int
            Number of posterior draws.

        Returns
        -------
        dict
            ``means`` of shape (nsamp, N_new, M), the row covariances ``K`` of
            shape (nsamp, N_new, N_new) and the column covariances ``Omega``
            of shape (nsamp, M, M).
        """
        self._check_fitted()
        check_positive_int(nsamp=nsamp)
        moments = self.model_.calc_pred_moments(self._new_x(x_new), nsamp)
        return {k: v.cpu().numpy() for k, v in moments.items()}

    def predict(self, x_new=None, nsamp=100):
        """Draws from the posterior predictive distribution.

        Parameters are as for :meth:`calc_pred_moments`.

        Returns
        -------
        numpy.ndarray of shape (nsamp, N_new, M)
        """
        self._check_fitted()
        check_positive_int(nsamp=nsamp)
        return self.model_.predict(self._new_x(x_new), nsamp).cpu().numpy()

    def _single_row(self, x_new):
        if x_new is None:
            raise ValueError("'x_new' must contain exactly one row")
        x_t = self._new_x(x_new)
        if x_t.shape[0] != 1:
            raise ValueError("'x_new' must contain exactly one row")
        return x_t

    def eval_pred_dens(self, y_eval, x_new, nsamp=100, log=False):
        """Posterior predictive density at a single test point.

        Parameters
        ----------
        y_eval : array-like of shape (n, M) or (M,)
            Response vectors at which to evaluate the density; a 1-D input is a
            single point.
        x_new : array-like of shape (1, p)
            The test point.
        nsamp : int
            Number of posterior draws averaged over.
        log : bool
            Return the log density.

        Returns
        -------
        numpy.ndarray of shape (n,)
        """
        self._check_fitted()
        check_positive_int(nsamp=nsamp)
        check_bool(log=log)
        x_t = self._single_row(x_new)
        M = self.model_.M
        if np.ndim(y_eval) == 1:
            y_eval = _as_numpy(y_eval)[None, :]
        y_np, _ = as_matrix(y_eval, "y_eval", "y")
        if y_np.shape[1] != M:
            raise ValueError(f"'y_eval' must have {M} columns (one per response), "
                             f"got {y_np.shape[1]}")
        y = torch.tensor(y_np, dtype=x_t.dtype, device=x_t.device)
        return self.model_.eval_pred_dens(y, x_t, nsamp, log=log).cpu().numpy()

    def lpds(self, y_new, x_new, nsamp=100):
        """Log predictive density score of an observed response vector ``y_new`` (length M)
        at a single test point."""
        self._check_fitted()
        y_new = _as_numpy(y_new)
        if y_new.size != self.model_.M:
            raise ValueError(f"'y_new' must be a single response vector of length "
                             f"{self.model_.M}")
        return float(self.eval_pred_dens(y_new.reshape(1, -1), x_new, nsamp, log=True)[0])


def _as_numpy(obj):
    return np.asarray(obj.detach().cpu() if isinstance(obj, torch.Tensor) else obj, dtype=float)
