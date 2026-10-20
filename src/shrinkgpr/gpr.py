"""User-facing Gaussian process regression with shrinkage priors (port of R/shrinkGPR.R)."""

import copy
import inspect
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ._data import (
    as_matrix,
    as_response,
    check_bool,
    check_positive_int,
    check_positive_scalar,
)
from ._gpr_module import GPRModule
from ._utils import lightweight_ols, merge_args
from .flows import sylvester
from .kernels import kernel_se

# Number of epochs per auto-stop check
N_CHECK = 100

ADAM_DEFAULTS = {
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 1e-3,
    "amsgrad": False,
}


def _flow_defaults(flow_func):
    """Keyword arguments of a flow and their defaults, excluding ``d``."""
    params = inspect.signature(flow_func).parameters
    return {k: p.default for k, p in params.items()
            if k != "d" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}


def _default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ShrinkGPR:
    """Gaussian process regression with hierarchical shrinkage priors.

    The posterior of the kernel parameters is approximated by normalizing
    flows trained through variational inference. A triple gamma prior on the
    inverse length scales ``theta`` shrinks irrelevant covariates towards
    zero; an optional linear mean has a normal-gamma-gamma prior on its
    coefficients.

    Parameters
    ----------
    y : array-like of shape (N,)
        Response.
    x : array-like of shape (N, p)
        Covariates of the covariance function. DataFrame column names are kept
        as labels.
    x_mean : array-like of shape (N, q), optional
        Covariates of the linear mean. No intercept is added.
    a, c : float
        Triple gamma prior hyperparameters for ``theta``.
    a_mean, c_mean : float
        Normal-gamma-gamma prior hyperparameters for the mean coefficients.
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
    model_ : GPRModule
        Model with the lowest loss seen during training.
    loss_ : float
        That lowest loss (negative ELBO).
    loss_stor_ : numpy.ndarray
        Loss of every epoch, across all calls to :meth:`fit`.
    last_model_ : GPRModule
        Model after the last epoch; training continues from it.
    optimizer_ : torch.optim.Adam
        Optimizer, with its state kept for continued training.
    stop_reason_ : str
        How the last call to :meth:`fit` ended: ``"max_iterations"``,
        ``"auto_stop"``, ``"interrupted"`` or ``"error"``.
    """

    _module_class = GPRModule

    def __init__(self, y, x, x_mean=None, *, a=0.5, c=0.5, a_mean=0.5, c_mean=0.5,
                 sigma2_rate=10, kernel_func=kernel_se, n_layers=10, flow_func=sylvester,
                 flow_args=None, device=None):
        check_positive_scalar(a=a, c=c, a_mean=a_mean, c_mean=c_mean, sigma2_rate=sigma2_rate)
        check_positive_int(n_layers=n_layers)
        if not callable(kernel_func):
            raise TypeError("'kernel_func' must be callable")
        if not callable(flow_func):
            raise TypeError("'flow_func' must be callable")
        if flow_args is not None and not isinstance(flow_args, dict):
            raise TypeError("'flow_args', if provided, has to be a dict")

        y_np = as_response(y)
        x_np, self.x_names_ = as_matrix(x, "x", "x")
        if x_np.shape[0] != y_np.shape[0]:
            raise ValueError("'y' and 'x' must have the same number of rows")
        if x_mean is not None:
            x_mean_np, self.x_mean_names_ = as_matrix(x_mean, "x_mean", "x_mean")
            if x_mean_np.shape[0] != y_np.shape[0]:
                raise ValueError("'y' and 'x_mean' must have the same number of rows")
        else:
            x_mean_np, self.x_mean_names_ = None, None

        self.device = _default_device() if device is None else torch.device(device)
        self.a, self.c, self.a_mean, self.c_mean = a, c, a_mean, c_mean
        self.sigma2_rate = sigma2_rate
        self.kernel_func = kernel_func
        self.n_layers = n_layers
        self.flow_func = flow_func
        self.flow_args = merge_args(_flow_defaults(flow_func), flow_args, "flow_args")

        def tensor(arr):
            return None if arr is None else torch.tensor(arr, dtype=torch.float32)

        self._module = self._module_class(
            tensor(y_np), tensor(x_np), tensor(x_mean_np), **self._module_kwargs()
        ).to(self.device)

    def _module_kwargs(self):
        return {"a": self.a, "c": self.c, "a_mean": self.a_mean, "c_mean": self.c_mean,
                "sigma2_rate": self.sigma2_rate, "n_layers": self.n_layers,
                "flow_func": self.flow_func, "flow_args": self.flow_args,
                "kernel_func": self.kernel_func}

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        mean = "none" if self.x_mean_names_ is None else ", ".join(self.x_mean_names_)
        return (f"{type(self).__name__}(N={self._module.N}, covariates=[{', '.join(self.x_names_)}],"
                f" mean=[{mean}], flow={getattr(self.flow_func, '__name__', self.flow_func)},"
                f" kernel={getattr(self.kernel_func, '__name__', self.kernel_func)}, {status})")

    @property
    def is_fitted(self):
        return hasattr(self, "model_")

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(f"This {type(self).__name__} is not fitted yet; call fit() first")

    def fit(self, n_epochs=1000, n_latent=10, auto_stop=True, display_progress=True,
            optim_control=None):
        """Train the variational approximation by maximizing the ELBO with Adam.

        Calling ``fit`` again continues from the last model and optimizer state.

        Parameters
        ----------
        n_epochs : int
            Maximum number of epochs.
        n_latent : int
            Number of samples from the variational distribution per epoch.
        auto_stop : bool
            Every 100 epochs, stop if a regression of the last 100 losses on
            the epoch number has a slope p-value above 0.2.
        display_progress : bool
            Show a progress bar and timing information.
        optim_control : dict, optional
            Arguments for ``torch.optim.Adam``; defaults ``lr=1e-3``,
            ``weight_decay=1e-3``. Only used on the first call.

        Returns
        -------
        self
        """
        check_positive_int(n_epochs=n_epochs, n_latent=n_latent)
        check_bool(auto_stop=auto_stop, display_progress=display_progress)
        if optim_control is not None and not isinstance(optim_control, dict):
            raise TypeError("'optim_control', if provided, has to be a dict")

        model = self._module
        if self.is_fitted:
            if optim_control is not None:
                warnings.warn("'optim_control' is ignored when continuing training", stacklevel=2)
            optimizer = self.optimizer_
            best_loss, best_state = self.loss_, self._best_state
        else:
            optim_args = merge_args(ADAM_DEFAULTS, optim_control, "optim_control")
            optimizer = torch.optim.Adam(model.parameters(), **optim_args)
            best_loss, best_state = np.inf, None

        loss_stor = np.full(n_epochs, np.nan)
        stop_reason = "max_iterations"
        epoch = n_done = 0
        pbar = tqdm(total=n_epochs, disable=not display_progress, dynamic_ncols=True)
        start = time.perf_counter()
        model.train()

        try:
            for epoch in range(1, n_epochs + 1):
                zk_pos, log_det_J = model(model.gen_batch(n_latent))
                loss = -model.elbo(zk_pos, log_det_J)
                loss_value = loss.item()
                loss_stor[epoch - 1] = loss_value
                n_done = epoch

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # As in R, the best model is stored after the step that followed its loss
                if np.isfinite(loss_value) and loss_value < best_loss:
                    best_loss = loss_value
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

                if auto_stop and epoch % N_CHECK == 0:
                    p_val = lightweight_ols(loss_stor[epoch - N_CHECK : epoch],
                                            np.arange(1, N_CHECK + 1))
                    if p_val > 0.2:
                        stop_reason = "auto_stop"
                        break

                if display_progress:
                    if epoch >= 50 and epoch % 10 == 0:
                        avg = np.mean(loss_stor[epoch - 50 : epoch])
                        pbar.set_postfix_str(f"Avg. loss last 50 iter.: {avg:.2f}")
                    pbar.update()
        except KeyboardInterrupt:
            stop_reason = "interrupted"
            warnings.warn(f"Training interrupted at iteration {epoch}. "
                          "Returning model trained so far.", stacklevel=2)
        except Exception as err:
            if best_state is None:
                raise
            stop_reason = "error"
            warnings.warn(f"Error occurred at iteration {epoch} ({type(err).__name__}: {err}). "
                          "Returning model trained so far.", stacklevel=2)
        finally:
            pbar.close()
        elapsed = time.perf_counter() - start

        if best_state is None:
            raise RuntimeError("Training stopped before a finite loss was computed")

        new_losses = loss_stor[:n_done]
        self.loss_stor_ = (new_losses if not self.is_fitted
                           else np.concatenate([self.loss_stor_, new_losses]))

        best_model = copy.deepcopy(model)
        best_model.load_state_dict(best_state)
        best_model.eval()

        self.model_ = best_model
        self.loss_ = best_loss
        self.last_model_ = model
        self.optimizer_ = optimizer
        self.stop_reason_ = stop_reason
        self._best_state = best_state

        if display_progress:
            msg = [f"Timing (elapsed): {elapsed:.2f} seconds.",
                   f"{round(n_done / elapsed) if elapsed > 0 else 0} iterations per second."]
            if stop_reason == "auto_stop" and epoch < n_epochs:
                msg.append(f"Auto stop triggered, iteration {epoch}")
            elif stop_reason == "max_iterations":
                msg.append(f"Max iterations reached, stopping at iteration {epoch}")
                msg.append("Check if convergence is reached by looking at the loss_stor_ "
                           "attribute")
            print("\n".join(msg), file=sys.stderr)

        return self

    # Prediction

    def _design(self, obj, name, names):
        """Covariates for prediction as a tensor on the model's device.

        DataFrame columns are matched by name when the model was fitted on
        named columns. With several covariates, a 1-D input is a single row.
        """
        if isinstance(obj, pd.DataFrame) and set(names) <= set(map(str, obj.columns)):
            obj = obj.rename(columns=str)[names]
        elif (len(names) > 1 and not isinstance(obj, (pd.DataFrame, pd.Series))
              and np.ndim(obj) == 1):
            obj = np.asarray(obj.cpu() if isinstance(obj, torch.Tensor) else obj)[None, :]
        arr, _ = as_matrix(obj, name, name)
        if arr.shape[1] != len(names):
            raise ValueError(f"'{name}' must have {len(names)} columns, got {arr.shape[1]}")
        module = self.model_
        return torch.tensor(arr, dtype=module.x.dtype, device=module.x.device)

    def _new_data(self, x_new, x_mean_new):
        """Test covariates and, for models with a mean, test mean covariates."""
        module = self.model_
        if x_new is None:
            if x_mean_new is not None:
                raise ValueError("'x_mean_new' requires 'x_new'")
            return module.x, module.x_mean
        x_t = self._design(x_new, "x_new", self.x_names_)
        if module.mean_zero:
            if x_mean_new is not None:
                warnings.warn("'x_mean_new' is ignored: the model has no mean", stacklevel=3)
            return x_t, None
        if x_mean_new is None:
            raise ValueError("The model has a mean; 'x_mean_new' is required with 'x_new'")
        xm_t = self._design(x_mean_new, "x_mean_new", self.x_mean_names_)
        if xm_t.shape[0] != x_t.shape[0]:
            raise ValueError("'x_new' and 'x_mean_new' must have the same number of rows")
        return x_t, xm_t

    def calc_pred_moments(self, x_new=None, nsamp=100, *, x_mean_new=None):
        """Moments of the posterior predictive distribution.

        For each of ``nsamp`` posterior draws of the parameters, the predictive
        distribution at ``x_new`` is Gaussian; its means and covariances are
        returned.

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
            ``means`` of shape (nsamp, N_new) and ``vars`` of shape
            (nsamp, N_new, N_new).
        """
        self._check_fitted()
        check_positive_int(nsamp=nsamp)
        x_t, xm_t = self._new_data(x_new, x_mean_new)
        means, covs = self.model_.calc_pred_moments(x_t, nsamp, xm_t)
        return {"means": means.cpu().numpy(), "vars": covs.cpu().numpy()}

    def predict(self, x_new=None, nsamp=100, *, x_mean_new=None):
        """Draws from the posterior predictive distribution.

        Parameters are as for :meth:`calc_pred_moments`.

        Returns
        -------
        numpy.ndarray of shape (nsamp, N_new)
        """
        self._check_fitted()
        check_positive_int(nsamp=nsamp)
        x_t, xm_t = self._new_data(x_new, x_mean_new)
        return self.model_.predict(x_t, nsamp, xm_t).cpu().numpy()

    def _single_row(self, x_new, x_mean_new):
        x_t, xm_t = self._new_data(x_new, x_mean_new)
        if x_new is None or x_t.shape[0] != 1:
            raise ValueError("'x_new' must contain exactly one row")
        return x_t, xm_t

    def eval_pred_dens(self, y_eval, x_new, nsamp=100, log=False, *, x_mean_new=None):
        """Posterior predictive density at a single test point.

        Parameters
        ----------
        y_eval : float or array-like of shape (n,)
            Values at which to evaluate the density.
        x_new : array-like of shape (1, p)
            The test point.
        nsamp : int
            Number of posterior draws averaged over.
        log : bool
            Return the log density.
        x_mean_new : array-like of shape (1, q), optional
            Mean covariates of the test point, for models with a mean.

        Returns
        -------
        numpy.ndarray of shape (n,)
        """
        self._check_fitted()
        check_positive_int(nsamp=nsamp)
        check_bool(log=log)
        x_t, xm_t = self._single_row(x_new, x_mean_new)
        y = torch.tensor(np.atleast_1d(as_response(np.atleast_1d(y_eval))),
                         dtype=x_t.dtype, device=x_t.device)
        return self.model_.eval_pred_dens(y, x_t, nsamp, xm_t, log=log).cpu().numpy()

    def lpds(self, y_new, x_new, nsamp=100, *, x_mean_new=None):
        """Log predictive density score of an observation ``y_new`` at a single test point."""
        y_new = np.atleast_1d(np.asarray(y_new, dtype=float))
        if y_new.size != 1:
            raise ValueError("'y_new' must be a single value")
        return float(self.eval_pred_dens(y_new, x_new, nsamp, log=True,
                                         x_mean_new=x_mean_new)[0])
