"""Shared machinery of the user-facing model classes: setup, training loop and input helpers."""

import copy
import inspect
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ._data import as_matrix, check_bool, check_positive_int
from ._marginal import MarginalSamples
from ._utils import lightweight_ols, merge_args

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
    """Keyword arguments of a flow and their defaults, excluding ``d``.

    Arguments without a default map to ``inspect.Parameter.empty``.
    """
    params = inspect.signature(flow_func).parameters
    return {k: p.default for k, p in params.items()
            if k != "d" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}


def _default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _eval_ranges(eval_range, to_eval, data):
    """Grid ranges of gen_marginal_samples: given, or the ranges in the training data."""
    if eval_range is None:
        return [(float(data[v].min()), float(data[v].max())) for v in to_eval]
    pairs = [eval_range] if len(to_eval) == 1 else list(eval_range)
    shape = "a pair of numbers" if len(to_eval) == 1 else "two pairs of numbers"
    try:
        ranges = [tuple(float(b) for b in pair) for pair in pairs]
    except (TypeError, ValueError):
        raise ValueError(f"'eval_range' must be {shape}") from None
    if len(ranges) != len(to_eval) or any(len(r) != 2 or not np.isfinite(r).all()
                                          for r in ranges):
        raise ValueError(f"'eval_range' must be {shape}")
    return ranges


def _one_row(fixed_x, columns):
    """``fixed_x`` as a one-row DataFrame with the given columns."""
    if isinstance(fixed_x, dict):
        fixed_x = pd.DataFrame({k: [v] for k, v in fixed_x.items()})
    elif isinstance(fixed_x, pd.Series):
        fixed_x = fixed_x.to_frame().T
    if not isinstance(fixed_x, pd.DataFrame):
        raise TypeError("'fixed_x' must be a DataFrame, Series or dict")
    if len(fixed_x) == 0:
        raise ValueError("'fixed_x' has no rows")
    missing = [c for c in columns if c not in fixed_x.columns]
    if missing:
        raise ValueError(f"'fixed_x' lacks the covariates {missing}")
    return fixed_x.iloc[[0]][list(columns)].reset_index(drop=True)


def load(file, device=None):
    """Load a model saved with ``save``.

    Parameters
    ----------
    file : str, path or file-like
        The saved model.
    device : str or torch.device, optional
        Device to load to. By default the device it was saved from, or the
        CPU, with a warning, if that was CUDA and CUDA is not available.

    Returns
    -------
    ShrinkGPR, ShrinkTPR, ShrinkMVGPR or ShrinkMVTPR

    Notes
    -----
    The file is unpickled, which can run arbitrary code: only load files
    you trust.
    """
    content = torch.load(file, map_location="cpu", weights_only=False)
    if not isinstance(content, dict) or not isinstance(content.get("model"), ShrinkBase):
        raise TypeError("The file does not contain a model saved with save()")
    model = content["model"]
    from . import __version__

    saved = content.get("shrinkgpr_version")
    if saved != __version__:
        warnings.warn(f"Model was saved with shrinkgpr {saved}, this is {__version__}",
                      stacklevel=2)
    if device is None:
        device = model.device
        if device.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("Model was trained on CUDA but CUDA is not available. "
                          "Loading on CPU.", stacklevel=2)
            device = torch.device("cpu")
    return model._to_device(device)


class ShrinkBase:
    """Base of the user-facing models.

    Subclasses validate their data, call ``_setup`` and assign the internal
    ``nn.Module`` to ``self._module``. That module must provide
    ``gen_batch(n)``, ``forward(z) -> (zk, log_det_J)`` and
    ``elbo(zk, log_det_J)``, and keep its training covariates in the buffer
    ``x``.
    """

    # Adam defaults of fit() and the max. gradient norm to clip to (None: no
    # clipping); the multivariate models use lr=1e-4 and clip to 0.5, as in R
    _adam_defaults = ADAM_DEFAULTS
    _max_grad_norm = None

    # FormulaDesign of models built with from_formula
    _formula = None

    def _setup(self, *, kernel_func, n_layers, flow_func, flow_args, device):
        """Check and store the arguments shared by all models."""
        check_positive_int(n_layers=n_layers)
        if not callable(kernel_func):
            raise TypeError("'kernel_func' must be callable")
        if not callable(flow_func):
            raise TypeError("'flow_func' must be callable")
        if flow_args is not None and not isinstance(flow_args, dict):
            raise TypeError("'flow_args', if provided, has to be a dict")

        self.device = _default_device() if device is None else torch.device(device)
        self.kernel_func = kernel_func
        self.n_layers = n_layers
        self.flow_func = flow_func
        self.flow_args = merge_args(_flow_defaults(flow_func), flow_args, "flow_args")
        missing = [k for k, v in self.flow_args.items() if v is inspect.Parameter.empty]
        if missing:
            raise TypeError("'flow_func' requires " + ", ".join(f"'{k}'" for k in missing)
                            + ", which 'flow_args' does not give")

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
            ``weight_decay=1e-3`` (``lr=1e-4`` for the multivariate models).
            Only used on the first call.

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
            optim_args = merge_args(self._adam_defaults, optim_control, "optim_control")
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
                if self._max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self._max_grad_norm)
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

    @property
    def formula_(self):
        """The covariance formula of models built with ``from_formula``, else None."""
        return None if self._formula is None else self._formula.formula

    def _formula_data(self, x_new):
        """Check new data of a formula model; returns the formula design."""
        if not isinstance(x_new, pd.DataFrame):
            raise TypeError(f"This {type(self).__name__} was built with from_formula; "
                            "'x_new' must be a DataFrame with the variables of the formula")
        return self._formula

    def _posterior_draws(self, nsamp):
        """``nsamp`` draws of the transformed latent vector from the best model."""
        self._check_fitted()
        check_positive_int(nsamp=nsamp)
        module = self.model_
        with torch.no_grad():
            zk, _ = module(module.gen_batch(nsamp))
        return zk

    def plot(self, nsamp=1000, ax=None, **kwargs):
        """Boxplots of posterior draws of the inverse length scales ``theta``.

        Parameters
        ----------
        nsamp : int
            Number of posterior draws.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created by default.
        **kwargs
            Passed to ``Axes.boxplot``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        thetas = self.gen_posterior_samples(nsamp)["thetas"]
        if ax is None:
            _, ax = plt.subplots()
        ax.boxplot(thetas.to_numpy(), **kwargs)
        # Vertical labels, like R's las = 2
        ax.set_xticks(np.arange(1, thetas.shape[1] + 1), thetas.columns, rotation=90)
        return ax

    # Marginal predictions

    def gen_marginal_samples(self, to_eval, nsamp=200, fixed_x=None, n_eval_points=None,
                             eval_range=None, display_progress=True):
        """Posterior predictive means over a grid of one or two covariates.

        For each of ``nsamp`` draws, the other covariates are taken from a
        random row of the training data, or from ``fixed_x``; the covariates
        in ``to_eval`` are swept over a regular grid, and the predictive mean
        of one posterior draw is computed at every grid point.

        Parameters
        ----------
        to_eval : str or sequence of one or two str
            Covariates to sweep: variables of the formulas for models built
            with ``from_formula``, otherwise columns of ``x`` (or ``x_mean``).
        nsamp : int
            Number of draws.
        fixed_x : DataFrame, Series or dict, optional
            Values of the covariates that are not swept, used for every draw;
            the first row of a DataFrame is used. Swept covariates may be left out.
        n_eval_points : int, optional
            Grid points per covariate; 100 for one covariate and 30 for two.
            For two, every draw predicts at ``n_eval_points**2`` points.
        eval_range : pair of float, or two pairs for two covariates, optional
            Grid ranges; default to the ranges in the training data.
        display_progress : bool
            Show a progress bar.

        Returns
        -------
        MarginalSamples
            Its ``mean_pred`` has shape (nsamp, n) or (nsamp, n, n), plus a
            trailing response axis for the multivariate models; ``plot()``
            draws it.
        """
        self._check_fitted()
        to_eval = [to_eval] if isinstance(to_eval, str) else list(to_eval)
        if not 1 <= len(to_eval) <= 2 or not all(isinstance(v, str) for v in to_eval):
            raise ValueError("'to_eval' must be one or two covariate names")
        if len(set(to_eval)) != len(to_eval):
            raise ValueError("'to_eval' must name two different covariates")
        data, valid = self._marginal_frame()
        unknown = [v for v in to_eval if v not in valid]
        if unknown:
            raise ValueError(f"Unknown covariates in 'to_eval': {unknown}; the model's "
                             f"covariates are {valid}")
        non_numeric = [v for v in to_eval if not pd.api.types.is_numeric_dtype(data[v])]
        if non_numeric:
            raise TypeError(f"Only numeric covariates can be swept, not {non_numeric}")
        check_positive_int(nsamp=nsamp)
        check_bool(display_progress=display_progress)
        if n_eval_points is None:
            n_eval_points = 100 if len(to_eval) == 1 else 30
        check_positive_int(n_eval_points=n_eval_points)
        ranges = _eval_ranges(eval_range, to_eval, data)
        fixed = None if fixed_x is None else _one_row(
            fixed_x, [c for c in data.columns if c not in to_eval])

        grids = [np.linspace(lo, hi, n_eval_points) for lo, hi in ranges]
        if len(grids) == 1:
            sweep = {to_eval[0]: grids[0]}
        else:
            # Row order j * n + k, so that predictions reshape to [j, k]
            sweep = {to_eval[0]: np.repeat(grids[0], n_eval_points),
                     to_eval[1]: np.tile(grids[1], n_eval_points)}
        n_rows = n_eval_points ** len(grids)

        draws = []
        for _ in tqdm(range(nsamp), disable=not display_progress, dynamic_ncols=True):
            row = fixed if fixed is not None else data.iloc[
                [torch.randint(len(data), (1,)).item()]]
            new = pd.DataFrame({c: sweep[c] if c in sweep else np.repeat(row[c].to_numpy(), n_rows)
                                for c in data.columns})
            draws.append(self._marginal_means(new))
        mean_pred = np.stack(draws)
        if len(grids) == 2:
            mean_pred = mean_pred.reshape(nsamp, n_eval_points, n_eval_points,
                                          *mean_pred.shape[2:])
        return MarginalSamples(mean_pred, grids[0] if len(grids) == 1 else tuple(grids),
                               to_eval, self._response_names())

    # Saving and loading

    def save(self, file):
        """Save the model, fitted or not, with its optimizer state, to ``file``.

        The file is written with ``torch.save`` and read back with
        [`load`][shrinkgpr.load]; training can then continue with `fit`.
        """
        from . import __version__

        torch.save({"shrinkgpr_version": __version__, "model": self}, file)

    def _to_device(self, device):
        """Move the models, the optimizer state and the stored best state to ``device``."""
        device = torch.device(device)
        self._module.to(device)
        if self.is_fitted:
            # Module.to keeps the Parameter objects, so the optimizer still holds them
            self.model_.to(device)
            for state in self.optimizer_.state.values():
                for k, v in state.items():
                    # Adam keeps its step counts on the CPU (optim_control can't
                    # set capturable or fused, which would need them on the device)
                    if isinstance(v, torch.Tensor) and k != "step":
                        state[k] = v.to(device)
            self._best_state = {k: v.to(device) for k, v in self._best_state.items()}
        self.device = device
        return self

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
