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
    """Keyword arguments of a flow and their defaults, excluding ``d``."""
    params = inspect.signature(flow_func).parameters
    return {k: p.default for k, p in params.items()
            if k != "d" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}


def _default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ShrinkBase:
    """Base of the user-facing models.

    Subclasses validate their data, call :meth:`_setup` and assign the internal
    ``nn.Module`` to ``self._module``. That module must provide
    ``gen_batch(n)``, ``forward(z) -> (zk, log_det_J)`` and
    ``elbo(zk, log_det_J)``, and keep its training covariates in the buffer
    ``x``.
    """

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
