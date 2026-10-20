"""Conversion and validation of user inputs."""

import numbers

import numpy as np
import pandas as pd
import torch


def as_matrix(obj, name, prefix):
    """Convert array-like input to a 2-D float64 numpy array plus column names.

    1-D input is treated as a single column. Names come from DataFrame or
    Series labels, otherwise they are ``<prefix>1, <prefix>2, ...``.
    """
    names = None
    if isinstance(obj, pd.DataFrame):
        non_numeric = [c for c in obj.columns if not pd.api.types.is_numeric_dtype(obj[c])]
        if non_numeric:
            raise TypeError(
                f"'{name}' has non-numeric columns {non_numeric}; use ShrinkGPR.from_formula "
                "for categorical covariates"
            )
        names = [str(c) for c in obj.columns]
        obj = obj.to_numpy(dtype=float, na_value=np.nan)
    elif isinstance(obj, pd.Series):
        names = [str(obj.name)] if obj.name is not None else None
        obj = obj.to_numpy(dtype=float, na_value=np.nan)
    elif isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()

    try:
        arr = np.asarray(obj, dtype=float)
    except (TypeError, ValueError) as err:
        raise TypeError(f"'{name}' has to be numeric") from err

    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"'{name}' has to be 1- or 2-dimensional")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"'{name}' must not be empty")
    if np.isnan(arr).any():
        raise ValueError(f"No NA values are allowed in '{name}'")
    if not np.isfinite(arr).all():
        raise ValueError(f"'{name}' must only contain finite values")

    if names is None:
        names = [f"{prefix}{j + 1}" for j in range(arr.shape[1])]
    return arr, names


def as_response(obj):
    """Convert the response to a 1-D float64 numpy array."""
    arr, _ = as_matrix(obj, "y", "y")
    if arr.shape[1] != 1:
        raise ValueError("'y' has to be one-dimensional")
    return arr[:, 0]


def check_positive_scalar(**kwargs):
    bad = [k for k, v in kwargs.items()
           if isinstance(v, bool) or not isinstance(v, numbers.Real) or not v > 0
           or not np.isfinite(v)]
    if bad:
        verb = "must" if len(bad) == 1 else "must all"
        raise ValueError(f"{', '.join(bad)} {verb} be positive numeric scalars")


def check_positive_int(**kwargs):
    bad = [k for k, v in kwargs.items()
           if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 1]
    if bad:
        verb = "must" if len(bad) == 1 else "must all"
        raise ValueError(f"{', '.join(bad)} {verb} be positive integers")


def check_bool(**kwargs):
    for k, v in kwargs.items():
        if not isinstance(v, (bool, np.bool_)):
            raise TypeError(f"'{k}' must be a single boolean value")
