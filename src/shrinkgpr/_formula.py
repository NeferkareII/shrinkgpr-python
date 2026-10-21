"""Design matrices from model formulas: the formula interface of the R fit functions."""

import re
import warnings

import numpy as np
import pandas as pd
from formulaic import model_matrix
from formulaic.errors import DataMismatchWarning


def _check_formula(formula, name):
    if not isinstance(formula, str):
        raise TypeError(f"'{name}' must be a formula string such as 'y ~ x1 + x2'")


def _sides(formula):
    """Left- and right-hand side of a two-sided formula string."""
    _check_formula(formula, "formula")
    if formula.count("~") != 1 or not formula.split("~")[0].strip():
        raise ValueError(f"'formula' must have the form 'response ~ covariates', got {formula!r}")
    lhs, rhs = (s.strip() for s in formula.split("~"))
    # R writes several responses as cbind(y1, y2); formulaic as y1 + y2
    m = re.fullmatch(r"cbind\((.*)\)", lhs)
    if m:
        lhs = " + ".join(s.strip() for s in m.group(1).split(","))
    return lhs, rhs


def _frame(mm):
    """A formulaic model matrix as a plain DataFrame of floats."""
    return pd.DataFrame(np.asarray(mm, dtype=float), columns=[str(c) for c in mm.columns])


def _build(spec_or_formula, data, what):
    """Model matrix from a formula or a stored model spec; NA values and unseen levels raise."""
    with warnings.catch_warnings():
        # formulaic maps unseen factor levels to an all-zero row; R raises
        warnings.simplefilter("error", DataMismatchWarning)
        try:
            if isinstance(spec_or_formula, str):
                return model_matrix(spec_or_formula, data, na_action="raise")
            return spec_or_formula.get_model_matrix(data, na_action="raise")
        except DataMismatchWarning as err:
            raise ValueError(f"New data for {what} has unseen factor levels: {err}") from None
        except ValueError as err:
            if "null values" in str(err):
                raise ValueError(f"No NA values are allowed in {what}") from None
            raise


class FormulaDesign:
    """Design matrices of a model built from formulas, and their model specs.

    As in R, the covariance formula ``formula`` builds its design matrix
    without an intercept, while the mean formula ``formula_mean`` keeps its
    intercept (named ``Intercept``). The specs rebuild the same columns from
    new data at prediction time, the equivalent of R's ``terms`` and
    ``xlevels``.
    """

    def __init__(self, formula, data, formula_mean=None):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("'data' must be a pandas DataFrame")
        lhs, rhs = _sides(formula)
        self.formula, self.formula_mean = formula, formula_mean

        mm = _build(f"{lhs} ~ {rhs} - 1", data, "the covariates")
        self.y = _frame(mm.lhs)
        self.x = _frame(mm.rhs)
        self._x_spec = mm.rhs.model_spec

        self.x_mean, self._x_mean_spec = None, None
        if formula_mean is not None:
            _check_formula(formula_mean, "formula_mean")
            # Any left-hand side of the mean formula is ignored, as in R's model.matrix
            mm_mean = _build(formula_mean.split("~")[-1], data, "the covariates of the mean")
            self.x_mean = _frame(mm_mean)
            self._x_mean_spec = mm_mean.model_spec

        # Training data and the covariates the formulas use, in data order (for
        # gen_marginal_samples, like R's model_internals$data)
        # variables_by_source, not required_variables, which reads x.1 as x
        specs = [self._x_spec] + ([self._x_mean_spec] if self._x_mean_spec else [])
        used = set().union(*(s.variables_by_source.get("data", set()) for s in specs))
        self.variables = [str(c) for c in data.columns if c in used]
        self.data = data[self.variables].copy()

    def new_x(self, data):
        """Covariance design matrix for new data."""
        return _frame(_build(self._x_spec, data, "the covariates"))

    def new_x_mean(self, data):
        """Mean design matrix for new data (None for models without a mean)."""
        if self._x_mean_spec is None:
            return None
        return _frame(_build(self._x_mean_spec, data, "the covariates of the mean"))
