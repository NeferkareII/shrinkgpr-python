import numpy as np
import pandas as pd
import pytest
import torch

from shrinkgpr import ShrinkGPR, ShrinkMVGPR, ShrinkMVTPR, ShrinkTPR
from shrinkgpr._formula import FormulaDesign


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(5)
    n = 18
    out = pd.DataFrame({"x1": rng.normal(size=n), "x2": np.exp(rng.normal(size=n)),
                        "f": list("abc") * (n // 3)})
    out["y"] = np.sin(2 * out["x1"]) + 0.1 * rng.normal(size=n)
    out["y2"] = np.cos(out["x2"]) + 0.1 * rng.normal(size=n)
    return out


def fitted(cls, formula, data, n_epochs=5, **kwargs):
    torch.manual_seed(0)
    return cls.from_formula(formula, data, device="cpu", **kwargs).fit(
        n_epochs=n_epochs, display_progress=False)


# Design matrices


def test_covariance_design_has_no_intercept_and_full_factor_coding(df):
    design = FormulaDesign("y ~ f + x1", df)
    # As R with attr(terms, "intercept") <- 0: one column per level
    assert list(design.x.columns) == ["f[a]", "f[b]", "f[c]", "x1"]
    np.testing.assert_array_equal(design.x[["f[a]", "f[b]", "f[c]"]].sum(axis=1), 1)
    assert list(design.y.columns) == ["y"]
    assert design.x_mean is None and design.new_x_mean(df) is None


def test_mean_design_keeps_intercept(df):
    design = FormulaDesign("y ~ x1", df, formula_mean="~ x2 + f")
    assert list(design.x_mean.columns) == ["Intercept", "x2", "f[T.b]", "f[T.c]"]
    # A left-hand side of the mean formula is ignored
    other = FormulaDesign("y ~ x1", df, formula_mean="y ~ x2 + f")
    pd.testing.assert_frame_equal(design.x_mean, other.x_mean)


def test_transforms_and_interactions(df):
    design = FormulaDesign("np.log(y2 + 5) ~ x1 * np.log(x2)", df)
    assert list(design.x.columns) == ["x1", "np.log(x2)", "x1:np.log(x2)"]
    np.testing.assert_allclose(design.x["x1:np.log(x2)"], df["x1"] * np.log(df["x2"]))
    np.testing.assert_allclose(design.y.iloc[:, 0], np.log(df["y2"] + 5))


def test_new_data_reuses_the_specs(df):
    design = FormulaDesign("y ~ f + x1", df, formula_mean="~ f")
    new = pd.DataFrame({"x1": [0.5, -1.0], "f": ["c", "c"]})
    x_new = design.new_x(new)
    # All levels of the training data, although only "c" occurs
    assert list(x_new.columns) == list(design.x.columns)
    np.testing.assert_array_equal(x_new.to_numpy(), [[0, 0, 1, 0.5], [0, 0, 1, -1.0]])
    np.testing.assert_array_equal(design.new_x_mean(new).to_numpy(), [[1, 0, 1], [1, 0, 1]])


def test_unseen_levels_and_na_raise(df):
    design = FormulaDesign("y ~ f + x1", df)
    with pytest.raises(ValueError, match="unseen factor levels"):
        design.new_x(pd.DataFrame({"x1": [0.0], "f": ["z"]}))
    bad = df.copy()
    bad.loc[3, "x1"] = np.nan
    with pytest.raises(ValueError, match="No NA values"):
        FormulaDesign("y ~ x1", bad)
    bad = df.copy()
    bad.loc[3, "y"] = np.nan
    with pytest.raises(ValueError, match="No NA values"):
        FormulaDesign("y ~ x1", bad)
    with pytest.raises(ValueError, match="No NA values"):
        design.new_x(pd.DataFrame({"x1": [np.nan], "f": ["a"]}))


def test_cbind_response(df):
    a = FormulaDesign("cbind(y, y2) ~ x1", df)
    b = FormulaDesign("y + y2 ~ x1", df)
    assert list(a.y.columns) == ["y", "y2"]
    pd.testing.assert_frame_equal(a.y, b.y)


@pytest.mark.parametrize("formula, err", [("~ x1", ValueError), ("y ~ x1 ~ x2", ValueError),
                                          (None, TypeError)])
def test_invalid_formulas(df, formula, err):
    with pytest.raises(err):
        FormulaDesign(formula, df)


def test_data_must_be_a_dataframe(df):
    with pytest.raises(TypeError, match="DataFrame"):
        FormulaDesign("y ~ x1", df.to_dict())


# Models


@pytest.mark.parametrize("cls", [ShrinkGPR, ShrinkTPR])
def test_from_formula_matches_arrays(df, cls):
    """The same data as arrays and as formulas give the same fit and predictions."""
    by_formula = fitted(cls, "y ~ x1 + f", df, formula_mean="~ x2")
    design = FormulaDesign("y ~ x1 + f", df, "~ x2")
    torch.manual_seed(0)
    by_arrays = cls(df["y"], design.x, design.x_mean, device="cpu").fit(
        n_epochs=5, display_progress=False)
    np.testing.assert_array_equal(by_formula.loss_stor_, by_arrays.loss_stor_)
    assert by_formula.x_names_ == ["x1", "f[a]", "f[b]", "f[c]"]
    assert by_formula.x_mean_names_ == ["Intercept", "x2"]
    assert by_formula.formula_ == "y ~ x1 + f" and by_arrays.formula_ is None

    new = df.iloc[:4]
    torch.manual_seed(1)
    a = by_formula.predict(new, nsamp=5)
    torch.manual_seed(1)
    b = by_arrays.predict(design.new_x(new), nsamp=5, x_mean_new=design.new_x_mean(new))
    np.testing.assert_array_equal(a, b)
    torch.manual_seed(2)
    lp = by_formula.lpds(new["y"].iloc[0], new.iloc[[0]])
    torch.manual_seed(2)
    assert lp == by_arrays.lpds(new["y"].iloc[0], design.new_x(new.iloc[[0]]),
                                x_mean_new=design.new_x_mean(new.iloc[[0]]))


@pytest.mark.parametrize("cls", [ShrinkMVGPR, ShrinkMVTPR])
def test_mv_from_formula(df, cls):
    model = fitted(cls, "cbind(y, y2) ~ x1 + f", df)
    assert model.y_names_ == ["y", "y2"] and model.x_names_ == ["x1", "f[a]", "f[b]", "f[c]"]
    new = df.iloc[:3]
    assert model.predict(new, nsamp=4).shape == (4, 3, 2)
    assert model.calc_pred_moments(new, nsamp=4)["means"].shape == (4, 3, 2)
    assert np.isfinite(model.lpds(new[["y", "y2"]].iloc[0], new.iloc[[0]]))
    with pytest.raises(TypeError, match="DataFrame"):
        model.predict(new[["x1"]].to_numpy())


def test_formula_model_prediction_arguments(df):
    model = fitted(ShrinkGPR, "y ~ x1 + f", df, formula_mean="~ x2")
    new = df.iloc[:2]
    with pytest.raises(TypeError, match="DataFrame"):
        model.predict(new[["x1"]].to_numpy())
    with pytest.raises(ValueError, match="x_mean_new"):
        model.predict(new, x_mean_new=np.ones((2, 2)))
    with pytest.raises(ValueError, match="unseen factor levels"):
        model.predict(new.assign(f="z"))
    # Default: the training data
    assert model.predict(nsamp=2).shape == (2, len(df))


def test_from_formula_argument_checks(df):
    with pytest.raises(ValueError, match="ShrinkMVGPR"):
        ShrinkGPR.from_formula("y + y2 ~ x1", df, device="cpu")
    with pytest.raises(TypeError, match="formula_mean"):
        ShrinkGPR.from_formula("y ~ x1", df, x_mean=np.ones((len(df), 1)), device="cpu")
    with pytest.raises(ValueError, match="at least 2 columns"):
        ShrinkMVGPR.from_formula("y ~ x1", df, device="cpu")
    # Constructor arguments pass through
    model = ShrinkTPR.from_formula("y ~ x1", df, nu_alpha=2.0, a=0.3, device="cpu")
    assert model.nu_alpha == 2.0 and model.a == 0.3
