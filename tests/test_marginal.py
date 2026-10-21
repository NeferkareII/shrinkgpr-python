import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shrinkgpr import MarginalSamples, ShrinkGPR, ShrinkMVGPR, ShrinkMVTPR, ShrinkTPR

MODELS = [ShrinkGPR, ShrinkTPR, ShrinkMVGPR, ShrinkMVTPR]
MULTIVARIATE = (ShrinkMVGPR, ShrinkMVTPR)


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(7)
    n = 15
    out = pd.DataFrame({"x1": rng.uniform(size=n), "x2": rng.normal(size=n),
                        "f": list("abc") * (n // 3)})
    out["y"] = np.sin(2 * np.pi * out["x1"]) + 0.1 * rng.normal(size=n)
    out["y2"] = np.cos(out["x2"]) + 0.1 * rng.normal(size=n)
    return out


def fitted(cls, df, **kwargs):
    torch.manual_seed(0)
    if cls in MULTIVARIATE:
        model = cls(df[["y", "y2"]], df[["x1", "x2"]], device="cpu", **kwargs)
    else:
        model = cls(df["y"], df[["x1", "x2"]], device="cpu", **kwargs)
    return model.fit(n_epochs=5, display_progress=False)


@pytest.fixture(scope="module", params=MODELS, ids=lambda c: c.__name__)
def model(request, df):
    return fitted(request.param, df)


def marginal(model, to_eval, **kwargs):
    return model.gen_marginal_samples(to_eval, display_progress=False, **kwargs)


def test_shapes_grids_and_defaults(model, df):
    out = (2,) if isinstance(model, ShrinkMVGPR) else ()
    ms = marginal(model, "x1", nsamp=4)
    assert isinstance(ms, MarginalSamples) and ms.to_eval == ["x1"]
    assert ms.mean_pred.shape == (4, 100, *out) and np.isfinite(ms.mean_pred).all()
    np.testing.assert_allclose(ms.grid, np.linspace(df["x1"].min(), df["x1"].max(), 100))
    assert ms.response == (["y", "y2"] if out else ["y"]) and ms.M == len(ms.response)

    ms2 = marginal(model, ("x1", "x2"), nsamp=2)
    assert ms2.mean_pred.shape == (2, 30, 30, *out)
    np.testing.assert_allclose(ms2.grid[1], np.linspace(df["x2"].min(), df["x2"].max(), 30))

    ms3 = marginal(model, "x2", nsamp=3, n_eval_points=5, eval_range=(-1, 2))
    np.testing.assert_allclose(ms3.grid, np.linspace(-1, 2, 5))


def test_fixed_x_matches_calc_pred_moments(model, df):
    """Each draw is the predictive mean of one posterior draw, at the grid with fixed_x."""
    fixed = df.iloc[[3]]
    torch.manual_seed(5)
    ms = marginal(model, ["x1", "x2"], nsamp=2, fixed_x=fixed, n_eval_points=3,
                  eval_range=[(0, 1), (-1, 1)])
    g1, g2 = np.meshgrid(np.linspace(0, 1, 3), np.linspace(-1, 1, 3), indexing="ij")
    grid = pd.DataFrame({"x1": g1.ravel(), "x2": g2.ravel()})
    torch.manual_seed(5)
    expected = [model.calc_pred_moments(grid, nsamp=1)["means"][0] for _ in range(2)]
    # mean_pred[i, j, k] is at (grid1[j], grid2[k])
    np.testing.assert_allclose(ms.mean_pred,
                               np.stack(expected).reshape(ms.mean_pred.shape), rtol=1e-6)


def test_random_rows_fill_the_other_covariates(df):
    """Without fixed_x, a draw's prediction equals the one with that training row fixed."""
    model = fitted(ShrinkGPR, df)
    torch.manual_seed(9)
    ms = marginal(model, "x1", nsamp=6, n_eval_points=4)
    torch.manual_seed(9)
    for i in range(6):
        idx = torch.randint(len(df), (1,)).item()
        new = pd.DataFrame({"x1": ms.grid, "x2": np.repeat(df["x2"].iloc[idx], 4)})
        np.testing.assert_allclose(ms.mean_pred[i],
                                   model.calc_pred_moments(new, nsamp=1)["means"][0], rtol=1e-6)


def test_fixed_x_forms():
    model_df = pd.DataFrame({"x1": [0.1, 0.5, 0.9, 0.3], "x2": [1.0, 2.0, 0.5, 1.5]})
    torch.manual_seed(0)
    model = ShrinkGPR(np.sin(model_df["x1"]), model_df, device="cpu").fit(
        n_epochs=2, display_progress=False)
    results = []
    for fixed in ({"x1": 0.2, "x2": 1.0}, pd.Series({"x1": 0.2, "x2": 1.0}),
                  pd.DataFrame({"x2": [1.0, 7.0], "x1": [0.2, 0.0], "z": [0, 0]})):
        torch.manual_seed(1)
        results.append(marginal(model, "x1", nsamp=2, fixed_x=fixed, n_eval_points=3).mean_pred)
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[0], results[2])
    with pytest.raises(ValueError, match="lacks the covariates"):
        marginal(model, "x1", fixed_x={"x1": 0.2})
    with pytest.raises(TypeError, match="fixed_x"):
        marginal(model, "x1", fixed_x=[0.2, 1.0])


def test_formula_model_sweeps_variables_with_factors(df):
    torch.manual_seed(0)
    model = ShrinkTPR.from_formula("y ~ x1 + f", df, formula_mean="~ x2", device="cpu").fit(
        n_epochs=3, display_progress=False)
    ms = marginal(model, "x2", nsamp=3, n_eval_points=5)
    assert ms.mean_pred.shape == (3, 5) and ms.response == ["y"]
    ms = marginal(model, ["x1", "x2"], nsamp=2, n_eval_points=3, fixed_x=df.iloc[[0]])
    assert ms.mean_pred.shape == (2, 3, 3)
    with pytest.raises(TypeError, match="numeric"):
        marginal(model, "f")
    with pytest.raises(ValueError, match="Unknown covariates"):
        marginal(model, "y")


def test_mean_covariates_can_be_swept(df):
    torch.manual_seed(0)
    x_mean = pd.DataFrame({"m": df["x2"] ** 2})
    model = ShrinkGPR(df["y"], df[["x1"]], x_mean, device="cpu").fit(n_epochs=3,
                                                                      display_progress=False)
    assert marginal(model, ["x1", "m"], nsamp=2, n_eval_points=3).mean_pred.shape == (2, 3, 3)
    # A shared column is one covariate when the values agree
    shared = ShrinkGPR(df["y"], df[["x1"]], df[["x1"]], device="cpu").fit(
        n_epochs=2, display_progress=False)
    assert marginal(shared, "x1", nsamp=2, n_eval_points=3).mean_pred.shape == (2, 3)
    clash = ShrinkGPR(df["y"], df[["x1"]], df[["x1"]] * 2, device="cpu").fit(
        n_epochs=2, display_progress=False)
    with pytest.raises(ValueError, match="different values"):
        marginal(clash, "x1")


@pytest.mark.parametrize("kwargs, err, match", [
    ({"to_eval": ["x1", "x2", "x1"]}, ValueError, "one or two"),
    ({"to_eval": "x9"}, ValueError, "Unknown covariates"),
    ({"to_eval": "x1", "nsamp": 0}, ValueError, "nsamp"),
    ({"to_eval": "x1", "n_eval_points": 0}, ValueError, "n_eval_points"),
    ({"to_eval": "x1", "eval_range": (0, 1, 2)}, ValueError, "a pair"),
    ({"to_eval": ["x1", "x2"], "eval_range": (0, 1)}, ValueError, "two pairs"),
    ({"to_eval": "x1", "display_progress": "no"}, TypeError, "display_progress"),
])
def test_argument_checks(df, kwargs, err, match):
    model = fitted(ShrinkGPR, df)
    kwargs = {"display_progress": False, **kwargs}
    with pytest.raises(err, match=match):
        model.gen_marginal_samples(**kwargs)


def test_unfitted_model(df):
    with pytest.raises(RuntimeError, match="not fitted"):
        ShrinkGPR(df["y"], df[["x1"]], device="cpu").gen_marginal_samples("x1")


# Plots


def test_plot_1d(model):
    ms = marginal(model, "x1", nsamp=20, n_eval_points=10)
    ax = ms.plot()
    if ms.M == 1:
        # Two bands and the median
        assert len(ax.collections) == 2 and len(ax.lines) == 1
        assert ax.get_xlabel() == "x1" and ax.get_ylabel() == "y"
        np.testing.assert_allclose(ax.lines[0].get_ydata(), np.median(ms.mean_pred, axis=0))
    else:
        assert len(ax) == 2 and [a.get_ylabel() for a in ax] == ["y", "y2"]
    _, own = plt.subplots(ms.M, 1, squeeze=False)
    ms.plot(ax=own[:, 0], probs=(0.1, 0.9), xlabel="X", ylabel="Y")
    assert own[-1, 0].get_xlabel() == "X" and len(own[0, 0].collections) == 1
    with pytest.raises(ValueError, match="probs"):
        ms.plot_1d(probs=(0.1, 0.5, 0.9))
    with pytest.raises(ValueError, match="two covariates"):
        ms.plot_2d()
    plt.close("all")


def test_plot_2d(model, monkeypatch):
    ms = marginal(model, ["x1", "x2"], nsamp=5, n_eval_points=4)
    ax = ms.plot(alpha=0.5)
    axes = np.atleast_1d(ax)
    assert len(axes) == ms.M and axes[0].name == "3d"
    assert axes[0].get_xlabel() == "x1" and axes[0].get_zlabel() == ms.response[0]
    with pytest.raises(ValueError, match="single covariate"):
        ms.plot_1d()

    # One response, by index or name; its surface is that response's median
    from mpl_toolkits.mplot3d import Axes3D

    surfaces = []
    draw = Axes3D.plot_surface
    monkeypatch.setattr(Axes3D, "plot_surface",
                        lambda self, X, Y, Z, **kw: surfaces.append(Z) or draw(self, X, Y, Z, **kw))
    last = ms.M - 1
    median = np.median(ms.mean_pred[..., last] if ms.M > 1 else ms.mean_pred, axis=0)
    for response in (last, ms.response[last]):
        surfaces.clear()
        one = ms.plot_2d(response=response)
        assert one.get_zlabel() == ms.response[last] and len(surfaces) == 1
        np.testing.assert_array_equal(surfaces[0], median)
    for bad in (ms.M, -1, "nope", True, 0.5):
        with pytest.raises(ValueError, match="response"):
            ms.plot_2d(response=bad)
    plt.close("all")
