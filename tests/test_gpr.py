import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from shrinkgpr import ShrinkGPR, iaf, kernel_matern_32, nsf_cl, sylvester
from shrinkgpr._gpr_module import GPRModule


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(15, 2))
    y = np.sin(2 * x[:, 0]) + 0.1 * rng.normal(size=15)
    return y, x


def fitted(y, x, n_epochs=10, **kwargs):
    torch.manual_seed(0)
    return ShrinkGPR(y, x, device="cpu", **kwargs).fit(n_epochs=n_epochs, display_progress=False)


# Construction


def test_construction_does_not_train(data):
    model = ShrinkGPR(*data, device="cpu")
    assert not model.is_fitted and "not fitted" in repr(model)
    assert model.x_names_ == ["x1", "x2"] and model.x_mean_names_ is None
    assert model._module.d == 4 and model._module.y.dtype == torch.float32


def test_accepts_pandas_tensors_and_keeps_names(data):
    y, x = data
    df = pd.DataFrame(x, columns=["temp", "rain"])
    x_mean = pd.DataFrame({"Intercept": np.ones(15), "temp": x[:, 0]})
    model = ShrinkGPR(pd.Series(y), df, x_mean, device="cpu")
    assert model.x_names_ == ["temp", "rain"]
    assert model.x_mean_names_ == ["Intercept", "temp"]
    assert model._module.d == 2 + 2 + 2 + 1
    from_tensor = ShrinkGPR(torch.tensor(y), torch.tensor(x), device="cpu")
    torch.testing.assert_close(from_tensor._module.x, model._module.x)


def test_one_dimensional_x_is_a_column(data):
    y, x = data
    assert ShrinkGPR(y, x[:, 0], device="cpu")._module.x.shape == (15, 1)


@pytest.mark.parametrize(
    "kwargs, err, match",
    [
        ({"a": -1}, ValueError, "a must be positive"),
        ({"c": 0, "sigma2_rate": "1"}, ValueError, "c, sigma2_rate must all be positive"),
        ({"a": True}, ValueError, "a must be positive"),
        ({"n_layers": 2.5}, ValueError, "n_layers must be positive integers"),
        ({"kernel_func": "se"}, TypeError, "kernel_func"),
        ({"flow_func": None}, TypeError, "flow_func"),
        ({"flow_args": [1]}, TypeError, "flow_args"),
    ],
)
def test_invalid_arguments(data, kwargs, err, match):
    with pytest.raises(err, match=match):
        ShrinkGPR(*data, device="cpu", **kwargs)


def test_invalid_data(data):
    y, x = data
    x_na = x.copy()
    x_na[3, 1] = np.nan
    with pytest.raises(ValueError, match="No NA values are allowed in 'x'"):
        ShrinkGPR(y, x_na)
    y_na = y.copy()
    y_na[0] = np.nan
    with pytest.raises(ValueError, match="No NA values are allowed in 'y'"):
        ShrinkGPR(y_na, x)
    with pytest.raises(ValueError, match="same number of rows"):
        ShrinkGPR(y[:-1], x)
    with pytest.raises(ValueError, match="same number of rows"):
        ShrinkGPR(y, x, x[:-1])
    with pytest.raises(TypeError, match="from_formula"):
        ShrinkGPR(y, pd.DataFrame({"g": ["a"] * 15}))


def test_flow_args_are_merged_with_defaults(data):
    model = ShrinkGPR(*data, flow_func=iaf, flow_args={"hidden_dim": 8}, n_layers=2,
                      device="cpu")
    assert model.flow_args == {"hidden_dim": 8, "n_hidden": 1, "forget_bias": 2, "perm": None}
    assert model._module.layers[0].hidden_layers[0].weight.shape == (8, 4)
    with pytest.warns(UserWarning, match="Kk has been incorrectly named in 'flow_args'"):
        ShrinkGPR(*data, flow_func=nsf_cl, flow_args={"Kk": 3}, device="cpu")


# Fitting


@pytest.mark.parametrize("flow", [sylvester, iaf, nsf_cl])
@pytest.mark.parametrize("with_mean", [False, True])
def test_fit_sets_fitted_attributes(data, flow, with_mean):
    y, x = data
    x_mean = np.column_stack([np.ones(15), x[:, 0]]) if with_mean else None
    model = fitted(y, x, x_mean=x_mean, flow_func=flow, kernel_func=kernel_matern_32)
    assert model.is_fitted
    assert isinstance(model.model_, GPRModule) and model.last_model_ is model._module
    assert model.model_ is not model.last_model_
    assert isinstance(model.optimizer_, torch.optim.Adam)
    assert model.loss_stor_.shape == (10,) and np.isfinite(model.loss_stor_).all()
    assert model.loss_ == pytest.approx(model.loss_stor_.min())
    assert model.stop_reason_ == "max_iterations"


def test_best_model_is_a_snapshot(data):
    model = fitted(*data)
    best = model.model_.state_dict()
    model.fit(n_epochs=5, display_progress=False)
    if model.loss_ == model.loss_stor_[:10].min():
        # No improvement in the continued run: the stored best model is unchanged
        for k, v in model.model_.state_dict().items():
            assert torch.equal(v, best[k])
    for p_best, p_last in zip(model.model_.parameters(), model.last_model_.parameters(),
                              strict=True):
        assert p_best.data_ptr() != p_last.data_ptr()


def test_fit_again_continues_training(data):
    model = fitted(*data)
    optimizer = model.optimizer_
    steps = optimizer.state[next(iter(model._module.parameters()))]["step"].item()
    model.fit(n_epochs=7, display_progress=False)
    assert model.optimizer_ is optimizer
    assert optimizer.state[next(iter(model._module.parameters()))]["step"].item() == steps + 7
    assert model.loss_stor_.shape == (17,)
    assert model.loss_ <= model.loss_stor_[:10].min()
    with pytest.warns(UserWarning, match="optim_control"):
        model.fit(n_epochs=1, display_progress=False, optim_control={"lr": 1.0})


def test_optim_control(data):
    torch.manual_seed(0)
    model = ShrinkGPR(*data, device="cpu")
    with pytest.warns(UserWarning, match="lrr has been incorrectly named"):
        model.fit(n_epochs=2, display_progress=False, optim_control={"lr": 0.01, "lrr": 1})
    group = model.optimizer_.param_groups[0]
    assert group["lr"] == 0.01 and group["weight_decay"] == 1e-3


def test_fit_is_reproducible_with_torch_seed(data):
    a, b = fitted(*data, flow_func=iaf), fitted(*data, flow_func=iaf)
    np.testing.assert_array_equal(a.loss_stor_, b.loss_stor_)


def test_auto_stop(data, monkeypatch):
    monkeypatch.setattr("shrinkgpr._base.lightweight_ols", lambda y, x: 0.5)
    model = fitted(*data, n_epochs=250)
    assert model.stop_reason_ == "auto_stop" and model.loss_stor_.shape == (100,)
    torch.manual_seed(0)
    no_stop = ShrinkGPR(*data, device="cpu").fit(n_epochs=150, auto_stop=False,
                                                 display_progress=False)
    assert no_stop.stop_reason_ == "max_iterations" and no_stop.loss_stor_.shape == (150,)


@pytest.mark.parametrize("exc, reason", [(KeyboardInterrupt, "interrupted"),
                                         (RuntimeError("boom"), "error")])
def test_interrupt_or_error_returns_partial_model(data, monkeypatch, exc, reason):
    model = ShrinkGPR(*data, device="cpu")
    calls = {"n": 0}
    original = GPRModule.elbo

    def elbo(self, *args):
        calls["n"] += 1
        if calls["n"] == 4:
            raise exc
        return original(self, *args)

    monkeypatch.setattr(GPRModule, "elbo", elbo)
    with pytest.warns(UserWarning, match="iteration 4"):
        model.fit(n_epochs=10, display_progress=False)
    assert model.stop_reason_ == reason and model.loss_stor_.shape == (3,)
    assert model.is_fitted


def test_error_in_first_epoch_raises(data, monkeypatch):
    def elbo(self, *args):
        raise RuntimeError("ELBO is NaN")

    monkeypatch.setattr(GPRModule, "elbo", elbo)
    with pytest.raises(RuntimeError, match="ELBO is NaN"):
        ShrinkGPR(*data, device="cpu").fit(n_epochs=5, display_progress=False)


def test_fit_argument_checks(data):
    model = ShrinkGPR(*data, device="cpu")
    with pytest.raises(ValueError, match="n_epochs"):
        model.fit(n_epochs=0)
    with pytest.raises(TypeError, match="auto_stop"):
        model.fit(auto_stop="yes")
    with pytest.raises(TypeError, match="optim_control"):
        model.fit(optim_control=[0.1])


def test_display_progress_prints_summary(data, capsys):
    torch.manual_seed(0)
    ShrinkGPR(*data, device="cpu").fit(n_epochs=60, auto_stop=False)
    err = capsys.readouterr().err
    assert "Timing (elapsed)" in err and "Max iterations reached, stopping at iteration 60" in err
    assert "Avg. loss last 50 iter." in err


def test_fit_on_default_device(data):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = ShrinkGPR(*data).fit(n_epochs=5, display_progress=False)
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert model.device.type == expected and model.model_.x.device.type == expected
