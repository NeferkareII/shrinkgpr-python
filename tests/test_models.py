"""Behaviour shared by all models: the ShrinkBase training loop, prediction,
posterior samples and plots."""

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shrinkgpr
from shrinkgpr import (
    ShrinkGPR,
    ShrinkMVGPR,
    ShrinkMVTPR,
    ShrinkTPR,
    iaf,
    kernel_matern_32,
    kernel_se,
    nsf_cl,
    sylvester,
)

MODELS = [ShrinkGPR, ShrinkTPR, ShrinkMVGPR, ShrinkMVTPR]
MULTIVARIATE = (ShrinkMVGPR, ShrinkMVTPR)
N_TRAIN, N_TEST, M = 15, 5, 2


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(N_TRAIN + N_TEST, 2))
    f = np.sin(2 * x[:, 0])
    y = np.column_stack([f, np.cos(x[:, 1])]) + 0.1 * rng.normal(size=(N_TRAIN + N_TEST, M))
    return y, x


def split(cls, data):
    """Training and test data, with a univariate response for the univariate models."""
    y, x = data
    if cls not in MULTIVARIATE:
        y = y[:, 0]
    return y[:N_TRAIN], x[:N_TRAIN], y[N_TRAIN:], x[N_TRAIN:]


def fitted(cls, data, n_epochs=10, **kwargs):
    y, x, _, _ = split(cls, data)
    torch.manual_seed(0)
    return cls(y, x, device="cpu", **kwargs).fit(n_epochs=n_epochs, display_progress=False)


@pytest.fixture(scope="module", params=MODELS, ids=lambda c: c.__name__)
def fitted_model(request, data):
    return fitted(request.param, data)


def test_models_are_exported():
    for cls in MODELS:
        assert cls.__name__ in shrinkgpr.__all__
    assert issubclass(ShrinkTPR, ShrinkGPR) and issubclass(ShrinkMVTPR, ShrinkMVGPR)


# Training


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("flow", [sylvester, iaf, nsf_cl])
@pytest.mark.parametrize("kernel", [kernel_se, kernel_matern_32])
@pytest.mark.parametrize("auto_stop", [True, False])
def test_fit_scenarios(data, cls, flow, kernel, auto_stop):
    y, x, _, _ = split(cls, data)
    torch.manual_seed(0)
    model = cls(y, x, flow_func=flow, kernel_func=kernel, device="cpu")
    assert not model.is_fitted and "not fitted" in repr(model)
    model.fit(n_epochs=10, auto_stop=auto_stop, display_progress=False)
    assert model.is_fitted and model.model_ is not model.last_model_
    assert model.loss_stor_.shape == (10,) and np.isfinite(model.loss_stor_).all()
    assert model.loss_ == pytest.approx(model.loss_stor_.min())
    assert model.stop_reason_ == "max_iterations"


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
def test_fit_again_continues_training(data, cls):
    model = fitted(cls, data)
    optimizer, first_loss = model.optimizer_, model.loss_
    model.fit(n_epochs=7, display_progress=False)
    assert model.optimizer_ is optimizer and model.loss_stor_.shape == (17,)
    assert model.loss_ <= first_loss


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
def test_fit_is_reproducible_with_torch_seed(data, cls):
    a, b = fitted(cls, data, flow_func=iaf), fitted(cls, data, flow_func=iaf)
    np.testing.assert_array_equal(a.loss_stor_, b.loss_stor_)


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
def test_auto_stop(data, cls, monkeypatch):
    monkeypatch.setattr("shrinkgpr._base.lightweight_ols", lambda y, x: 0.5)
    model = fitted(cls, data, n_epochs=250)
    assert model.stop_reason_ == "auto_stop" and model.loss_stor_.shape == (100,)


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
def test_error_returns_partial_model(data, cls, monkeypatch):
    y, x, _, _ = split(cls, data)
    model = cls(y, x, device="cpu")
    module_cls = type(model._module)
    original, calls = module_cls.elbo, {"n": 0}

    def elbo(self, *args):
        calls["n"] += 1
        if calls["n"] == 4:
            raise RuntimeError("boom")
        return original(self, *args)

    monkeypatch.setattr(module_cls, "elbo", elbo)
    with pytest.warns(UserWarning, match="iteration 4"):
        model.fit(n_epochs=10, display_progress=False)
    assert model.stop_reason_ == "error" and model.loss_stor_.shape == (3,)


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
def test_optimizer_defaults(data, cls):
    group = fitted(cls, data, n_epochs=1).optimizer_.param_groups[0]
    # R's multivariate fits use a smaller learning rate and clip gradients
    assert group["lr"] == (1e-4 if cls in MULTIVARIATE else 1e-3)
    assert cls._max_grad_norm == (0.5 if cls in MULTIVARIATE else None)


# Prediction


def expected_moments(model, nsamp, n_new):
    if isinstance(model, ShrinkMVGPR):
        shapes = {"means": (nsamp, n_new, M), "K": (nsamp, n_new, n_new), "Omega": (nsamp, M, M)}
    else:
        shapes = {"means": (nsamp, n_new), "vars": (nsamp, n_new, n_new)}
    if isinstance(model, (ShrinkTPR, ShrinkMVTPR)):
        shapes["nu"] = (nsamp,)
    return shapes


def test_prediction_shapes(fitted_model, data):
    model = fitted_model
    _, _, y_test, x_test = split(type(model), data)
    out_dim = (M,) if isinstance(model, ShrinkMVGPR) else ()

    torch.manual_seed(1)
    draws = model.predict(x_test, nsamp=100)
    assert draws.shape == (100, N_TEST, *out_dim) and np.isfinite(draws).all()
    assert model.predict(nsamp=3).shape == (3, N_TRAIN, *out_dim)

    moments = model.calc_pred_moments(x_test, nsamp=100)
    assert {k: v.shape for k, v in moments.items()} == expected_moments(model, 100, N_TEST)
    # Full shapes are kept for a single draw and a single test row
    one = model.calc_pred_moments(x_test[:1], nsamp=1)
    assert {k: v.shape for k, v in one.items()} == expected_moments(model, 1, 1)

    y_eval = np.stack([y_test[0], y_test[0] + 0.5, y_test[0] - 0.5])
    dens = model.eval_pred_dens(y_eval, x_test[:1])
    assert dens.shape == (3,) and (dens > 0).all()

    # lpds is the log predictive density at the observed value
    torch.manual_seed(2)
    lpds = model.lpds(y_test[0], x_test[:1])
    torch.manual_seed(2)
    log_dens = model.eval_pred_dens(y_eval[:1], x_test[:1], log=True)
    assert isinstance(lpds, float) and np.isfinite(lpds)
    assert lpds == pytest.approx(float(log_dens[0]))


def test_single_row_methods_reject_several_rows(fitted_model, data):
    _, _, y_test, x_test = split(type(fitted_model), data)
    with pytest.raises(ValueError, match="one row"):
        fitted_model.lpds(y_test[0], x_test[:2])


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
def test_unfitted_model_cannot_predict(data, cls):
    y, x, _, x_test = split(cls, data)
    model = cls(y, x, device="cpu")
    for method in (model.predict, model.calc_pred_moments):
        with pytest.raises(RuntimeError, match="not fitted"):
            method(x_test)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.gen_posterior_samples(5)


# Posterior samples and plots


def expected_posterior(model, nsamp):
    if isinstance(model, ShrinkMVGPR):
        shapes = {"thetas": (nsamp, 2), "tau": (nsamp,), "sigma2": (nsamp,),
                  "tau_Om": (nsamp,), "Omega": (nsamp, M, M)}
    else:
        shapes = {"thetas": (nsamp, 2), "sigma2": (nsamp,), "tau": (nsamp,)}
    if isinstance(model, (ShrinkTPR, ShrinkMVTPR)):
        shapes["nu"] = (nsamp,)
    return shapes


def test_gen_posterior_samples(fitted_model):
    post = fitted_model.gen_posterior_samples(50)
    assert {k: v.shape for k, v in post.items()} == expected_posterior(fitted_model, 50)
    assert list(post["thetas"].columns) == ["theta_x1", "theta_x2"]
    for k in ("thetas", "sigma2", "tau"):
        assert (np.asarray(post[k]) > 0).all()
    if "nu" in post:
        # nu is shifted by 2 in both t processes
        assert (post["nu"] > 2).all()
    if "Omega" in post:
        np.testing.assert_allclose(post["Omega"], post["Omega"].transpose(0, 2, 1), rtol=1e-6)
        assert (np.linalg.eigvalsh(post["Omega"]) > 0).all()
    with pytest.raises(ValueError, match="nsamp"):
        fitted_model.gen_posterior_samples(0)


@pytest.mark.parametrize("cls", [ShrinkGPR, ShrinkTPR], ids=lambda c: c.__name__)
def test_univariate_posterior_is_the_raw_flow_output(data, cls):
    """As in R: the transformed latent vector, sliced, with nu shifted by 2."""
    y, x, _, _ = split(cls, data)
    x_mean = np.column_stack([np.ones(N_TRAIN), x[:, 0]])
    torch.manual_seed(0)
    model = cls(y, x, x_mean, device="cpu").fit(n_epochs=3, display_progress=False)
    torch.manual_seed(3)
    post = model.gen_posterior_samples(8)
    torch.manual_seed(3)
    with torch.no_grad():
        zk = model.model_(model.model_.gen_batch(8))[0].numpy()
    np.testing.assert_array_equal(post["thetas"].to_numpy(), zk[:, :2])
    np.testing.assert_array_equal(post["sigma2"], zk[:, 2])
    np.testing.assert_array_equal(post["tau"], zk[:, 3])
    np.testing.assert_array_equal(post["betas"].to_numpy(), zk[:, 4:6])
    np.testing.assert_array_equal(post["tau_mean"], zk[:, 6])
    assert list(post["betas"].columns) == ["beta_x_mean1", "beta_x_mean2"]
    if cls is ShrinkTPR:
        np.testing.assert_array_equal(post["nu"], zk[:, -1] + 2)


def test_plot_draws_theta_boxplots(fitted_model):
    ax = fitted_model.plot(nsamp=30)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["theta_x1", "theta_x2"]
    assert all(t.get_rotation() == 90 for t in ax.get_xticklabels())
    # Drawing on given axes
    _, own = plt.subplots()
    assert fitted_model.plot(nsamp=30, ax=own) is own
    plt.close("all")
