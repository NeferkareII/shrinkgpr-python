import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from shrinkgpr import ShrinkGPR, kernel_matern_32
from shrinkgpr._gpr_module import GPRModule
from shrinkgpr.flows import sylvester
from shrinkgpr.kernels import kernel_se

F64 = torch.float64


def fixed_module(with_mean, nsamp=6, seed=0, kernel=kernel_se):
    """Float64 module whose gen_batch always returns the same base samples."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(10, 2, generator=g, dtype=F64)
    y = torch.sin(x[:, 0]) + 0.1 * torch.randn(10, generator=g, dtype=F64)
    x_mean = torch.cat([torch.ones(10, 1, dtype=F64), x[:, 1:]], 1) if with_mean else None
    torch.manual_seed(seed)
    module = GPRModule(y, x, x_mean, n_layers=2, flow_func=sylvester,
                       kernel_func=kernel).to(F64)
    z = torch.randn(nsamp, module.d, generator=g, dtype=F64)
    module.gen_batch = lambda n: z[:n] if n <= nsamp else z[torch.arange(n) % nsamp]
    return module, z


def closed_form(module, z, x_new, x_mean_new):
    """GP predictive moments per posterior draw, computed directly in numpy."""
    params = module.unpack(module(z)[0])
    th, tau, s2 = (params[k].detach().numpy() for k in ("thetas", "tau", "sigma2"))
    x, y = module.x.numpy(), module.y.numpy()[:, 0]
    xn = x_new.numpy()

    def k(a, b, t, ta):
        d = ((a[:, None, :] - b[None, :, :]) ** 2 * t).sum(-1)
        return np.exp(-0.5 * d) / ta

    means, covs = [], []
    for s in range(z.shape[0]):
        Kxx = k(x, x, th[s], tau[s]) + s2[s] * np.eye(len(x))
        Ksx = k(xn, x, th[s], tau[s])
        resid = y.copy()
        mean_new = np.zeros(len(xn))
        if not module.mean_zero:
            b = params["betas"][s].detach().numpy()
            resid -= module.x_mean.numpy() @ b
            mean_new = x_mean_new.numpy() @ b
        means.append(Ksx @ np.linalg.solve(Kxx, resid) + mean_new)
        covs.append(k(xn, xn, th[s], tau[s]) - Ksx @ np.linalg.solve(Kxx, Ksx.T)
                    + s2[s] * np.eye(len(xn)))
    return np.array(means), np.array(covs)


@pytest.mark.parametrize("with_mean", [False, True])
def test_moments_match_closed_form(with_mean):
    module, z = fixed_module(with_mean)
    x_new = torch.randn(4, 2, dtype=F64)
    x_mean_new = torch.cat([torch.ones(4, 1, dtype=F64), x_new[:, 1:]], 1) if with_mean else None
    means, covs = module.calc_pred_moments(x_new, 6, x_mean_new)
    exp_means, exp_covs = closed_form(module, z, x_new, x_mean_new)
    assert means.shape == (6, 4) and covs.shape == (6, 4, 4)
    np.testing.assert_allclose(means.numpy(), exp_means, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(covs.numpy(), exp_covs, rtol=1e-8, atol=1e-10)


def test_predict_draws_have_predictive_moments():
    # Every posterior draw is the same, so the draws share one Gaussian predictive
    module, _ = fixed_module(with_mean=True, nsamp=1)
    x_new = torch.randn(3, 2, dtype=F64)
    x_mean_new = torch.cat([torch.ones(3, 1, dtype=F64), x_new[:, 1:]], 1)
    torch.manual_seed(1)
    draws = module.predict(x_new, 40000, x_mean_new).numpy()
    mean, cov = (m[0].numpy() for m in module.calc_pred_moments(x_new, 1, x_mean_new))
    assert draws.shape == (40000, 3)
    sd = np.sqrt(np.diag(cov))
    np.testing.assert_allclose(draws.mean(0), mean, atol=4 * sd.max() / np.sqrt(40000))
    np.testing.assert_allclose(np.cov(draws.T), cov, atol=0.03 * np.abs(cov).max())


@pytest.mark.parametrize("log", [False, True])
def test_eval_pred_dens_averages_gaussian_densities(log):
    module, _ = fixed_module(with_mean=False, kernel=kernel_matern_32)
    x_new = torch.randn(1, 2, dtype=F64)
    y_eval = torch.tensor([-1.0, 0.0, 0.7], dtype=F64)
    means, covs = module.calc_pred_moments(x_new, 6)
    dens = stats.norm.pdf(y_eval.numpy()[:, None], means[:, 0].numpy(),
                          np.sqrt(covs[:, 0, 0].numpy())).mean(1)
    got = module.eval_pred_dens(y_eval, x_new, 6, log=log).numpy()
    np.testing.assert_allclose(got, np.log(dens) if log else dens, rtol=1e-10)


# ShrinkGPR methods


@pytest.fixture(scope="module")
def fitted_models():
    rng = np.random.default_rng(3)
    x = pd.DataFrame(rng.normal(size=(15, 2)), columns=["a", "b"])
    y = np.sin(2 * x["a"].to_numpy()) + 0.1 * rng.normal(size=15)
    x_mean = pd.DataFrame({"Intercept": np.ones(15), "b": x["b"]})
    torch.manual_seed(0)
    plain = ShrinkGPR(y, x).fit(n_epochs=10, display_progress=False)
    torch.manual_seed(0)
    mean = ShrinkGPR(y, x, x_mean).fit(n_epochs=10, display_progress=False)
    return plain, mean, x, x_mean, y


def test_shapes_and_defaults(fitted_models):
    plain, mean, x, x_mean, _ = fitted_models
    for model in (plain, mean):
        assert model.predict().shape == (100, 15)
        moments = model.calc_pred_moments(nsamp=7)
        assert moments["means"].shape == (7, 15) and moments["vars"].shape == (7, 15, 15)
    assert plain.predict(x.iloc[:4], nsamp=5).shape == (5, 4)
    assert mean.predict(x.iloc[:4], nsamp=5, x_mean_new=x_mean.iloc[:4]).shape == (5, 4)
    # A single draw and a single test point keep their dimensions
    assert plain.calc_pred_moments(x.iloc[:1], nsamp=1)["vars"].shape == (1, 1, 1)


def test_dataframe_columns_are_matched_by_name(fitted_models):
    plain, _, x, _, _ = fitted_models
    torch.manual_seed(5)
    a = plain.calc_pred_moments(x.iloc[:3], nsamp=4)["means"]
    torch.manual_seed(5)
    b = plain.calc_pred_moments(x.iloc[:3][["b", "a"]], nsamp=4)["means"]
    np.testing.assert_array_equal(a, b)


def test_eval_pred_dens_and_lpds(fitted_models):
    _, mean, x, x_mean, y = fitted_models
    row, row_mean = x.iloc[[2]], x_mean.iloc[[2]]
    torch.manual_seed(2)
    dens = mean.eval_pred_dens([y[2], 0.0], row, nsamp=50, x_mean_new=row_mean)
    torch.manual_seed(2)
    lpds = mean.lpds(y[2], row, nsamp=50, x_mean_new=row_mean)
    assert dens.shape == (2,) and (dens > 0).all()
    assert lpds == pytest.approx(np.log(dens[0]), rel=1e-5)
    # A 1-D array is one row when there are several covariates
    torch.manual_seed(2)
    one_d = mean.eval_pred_dens(y[2], row.to_numpy()[0], nsamp=50,
                                x_mean_new=row_mean.to_numpy()[0])
    np.testing.assert_allclose(one_d, dens[:1], rtol=1e-6)


def test_prediction_argument_checks(fitted_models):
    plain, mean, x, x_mean, y = fitted_models
    with pytest.raises(RuntimeError, match="not fitted"):
        ShrinkGPR(y, x).predict()
    with pytest.raises(ValueError, match="'x_mean_new' is required"):
        mean.predict(x.iloc[:2])
    with pytest.raises(ValueError, match="2 columns"):
        plain.predict(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="exactly one row"):
        plain.eval_pred_dens(0.0, x.iloc[:2])
    with pytest.raises(ValueError, match="exactly one row"):
        plain.lpds(0.0, None)
    with pytest.raises(ValueError, match="single value"):
        plain.lpds([0.0, 1.0], x.iloc[:1])
    with pytest.raises(ValueError, match="nsamp"):
        plain.predict(nsamp=0)
    with pytest.raises(ValueError, match="No NA"):
        plain.predict(x.iloc[:2].assign(a=np.nan))
    with pytest.warns(UserWarning, match="no mean"):
        plain.predict(x.iloc[:2], x_mean_new=x_mean.iloc[:2])


def test_prediction_on_default_device(fitted_models):
    plain, *_ = fitted_models
    out = plain.predict(nsamp=3)
    assert isinstance(out, np.ndarray) and np.isfinite(out).all()
