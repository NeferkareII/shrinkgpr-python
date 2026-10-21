import itertools

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from shrinkgpr._tpr_module import TPRModule, ldt, ldt_univariate
from shrinkgpr.flows import iaf, nsf_cl, sylvester
from shrinkgpr.kernels import kernel_matern_32, kernel_se
from shrinkgpr.tpr import ShrinkTPR

F64 = torch.float64


# Likelihood and priors


@pytest.mark.parametrize("with_mean", [False, True])
def test_ldt_matches_scipy_multivariate_t(with_mean):
    torch.manual_seed(1)
    B, N = 4, 6
    x = torch.randn(N, 2, dtype=F64)
    K = kernel_se(torch.rand(B, 2, dtype=F64), torch.rand(B, dtype=F64) + 0.5, x)
    sigma2 = torch.rand(B, dtype=F64) + 0.1
    nu = torch.tensor([2.5, 4.0, 10.0, 60.0], dtype=F64)
    y = torch.randn(N, 1, dtype=F64)
    x_mean = torch.randn(N, 2, dtype=F64) if with_mean else None
    beta = torch.randn(B, 2, dtype=F64) if with_mean else None

    got = ldt(K, None, sigma2, y, nu, x_mean, beta)
    cov = K + sigma2.view(B, 1, 1) * torch.eye(N, dtype=F64)
    loc = (x_mean @ beta.t()).t() if with_mean else torch.zeros(B, N, dtype=F64)
    # ldt is parameterized by the covariance, scipy by the scale matrix; it
    # leaves out the constant -N/2 log(pi)
    expected = [stats.multivariate_t.logpdf(y.numpy()[:, 0], loc[b].numpy(),
                                            cov[b].numpy() * (nu[b].item() - 2) / nu[b].item(),
                                            df=nu[b].item()) + 0.5 * N * np.log(np.pi)
                for b in range(B)]
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-10)
    torch.testing.assert_close(ldt(K, torch.linalg.cholesky(cov), sigma2, y, nu, x_mean, beta),
                               got)


def test_ldt_approaches_gaussian_for_large_nu():
    from shrinkgpr._gpr_module import ldnorm

    torch.manual_seed(2)
    x = torch.randn(5, 1, dtype=F64)
    K = kernel_se(torch.ones(2, 1, dtype=F64), torch.ones(2, dtype=F64), x)
    sigma2 = torch.full((2,), 0.2, dtype=F64)
    y = torch.randn(5, 1, dtype=F64)
    nu = torch.full((2,), 1e7, dtype=F64)
    # The Student-t constants lgamma((nu+N)/2) - lgamma(nu/2) - N/2 log(nu-2) tend to -N/2 log(2)
    torch.testing.assert_close(ldt(K, None, sigma2, y, nu) + 2.5 * np.log(2.0),
                               ldnorm(K, None, sigma2, y), rtol=1e-6, atol=1e-6)


def test_univariate_t_density_matches_scipy():
    y = torch.tensor([[-1.0], [0.3], [4.0]], dtype=F64)
    mean = torch.tensor([0.0, 0.5], dtype=F64)
    scale = torch.tensor([0.7, 2.0], dtype=F64)
    nu = torch.tensor([3.0, 20.0], dtype=F64)
    got = ldt_univariate(y, mean, scale, nu).numpy()
    expected = stats.t.logpdf(y.numpy(), df=nu.numpy(), loc=mean.numpy(),
                              scale=np.sqrt(scale.numpy() * (nu.numpy() - 2) / nu.numpy()))
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_gamma_prior_matches_scipy():
    x = torch.tensor([0.1, 0.5, 2.0, 7.0], dtype=F64)
    alpha, beta = torch.tensor(0.5, dtype=F64), torch.tensor(2.0, dtype=F64)
    got = TPRModule.ldg(x, alpha, beta).numpy()
    expected = stats.gamma.logpdf(x.numpy(), 0.5, scale=0.5)
    np.testing.assert_allclose(got - got[0], expected - expected[0], rtol=1e-12)


# Internal module


def make_module(with_mean=False, dtype=F64, flow=sylvester, kernel=kernel_se, seed=0, **kwargs):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(12, 3, generator=g, dtype=dtype)
    y = torch.sin(x[:, 0]) + 0.1 * torch.randn(12, generator=g, dtype=dtype)
    x_mean = torch.cat([torch.ones(12, 1, dtype=dtype), x[:, :1]], dim=1) if with_mean else None
    torch.manual_seed(seed)
    module = TPRModule(y, x, x_mean, n_layers=3, flow_func=flow, kernel_func=kernel, **kwargs)
    return module.to(dtype)


@pytest.mark.parametrize("with_mean, d", [(False, 3 + 3), (True, 3 + 3 + 3)])
def test_latent_layout(with_mean, d):
    module = make_module(with_mean)
    assert module.d == d
    zk, _ = module(module.gen_batch(50))
    params = module.unpack(zk)
    # nu is the last column, shifted by 2; tau_mean sits just before it
    torch.testing.assert_close(params["nu"], zk[:, -1] + 2)
    assert (params["nu"] > 2).all()
    if with_mean:
        assert not module.positive[5:7].any() and module.positive[7:].all()
        torch.testing.assert_close(params["tau_mean"], zk[:, -2])
        torch.testing.assert_close(params["betas"], zk[:, 5:7])
    else:
        assert module.positive.all() and params["betas"] is None


def test_prior_hyperparameters_are_buffers():
    module = make_module(nu_alpha=1.5, nu_beta=0.3)
    assert module.nu_alpha.item() == 1.5 and module.nu_beta.item() == pytest.approx(0.3)
    assert "nu_alpha" in dict(module.named_buffers())


@pytest.mark.parametrize("with_mean", [False, True])
def test_elbo_is_sum_of_components(with_mean):
    module = make_module(with_mean)
    zk, log_det_J = module(module.gen_batch(8))
    params = module.unpack(zk)
    K = module.kernel_func(params["thetas"], params["tau"], module.x)
    nu = params["nu"]
    expected = (ldt(K, None, params["sigma2"], module.y, nu, module.x_mean,
                    params["betas"]).mean()
                + module.ltg(params["thetas"], module.prior_a, module.prior_c, params["tau"])
                .sum(1).mean()
                + module.ldf(params["tau"], 2 * module.prior_c, 2 * module.prior_a).mean()
                + module.lexp(params["sigma2"], module.prior_rate).mean()
                + module.ldg(nu - 2, module.nu_alpha, module.nu_beta).mean()
                + log_det_J.mean())
    if with_mean:
        expected = (expected
                    + module.ngg(params["betas"], module.prior_a_mean, module.prior_c_mean,
                                 params["tau_mean"]).sum(1).mean()
                    + module.ldf(params["tau_mean"], 2 * module.prior_c_mean,
                                 2 * module.prior_a_mean).mean())
    torch.testing.assert_close(module.elbo(zk, log_det_J), expected)


@pytest.mark.parametrize("with_mean", [False, True])
@pytest.mark.parametrize("flow", [sylvester, iaf, nsf_cl])
def test_elbo_backprop_float32_on_device(with_mean, flow):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = make_module(with_mean, dtype=torch.float32, flow=flow,
                         kernel=kernel_matern_32).to(device)
    elbo = module.elbo(*module(module.gen_batch(10)))
    assert elbo.device.type == device and torch.isfinite(elbo)
    (-elbo).backward()
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_elbo_falls_back_to_robust_cholesky(monkeypatch):
    module = make_module(with_mean=True)
    zk, log_det_J = module(module.gen_batch(8))
    expected = module.elbo(zk, log_det_J)

    def failing_cholesky(*args, **kwargs):
        raise torch.linalg.LinAlgError("not positive-definite")

    monkeypatch.setattr(torch.linalg, "cholesky", failing_cholesky)
    torch.testing.assert_close(module.elbo(zk, log_det_J), expected)


# Prediction internals


def fixed_module(with_mean, nsamp=6, seed=0):
    """Float64 module whose gen_batch always returns the same base samples."""
    module = make_module(with_mean, seed=seed)
    z = torch.randn(nsamp, module.d, generator=torch.Generator().manual_seed(seed + 1),
                    dtype=F64)
    module.gen_batch = lambda n: z[:n] if n <= nsamp else z[torch.arange(n) % nsamp]
    return module, z


def closed_form(module, z, x_new, x_mean_new):
    """TP predictive moments per posterior draw (Shah et al., 2014), in numpy."""
    params = module.unpack(module(z)[0])
    th, tau, s2, nu = (params[k].detach().numpy() for k in ("thetas", "tau", "sigma2", "nu"))
    x, y = module.x.numpy(), module.y.numpy()[:, 0]
    xn = x_new.numpy()
    N = len(x)

    def k(a, b, t, ta):
        return np.exp(-0.5 * ((a[:, None, :] - b[None, :, :]) ** 2 * t).sum(-1)) / ta

    means, covs = [], []
    for s in range(z.shape[0]):
        Kxx = k(x, x, th[s], tau[s]) + s2[s] * np.eye(N)
        Ksx = k(xn, x, th[s], tau[s])
        resid = y.copy()
        mean_new = np.zeros(len(xn))
        if not module.mean_zero:
            b = params["betas"][s].detach().numpy()
            resid -= module.x_mean.numpy() @ b
            mean_new = x_mean_new.numpy() @ b
        alpha = np.linalg.solve(Kxx, resid)
        means.append(Ksx @ alpha + mean_new)
        gauss = (k(xn, xn, th[s], tau[s]) + s2[s] * np.eye(len(xn))
                 - Ksx @ np.linalg.solve(Kxx, Ksx.T))
        covs.append(gauss * (nu[s] + resid @ alpha - 2) / (nu[s] + N - 2))
    return np.array(means), np.array(covs), nu + N


@pytest.mark.parametrize("with_mean", [False, True])
def test_moments_match_closed_form(with_mean):
    module, z = fixed_module(with_mean)
    x_new = torch.randn(4, 3, dtype=F64)
    x_mean_new = torch.cat([torch.ones(4, 1, dtype=F64), x_new[:, :1]], 1) if with_mean else None
    means, covs, nu = module.calc_pred_moments(x_new, 6, x_mean_new)
    exp_means, exp_covs, exp_nu = closed_form(module, z, x_new, x_mean_new)
    assert means.shape == (6, 4) and covs.shape == (6, 4, 4) and nu.shape == (6,)
    np.testing.assert_allclose(means.numpy(), exp_means, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(covs.numpy(), exp_covs, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(nu.numpy(), exp_nu, rtol=1e-12)


def test_predict_draws_are_multivariate_t():
    # Every posterior draw is the same, so the draws share one t predictive
    module, _ = fixed_module(with_mean=True, nsamp=1)
    x_new = torch.randn(3, 3, dtype=F64)
    x_mean_new = torch.cat([torch.ones(3, 1, dtype=F64), x_new[:, :1]], 1)
    torch.manual_seed(1)
    n = 40000
    draws = module.predict(x_new, n, x_mean_new).numpy()
    mean, cov, nu = (m[0].numpy() for m in module.calc_pred_moments(x_new, 1, x_mean_new))
    assert draws.shape == (n, 3)
    # The covariance parameterization makes cov the covariance of the draws
    sd = np.sqrt(np.diag(cov))
    np.testing.assert_allclose(draws.mean(0), mean, atol=4 * sd.max() / np.sqrt(n))
    np.testing.assert_allclose(np.cov(draws.T), cov, atol=0.05 * np.abs(cov).max())
    # Marginally, the standardized draws follow a t with nu degrees of freedom
    std = (draws[:, 0] - mean[0]) / np.sqrt(cov[0, 0] * (nu - 2) / nu)
    assert stats.kstest(std, stats.t(df=float(nu)).cdf).pvalue > 1e-3


@pytest.mark.parametrize("log", [False, True])
def test_eval_pred_dens_averages_t_densities(log):
    module, _ = fixed_module(with_mean=False)
    x_new = torch.randn(1, 3, dtype=F64)
    y_eval = torch.tensor([-1.0, 0.0, 0.7], dtype=F64)
    means, covs, nu = (m.numpy() for m in module.calc_pred_moments(x_new, 6))
    scale = np.sqrt(covs[:, 0, 0] * (nu - 2) / nu)
    dens = stats.t.pdf(y_eval.numpy()[:, None], df=nu, loc=means[:, 0], scale=scale).mean(1)
    got = module.eval_pred_dens(y_eval, x_new, 6, log=log).numpy()
    np.testing.assert_allclose(got, np.log(dens) if log else dens, rtol=1e-10)


# ShrinkTPR


@pytest.fixture(scope="module")
def r_test_data():
    """Data shaped like R's test_shrinkTPR.R: 15 training and 5 test rows."""
    rng = np.random.default_rng(123)
    full = pd.DataFrame({"y": np.sin(2 * np.pi * rng.uniform(size=20))
                         + rng.normal(scale=0.1, size=20),
                         "x1": rng.uniform(size=20), "x2": rng.normal(size=20)})
    return full.iloc[:15], full.iloc[15:]


def test_construction_and_validation(r_test_data):
    train, _ = r_test_data
    model = ShrinkTPR(train["y"], train[["x1", "x2"]], device="cpu")
    assert not model.is_fitted and repr(model).startswith("ShrinkTPR(")
    assert isinstance(model._module, TPRModule) and model._module.d == 5
    assert (model.nu_alpha, model.nu_beta) == (0.5, 2)
    assert model._module.nu_beta.item() == 2.0
    with pytest.raises(ValueError, match="nu_alpha, nu_beta must all be positive"):
        ShrinkTPR(train["y"], train[["x1"]], nu_alpha=0, nu_beta=-1)
    with pytest.raises(ValueError, match="a must be positive"):
        ShrinkTPR(train["y"], train[["x1"]], a=-1)


SCENARIOS = list(itertools.product([True, False], [kernel_se, kernel_matern_32],
                                   [sylvester, nsf_cl, iaf], [True, False], [False, True]))


@pytest.mark.parametrize("auto_stop, kernel, flow, display_progress, with_mean", SCENARIOS,
                         ids=lambda v: getattr(v, "__name__", str(v)))
def test_r_scenarios(r_test_data, auto_stop, kernel, flow, display_progress, with_mean):
    train, test = r_test_data
    x_cols = ["x1", "x2"]

    def mean_of(df):
        return df[x_cols].assign(Intercept=1.0)[["Intercept", *x_cols]] if with_mean else None

    torch.manual_seed(123)
    model = ShrinkTPR(train["y"], train[x_cols], mean_of(train), kernel_func=kernel,
                      flow_func=flow, device="cpu")
    model.fit(n_epochs=10, auto_stop=auto_stop, display_progress=display_progress)
    assert isinstance(model.model_, TPRModule) and np.isfinite(model.loss_stor_).all()

    xm_test, xm_row = mean_of(test), mean_of(test.iloc[[0]])
    kw = {} if not with_mean else {"x_mean_new": xm_test}
    kw_row = {} if not with_mean else {"x_mean_new": xm_row}
    assert model.predict(test[x_cols], **kw).shape == (100, 5)
    lpds = model.lpds(test["y"].iloc[0], test[x_cols].iloc[[0]], **kw_row)
    assert isinstance(lpds, float) and np.isfinite(lpds)
    dens = model.eval_pred_dens([-2, 0, 2], test[x_cols].iloc[[0]], **kw_row)
    assert dens.shape == (3,) and (dens > 0).all()
    moments = model.calc_pred_moments(test[x_cols], nsamp=100, **kw)
    assert list(moments) == ["means", "vars", "nu"]
    assert moments["means"].shape == (100, 5) and moments["vars"].shape == (100, 5, 5)
    assert moments["nu"].shape == (100,) and (moments["nu"] > 2 + 15).all()

    # Training continues from the last state
    model.fit(n_epochs=10, auto_stop=auto_stop, display_progress=False)
    assert model.loss_stor_.shape == (20,) and model.loss_ <= model.loss_stor_[:10].min()


def test_lpds_is_log_of_eval_pred_dens(r_test_data):
    train, test = r_test_data
    torch.manual_seed(0)
    model = ShrinkTPR(train["y"], train[["x1", "x2"]], device="cpu").fit(
        n_epochs=10, display_progress=False)
    row = test[["x1", "x2"]].iloc[[1]]
    torch.manual_seed(2)
    dens = model.eval_pred_dens(test["y"].iloc[1], row, nsamp=50)
    torch.manual_seed(2)
    assert model.lpds(test["y"].iloc[1], row, nsamp=50) == pytest.approx(np.log(dens[0]),
                                                                          rel=1e-5)
    with pytest.raises(ValueError, match="exactly one row"):
        model.eval_pred_dens(0.0, test[["x1", "x2"]])


def test_fit_on_default_device(r_test_data):
    train, test = r_test_data
    model = ShrinkTPR(train["y"], train[["x1", "x2"]]).fit(n_epochs=5, display_progress=False)
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert model.model_.x.device.type == expected
    assert np.isfinite(model.predict(test[["x1", "x2"]], nsamp=3)).all()
