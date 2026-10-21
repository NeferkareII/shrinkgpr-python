import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import special, stats

from shrinkgpr._mvgpr_module import MVGPRModule
from shrinkgpr._mvtpr_module import MVTPRModule, ldt_multi, sample_matrix_t
from shrinkgpr.flows import iaf, nsf_cl, sylvester
from shrinkgpr.kernels import kernel_matern_32, kernel_se
from shrinkgpr.mvgpr import ShrinkMVGPR
from shrinkgpr.mvtpr import ShrinkMVTPR


@pytest.fixture
def data():
    """The mock data of the R tests: 20 rows, the first 15 for training."""
    rng = np.random.default_rng(123)
    df = pd.DataFrame({
        "y1": np.sin(2 * np.pi * rng.uniform(size=20)) + rng.normal(scale=0.1, size=20),
        "y2": np.cos(2 * np.pi * rng.uniform(size=20)) + rng.normal(scale=0.1, size=20),
        "x1": rng.uniform(size=20),
        "x2": rng.normal(size=20),
    })
    return df[["y1", "y2"]], df[["x1", "x2"]]


def train_test(data):
    y, x = data
    return y.iloc[:15], x.iloc[:15], y.iloc[15:], x.iloc[15:]


def module64(M=3, p=2, N=8, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    y = torch.tensor(rng.normal(size=(N, M)), dtype=torch.float32)
    x = torch.tensor(rng.normal(size=(N, p)), dtype=torch.float32)
    m = MVTPRModule(y, x, n_layers=2, flow_func=sylvester, kernel_func=kernel_se).double()
    with torch.no_grad():
        for prm in m.parameters():
            prm.add_(0.1 * torch.randn_like(prm))
    return m


def spd(rng, n, shift):
    a = rng.normal(size=(n, n))
    return a @ a.T + shift * np.eye(n)


def matrix_t_logpdf(Y, row, col, nu):
    """Matrix-t log density (Gupta and Nagar, Def. 4.2.1) with Gamma_p."""
    n, p = Y.shape
    A = np.eye(n) + np.linalg.solve(row, Y) @ np.linalg.solve(col, Y.T)
    return (special.multigammaln((nu + n + p - 1) / 2, p)
            - special.multigammaln((nu + p - 1) / 2, p)
            - n * p / 2 * np.log(np.pi)
            - p / 2 * np.linalg.slogdet(row)[1] - n / 2 * np.linalg.slogdet(col)[1]
            - (nu + n + p - 1) / 2 * np.linalg.slogdet(A)[1])


# Construction


def test_construction_and_latent_layout(data):
    y_tr, x_tr, _, _ = train_test(data)
    model = ShrinkMVTPR(y_tr, x_tr, nu_alpha=1.5, nu_beta=0.5, device="cpu")
    assert isinstance(model, ShrinkMVGPR)
    m = model._module
    assert isinstance(m, MVTPRModule) and isinstance(m, MVGPRModule)
    # M(M-1)/2 + M-1 + p + 3 + nu
    assert m.d == 1 + 1 + 2 + 3 + 1
    assert m.positive.tolist() == [False, False] + [True] * 6
    assert m.nu_alpha.item() == 1.5 and m.nu_beta.item() == 0.5
    with pytest.raises(ValueError, match="nu_alpha must be positive"):
        ShrinkMVTPR(y_tr, x_tr, nu_alpha=0, device="cpu")


# Likelihood and priors


@pytest.mark.parametrize("N, M", [(1, 3), (4, 1), (5, 3), (3, 2)])
def test_ldt_multi_matches_matrix_t(N, M):
    """ldt_multi is the matrix-t log density plus NM/2 log(pi)."""
    rng = np.random.default_rng(N * 10 + M)
    Y = rng.normal(size=(N, M))
    rows = np.array([spd(rng, N, N) for _ in range(3)])
    cols = np.array([spd(rng, M, 1) for _ in range(3)])
    nu = np.array([2.5, 4.0, 11.0])
    ll = ldt_multi(torch.linalg.cholesky(torch.tensor(rows)),
                   torch.linalg.cholesky(torch.tensor(cols)), torch.tensor(Y),
                   torch.tensor(nu)).numpy()
    expected = [matrix_t_logpdf(Y, r, c, v) for r, c, v in zip(rows, cols, nu, strict=True)]
    np.testing.assert_allclose(ll - 0.5 * N * M * np.log(np.pi), expected)
    if N == 1 or M == 1:
        # A single row or column is multivariate t with scale row kron col / nu
        mvt = [stats.multivariate_t(np.zeros(N * M), np.kron(r, c) / v, df=v).logpdf(Y.ravel())
               for r, c, v in zip(rows, cols, nu, strict=True)]
        np.testing.assert_allclose(expected, mvt)


def test_nu_prior_and_log_prior():
    m = module64()
    zk, _ = m(m.gen_batch(5))
    params = m.unpack(zk)
    # nu is the softplus output plus 2, and the gamma prior is on nu - 2
    assert params["nu"].shape == (5,) and torch.equal(params["nu"], zk[:, -1] + 2)
    omega = m.omega_factor(params)
    nu_raw = params["nu"] - 2
    ldg = (stats.gamma(0.5, scale=1 / 2).logpdf(nu_raw.detach().numpy())
           - stats.gamma(0.5, scale=1 / 2).logpdf(1.0))
    np.testing.assert_allclose(m.ldg(nu_raw, m.nu_alpha, m.nu_beta).detach().numpy()
                               - m.ldg(torch.tensor(1.0), m.nu_alpha, m.nu_beta).numpy(), ldg)
    diff = m.log_prior(params, omega) - MVGPRModule.log_prior(m, params, omega)
    assert diff.item() == pytest.approx(m.ldg(nu_raw, m.nu_alpha, m.nu_beta).mean().item())


def test_elbo_and_gradients():
    m = module64()
    zk, log_det_J = m(m.gen_batch(6))
    elbo = m.elbo(zk, log_det_J)
    assert torch.isfinite(elbo)
    elbo.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


# Prediction


def test_calc_pred_moments_adds_posterior_scale_and_dof():
    m = module64(N=7)
    x_new = torch.tensor([[0.3, -0.1], [0.0, 1.0]], dtype=torch.float64)
    torch.manual_seed(2)
    z = m.gen_batch(4)
    m.gen_batch = lambda n: z
    mom = m.calc_pred_moments(x_new, 4)
    gauss = MVGPRModule.calc_pred_moments(m, x_new, 4)
    assert list(mom) == ["means", "K", "Omega", "nu"] and mom["nu"].shape == (4,)
    torch.testing.assert_close(mom["means"], gauss["means"])
    torch.testing.assert_close(mom["K"], gauss["K"])
    with torch.no_grad():
        params = m.unpack(m(z)[0], sigma2_tol=1e-4)
    torch.testing.assert_close(mom["nu"], params["nu"] + 7)
    K = kernel_se(params["thetas"], params["tau"], m.x).numpy()
    for s in range(4):
        A = K[s] + params["sigma2"][s].item() * np.eye(7)
        y = m.y.numpy()
        np.testing.assert_allclose(mom["Omega"][s].numpy(),
                                   gauss["Omega"][s].numpy() + y.T @ np.linalg.solve(A, y))


def test_eval_pred_dens_matches_scipy():
    m = module64(N=7)
    x_new = torch.tensor([[0.3, -0.1]], dtype=torch.float64)
    torch.manual_seed(4)
    z = m.gen_batch(5)
    m.gen_batch = lambda n: z
    mom = m.calc_pred_moments(x_new, 5)
    y_eval = torch.tensor([[0.0, 0.1, -0.2], [2.0, -1.0, 0.5]], dtype=torch.float64)
    dens = m.eval_pred_dens(y_eval, x_new, 5)
    expected = np.mean([
        stats.multivariate_t(mom["means"][s, 0].numpy(),
                             mom["K"][s, 0, 0].item() * mom["Omega"][s].numpy()
                             / mom["nu"][s].item(),
                             df=mom["nu"][s].item()).pdf(y_eval.numpy())
        for s in range(5)], axis=0)
    np.testing.assert_allclose(dens.numpy(), expected)


def test_sample_matrix_t_moments():
    """As mniw::rMT: V ~ InvWishart(col, nu), so Cov(vec X) = row kron col / (nu - m - 1)."""
    rng = np.random.default_rng(0)
    n, m, B = 2, 3, 200000
    row, col = spd(rng, n, 1), spd(rng, m, 1)
    mean = rng.normal(size=(n, m))
    nu = 9.0
    torch.manual_seed(0)
    X = sample_matrix_t(torch.tensor(mean).expand(B, n, m), torch.tensor(row).expand(B, n, n),
                        torch.tensor(col).expand(B, m, m),
                        torch.full((B,), nu, dtype=torch.float64))
    flat = X.reshape(B, -1).numpy()
    cov = np.kron(row, col) / (nu - m - 1)
    np.testing.assert_allclose(flat.mean(0), mean.ravel(), atol=0.02 * np.sqrt(cov.max()))
    np.testing.assert_allclose(np.cov(flat.T), cov, atol=0.03 * np.abs(cov).max())
    # Each entry is univariate t with nu - m + 1 degrees of freedom and scale
    # sqrt(row_ii col_jj / (nu - m + 1))
    df = nu - m + 1
    entry = (flat[:, 0] - mean[0, 0]) / np.sqrt(row[0, 0] * col[0, 0] / df)
    assert stats.kstest(entry[:20000], stats.t(df).cdf).pvalue > 1e-3


def test_predict_draws_follow_eval_pred_dens():
    """With a single posterior draw, every entry of a test row is t with nu
    degrees of freedom and scale sqrt(k Omega_jj / nu), as in eval_pred_dens."""
    m = module64(M=3, N=7)
    x_new = torch.tensor([[0.3, -0.1]], dtype=torch.float64)
    torch.manual_seed(3)
    z = m.gen_batch(1)
    m.gen_batch = lambda n: z.repeat(n, 1)
    mom = m.calc_pred_moments(x_new, 1)
    nu, k = mom["nu"][0].item(), mom["K"][0, 0, 0].item()
    torch.manual_seed(4)
    draws = m.predict(x_new, 20000)[:, 0, :].numpy()
    for j in range(3):
        scale = math.sqrt(k * mom["Omega"][0, j, j].item() / nu)
        entry = (draws[:, j] - mom["means"][0, 0, j].item()) / scale
        assert stats.kstest(entry, stats.t(nu).cdf).pvalue > 1e-3
    # The variance is k Omega / (nu - 2), not k Omega / (nu - M - 1)
    np.testing.assert_allclose(np.cov(draws.T), k * mom["Omega"][0].numpy() / (nu - 2),
                               rtol=0.1, atol=0.02 * k * mom["Omega"][0].abs().max().item())


# Fitting


@pytest.mark.parametrize("display_progress", [True, False])
@pytest.mark.parametrize("flow", [sylvester, nsf_cl, iaf])
@pytest.mark.parametrize("kernel", [kernel_se, kernel_matern_32])
@pytest.mark.parametrize("auto_stop", [True, False])
def test_r_scenarios(data, auto_stop, kernel, flow, display_progress):
    """The scenarios of tests/testthat/test_shrinkMVTPR.R."""
    y_tr, x_tr, y_te, x_te = train_test(data)
    torch.manual_seed(123)
    model = ShrinkMVTPR(y_tr, x_tr, kernel_func=kernel, flow_func=flow, device="cpu")
    model.fit(n_epochs=10, auto_stop=auto_stop, display_progress=display_progress)
    assert model.is_fitted and isinstance(model.model_, MVTPRModule)

    preds = model.predict(x_te)
    assert preds.shape == (100, 5, 2) and np.isfinite(preds).all()

    lpds = model.lpds(y_te.iloc[0].to_numpy(), x_te.iloc[[0]])
    assert isinstance(lpds, float) and math.isfinite(lpds)

    eval_points = np.column_stack([[-2, 0, 2], [-2, 0, 2]])
    dens = model.eval_pred_dens(eval_points, x_te.iloc[[0]])
    assert dens.shape == (3,) and (dens >= 0).all()

    moments = model.calc_pred_moments(x_te, nsamp=100)
    assert list(moments) == ["means", "K", "Omega", "nu"]
    assert moments["means"].shape == (100, 5, 2)
    assert moments["K"].shape == (100, 5, 5)
    assert moments["Omega"].shape == (100, 2, 2)
    assert moments["nu"].shape == (100,) and (moments["nu"] > 15).all()

    model.fit(n_epochs=10, auto_stop=auto_stop, display_progress=False)
    assert model.loss_stor_.shape == (20,)


def test_fit_uses_r_training_defaults(data):
    torch.manual_seed(0)
    model = ShrinkMVTPR(*train_test(data)[:2], device="cpu").fit(n_epochs=3,
                                                                 display_progress=False)
    assert model._max_grad_norm == 0.5
    assert model.optimizer_.param_groups[0]["lr"] == 1e-4
    optimizer = model.optimizer_
    model.fit(n_epochs=2, display_progress=False)
    assert model.optimizer_ is optimizer and model.loss_stor_.shape == (5,)
