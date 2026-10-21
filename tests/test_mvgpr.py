import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import special, stats

from shrinkgpr._mvgpr_module import MVGPRModule, bias_diag, ldnorm_multi, make_corr_chol
from shrinkgpr.flows import iaf, nsf_cl, sylvester
from shrinkgpr.kernels import kernel_matern_32, kernel_se
from shrinkgpr.mvgpr import ShrinkMVGPR
from shrinkgpr.sim_mvgpr import rlkjcorr, sim_mvgpr


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


def fitted(y, x, n_epochs=10, **kwargs):
    torch.manual_seed(0)
    return ShrinkMVGPR(y, x, device="cpu", **kwargs).fit(n_epochs=n_epochs,
                                                         display_progress=False)


def module64(M=3, p=2, N=8, seed=0, n_layers=2):
    """A float64 module with perturbed flow parameters."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    y = torch.tensor(rng.normal(size=(N, M)))
    x = torch.tensor(rng.normal(size=(N, p)))
    m = MVGPRModule(y.float(), x.float(), n_layers=n_layers, flow_func=sylvester,
                    kernel_func=kernel_se).double()
    with torch.no_grad():
        for prm in m.parameters():
            prm.add_(0.1 * torch.randn_like(prm))
    return m


# Construction


def test_construction_and_latent_layout(data):
    y_tr, x_tr, _, _ = train_test(data)
    model = ShrinkMVGPR(y_tr, x_tr, device="cpu")
    assert not model.is_fitted and "not fitted" in repr(model)
    assert model.y_names_ == ["y1", "y2"] and model.x_names_ == ["x1", "x2"]
    m = model._module
    # M(M-1)/2 + M-1 + p + 3
    assert m.d == 1 + 1 + 2 + 3 and m.n_omega == 2
    assert m.positive.tolist() == [False, False, True, True, True, True, True]
    assert m.y.shape == (15, 2) and m.y.dtype == torch.float32
    four = ShrinkMVGPR(np.random.default_rng(0).normal(size=(10, 4)),
                       np.ones((10, 3)), device="cpu")
    assert four._module.d == 6 + 3 + 3 + 3 and four.y_names_ == ["y1", "y2", "y3", "y4"]


@pytest.mark.parametrize(
    "kwargs, err, match",
    [
        ({"eta": 0}, ValueError, "eta must be positive"),
        ({"a_Om": -1, "c_Om": "1"}, ValueError, "a_Om, c_Om must all be positive"),
        ({"n_layers": 0}, ValueError, "n_layers"),
        ({"flow_func": None}, TypeError, "flow_func"),
    ],
)
def test_invalid_arguments(data, kwargs, err, match):
    with pytest.raises(err, match=match):
        ShrinkMVGPR(*data, device="cpu", **kwargs)


def test_invalid_data(data):
    y, x = data
    with pytest.raises(ValueError, match="at least 2 columns"):
        ShrinkMVGPR(y["y1"], x)
    with pytest.raises(ValueError, match="same number of rows"):
        ShrinkMVGPR(y.iloc[:-1], x)
    y_na = y.copy()
    y_na.iloc[0, 1] = np.nan
    with pytest.raises(ValueError, match="No NA values are allowed in 'y'"):
        ShrinkMVGPR(y_na, x)


# Building blocks


@pytest.mark.parametrize("M", [2, 3, 4])
def test_make_corr_chol_gives_a_correlation_factor(M):
    torch.manual_seed(M)
    u = torch.randn(5, M * (M - 1) // 2, dtype=torch.float64)
    L, _ = make_corr_chol(u, M)
    assert torch.equal(L, torch.tril(L))
    corr = L @ L.transpose(1, 2)
    torch.testing.assert_close(torch.diagonal(corr, dim1=1, dim2=2),
                               torch.ones(5, M, dtype=torch.float64))
    assert (torch.linalg.eigvalsh(corr) > 0).all()


@pytest.mark.parametrize("M", [2, 3, 4])
def test_make_corr_chol_log_jacobian(M):
    """logJ is log|det| of the map to the strictly lower triangle of L."""
    P = M * (M - 1) // 2
    rows, cols = torch.tril_indices(M, M, offset=-1)
    torch.manual_seed(10 + M)
    u = torch.randn(3, P, dtype=torch.float64)
    _, logJ = make_corr_chol(u, M)
    for b in range(3):
        def lower(v):
            return make_corr_chol(v.unsqueeze(0), M)[0][0, rows, cols]

        jac = torch.autograd.functional.jacobian(lower, u[b])
        torch.testing.assert_close(logJ[b], torch.linalg.slogdet(jac).logabsdet)


def test_bias_diag():
    L = torch.tensor([[[0.5, 0.0], [0.3, -0.2]]], dtype=torch.float64)
    L2, diag = bias_diag(L)
    torch.testing.assert_close(diag, torch.tensor([[0.5, -0.2]], dtype=torch.float64))
    expected = 0.03 + torch.nn.functional.softplus(diag - 0.03, beta=10)
    torch.testing.assert_close(torch.diagonal(L2, dim1=1, dim2=2), expected)
    assert L2[0, 1, 0] == 0.3 and L2[0, 0, 1] == 0


def test_ldnorm_multi_matches_matrix_normal():
    rng = np.random.default_rng(1)
    N, M = 6, 3
    y = rng.normal(size=(N, M))
    rows, cols = [], []
    for _ in range(4):
        a = rng.normal(size=(N, N))
        b = rng.normal(size=(M, M))
        rows.append(a @ a.T + N * np.eye(N))
        cols.append(b @ b.T + np.eye(M))
    L_K = torch.linalg.cholesky(torch.tensor(np.array(rows)))
    L_Om = torch.linalg.cholesky(torch.tensor(np.array(cols)))
    ll = ldnorm_multi(L_K, L_Om, torch.tensor(y))
    expected = [stats.matrix_normal(np.zeros((N, M)), r, c).logpdf(y) for r, c in
                zip(rows, cols, strict=True)]
    np.testing.assert_allclose(ll.numpy() - 0.5 * N * M * math.log(2 * math.pi), expected)


def test_log_prior_by_hand():
    m = module64()
    zk, _ = m(m.gen_batch(5))
    params = m.unpack(zk)
    omega = m.omega_factor(params)
    prior = m.log_prior(params, omega).item()

    def ltg(x, a, c, tau):
        tau = tau[:, None]
        return -0.5 * np.log(tau) - 0.5 * np.log(x) + np.log(
            special.hyperu(c + 0.5, 1.5 - a, a * x / (2 * c * tau)))

    def ldf(x, d1, d2):
        return (d1 / 2 - 1) * np.log(x) - (d1 + d2) / 2 * np.log1p(d1 / d2 * x)

    th, tau, tau_om = (params[k].detach().numpy() for k in ("thetas", "tau", "tau_Om"))
    s, s2 = omega["S_diag"].detach().numpy(), params["sigma2"].detach().numpy()
    corr = (omega["L_D"] @ omega["L_D"].transpose(1, 2)).detach().numpy()
    expected = (ltg(th, 0.5, 0.5, tau).sum(1).mean() + ldf(tau, 1, 1).mean()
                + (4 - 1) * np.linalg.slogdet(corr)[1].mean()
                + ltg(s, 0.5, 0.5, tau_om).sum(1).mean() + ldf(tau_om, 1, 1).mean()
                + (np.log(10) - 10 * s2).mean())
    assert prior == pytest.approx(expected, rel=1e-10)
    # Scales have product 1
    np.testing.assert_allclose(s.prod(axis=1), 1.0)


def test_gen_batch_scaling():
    m = module64(M=3, p=2)
    torch.manual_seed(0)
    z = m.gen_batch(20000)
    q = m.n_omega  # 5
    assert z.shape == (20000, m.d)
    np.testing.assert_allclose(z[:, :q].std(0).numpy(), 0.2 / math.sqrt(q), rtol=0.03)
    np.testing.assert_allclose(z[:, q : q + 2].mean(0).numpy(), -2, atol=0.03)
    np.testing.assert_allclose(z[:, q + 2 :].std(0).numpy(), 1, rtol=0.03)


def test_elbo_and_gradients():
    m = module64()
    zk, log_det_J = m(m.gen_batch(6))
    elbo = m.elbo(zk, log_det_J)
    assert torch.isfinite(elbo)
    elbo.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


# Prediction against direct computations


def replayed(m, nsamp, seed=3):
    torch.manual_seed(seed)
    z = m.gen_batch(nsamp)
    m.gen_batch = lambda n: z
    with torch.no_grad():
        params = m.unpack(m(z)[0], sigma2_tol=1e-6)
        omega = m.omega_factor(params)
    Omega = (omega["L_Om"] @ omega["L_Om"].transpose(1, 2)).numpy()
    return params, Omega


def test_calc_pred_moments_by_hand():
    m = module64(N=7)
    x_new = torch.tensor(np.random.default_rng(5).normal(size=(3, 2)))
    params, Omega = replayed(m, 4)
    mom = m.calc_pred_moments(x_new, 4)
    assert mom["means"].shape == (4, 3, 3) and mom["K"].shape == (4, 3, 3)
    th, tau, s2 = params["thetas"], params["tau"], params["sigma2"]
    K = kernel_se(th, tau, m.x).numpy()
    Ks = kernel_se(th, tau, m.x, x_new).numpy()
    Kss = kernel_se(th, tau, x_new).numpy()
    y = m.y.numpy()
    for s in range(4):
        A = K[s] + s2[s].item() * np.eye(7)
        np.testing.assert_allclose(mom["means"][s].numpy(), Ks[s] @ np.linalg.solve(A, y))
        np.testing.assert_allclose(mom["K"][s].numpy(),
                                   Kss[s] - Ks[s] @ np.linalg.solve(A, Ks[s].T)
                                   + s2[s].item() * np.eye(3), atol=1e-12)
    np.testing.assert_allclose(mom["Omega"].numpy(), Omega)


def test_eval_pred_dens_matches_scipy():
    m = module64(N=7)
    x_new = torch.tensor([[0.3, -0.1]], dtype=torch.float64)
    replayed(m, 5)
    mom = m.calc_pred_moments(x_new, 5)
    y_eval = torch.tensor([[0.0, 0.1, -0.2], [1.0, -1.0, 0.5]], dtype=torch.float64)
    dens = m.eval_pred_dens(y_eval, x_new, 5)
    expected = np.mean([stats.multivariate_normal(mom["means"][s, 0].numpy(),
                                                  mom["K"][s, 0, 0].item()
                                                  * mom["Omega"][s].numpy()).pdf(y_eval.numpy())
                        for s in range(5)], axis=0)
    np.testing.assert_allclose(dens.numpy(), expected)
    np.testing.assert_allclose(m.eval_pred_dens(y_eval, x_new, 5, log=True).numpy(),
                               np.log(expected))


def test_predict_moments_monte_carlo():
    """With one posterior draw repeated, draws have mean `means` and covariance K kron Omega."""
    m = module64(M=2, N=7)
    x_new = torch.tensor([[0.3, -0.1], [1.0, 0.5]], dtype=torch.float64)
    torch.manual_seed(0)
    z = m.gen_batch(1).expand(40000, -1)
    m.gen_batch = lambda n: z[:n]
    mom = m.calc_pred_moments(x_new, 1)
    draws = m.predict(x_new, 40000).reshape(40000, -1).numpy()
    from shrinkgpr._utils import robust_chol
    L_Om, _ = bias_diag(robust_chol(mom["Omega"]))
    Om_used = (L_Om @ L_Om.transpose(1, 2))[0].numpy()  # R biases the factor again
    cov = np.kron(mom["K"][0].numpy(), Om_used)
    np.testing.assert_allclose(draws.mean(0), mom["means"][0].numpy().ravel(),
                               atol=4 * np.sqrt(cov.diagonal().max() / 40000))
    np.testing.assert_allclose(np.cov(draws.T), cov, atol=0.05 * np.abs(cov).max())


# Fitting


@pytest.mark.parametrize("display_progress", [True, False])
@pytest.mark.parametrize("flow", [sylvester, nsf_cl, iaf])
@pytest.mark.parametrize("kernel", [kernel_se, kernel_matern_32])
@pytest.mark.parametrize("auto_stop", [True, False])
def test_r_scenarios(data, auto_stop, kernel, flow, display_progress):
    """The scenarios of tests/testthat/test_shrinkMVGPR.R."""
    y_tr, x_tr, _, x_te = train_test(data)
    torch.manual_seed(123)
    model = ShrinkMVGPR(y_tr, x_tr, kernel_func=kernel, flow_func=flow, device="cpu")
    model.fit(n_epochs=10, auto_stop=auto_stop, display_progress=display_progress)
    assert model.is_fitted and isinstance(model.model_, MVGPRModule)

    preds = model.predict(x_te)
    assert preds.shape == (100, 5, 2) and preds.dtype == np.float32

    lpds = model.lpds(data[0].iloc[15].to_numpy(), x_te.iloc[[0]])
    assert isinstance(lpds, float)

    eval_points = np.column_stack([[-2, 0, 2], [-2, 0, 2]])
    dens = model.eval_pred_dens(eval_points, x_te.iloc[[0]])
    assert dens.shape == (3,) and (dens >= 0).all()

    moments = model.calc_pred_moments(x_te, nsamp=100)
    assert list(moments) == ["means", "K", "Omega"]
    assert moments["means"].shape == (100, 5, 2)
    assert moments["K"].shape == (100, 5, 5)
    assert moments["Omega"].shape == (100, 2, 2)

    model.fit(n_epochs=10, auto_stop=auto_stop, display_progress=False)
    assert model.loss_stor_.shape == (20,)


def test_r_defaults_for_training(data, monkeypatch):
    """Adam with lr=1e-4 and gradient clipping to norm 0.5 in every step."""
    calls = []
    original = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *args, **kwargs):
        calls.append(max_norm)
        return original(params, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    model = fitted(*train_test(data)[:2], n_epochs=4)
    assert calls == [0.5] * 4
    group = model.optimizer_.param_groups[0]
    assert group["lr"] == 1e-4 and group["weight_decay"] == 1e-3
    torch.manual_seed(0)
    model = ShrinkMVGPR(*train_test(data)[:2], device="cpu")
    model.fit(n_epochs=1, display_progress=False, optim_control={"lr": 0.01})
    assert model.optimizer_.param_groups[0]["lr"] == 0.01


def test_fit_again_continues_training(data):
    model = fitted(*train_test(data)[:2])
    optimizer = model.optimizer_
    model.fit(n_epochs=5, display_progress=False)
    assert model.optimizer_ is optimizer and model.loss_stor_.shape == (15,)
    assert model.loss_ <= model.loss_stor_[:10].min()


def test_fit_is_reproducible_with_torch_seed(data):
    a = fitted(*train_test(data)[:2], flow_func=nsf_cl)
    b = fitted(*train_test(data)[:2], flow_func=nsf_cl)
    np.testing.assert_array_equal(a.loss_stor_, b.loss_stor_)


def test_prediction_inputs(data):
    y_tr, x_tr, y_te, x_te = train_test(data)
    model = fitted(y_tr, x_tr)
    assert model.predict(nsamp=3).shape == (3, 15, 2)
    # Columns are matched by name; a 1-D array is one row
    assert model.predict(x_te[["x2", "x1"]], nsamp=2).shape == (2, 5, 2)
    assert model.predict(x_te.iloc[0].to_numpy(), nsamp=2).shape == (2, 1, 2)
    # A 1-D y_eval is one point
    assert model.eval_pred_dens(y_te.iloc[0].to_numpy(), x_te.iloc[[0]], nsamp=3).shape == (1,)
    with pytest.raises(ValueError, match="exactly one row"):
        model.eval_pred_dens([0.0, 0.0], x_te, nsamp=3)
    with pytest.raises(ValueError, match="2 columns"):
        model.eval_pred_dens(np.zeros((2, 3)), x_te.iloc[[0]], nsamp=3)
    with pytest.raises(ValueError, match="length 2"):
        model.lpds([0.0, 1.0, 2.0], x_te.iloc[[0]])
    with pytest.raises(RuntimeError, match="not fitted"):
        ShrinkMVGPR(y_tr, x_tr, device="cpu").predict()


# Simulation


def test_rlkjcorr():
    torch.manual_seed(0)
    for M in (2, 3, 5):
        R = rlkjcorr(M)
        np.testing.assert_allclose(np.diag(R), 1.0)
        np.testing.assert_allclose(R, R.T)
        assert np.linalg.eigvalsh(R).min() > 0


def test_rlkjcorr_marginal():
    """Under LKJ(eta), each correlation r satisfies (r + 1)/2 ~ Beta(b, b), b = eta - 1 + M/2."""
    torch.manual_seed(1)
    M, eta = 4, 2.0
    r = np.array([rlkjcorr(M, eta)[2, 0] for _ in range(3000)])
    b = eta - 1 + M / 2
    assert stats.kstest((r + 1) / 2, stats.beta(b, b).cdf).pvalue > 0.001


def test_sim_mvgpr():
    torch.manual_seed(123)
    sim = sim_mvgpr(N=30, M=3, d=4, device="cpu")
    assert list(sim["data"].columns) == ["y.1", "y.2", "y.3", "x.1", "x.2", "x.3", "x.4"]
    assert sim["data"].shape == (30, 7)
    tv = sim["true_vals"]
    assert (tv["theta"] == 0).sum() == 2 and tv["Omega"].shape == (3, 3)
    assert np.linalg.eigvalsh(tv["Omega"]).min() > 0
    torch.manual_seed(123)
    pd.testing.assert_frame_equal(sim_mvgpr(N=30, M=3, d=4, device="cpu")["data"], sim["data"])
    fixed = sim_mvgpr(N=10, theta=[1.0, 0.0, 0.5], Omega=np.eye(2), rho=0.5, device="cpu")
    np.testing.assert_array_equal(fixed["true_vals"]["theta"], [1.0, 0.0, 0.5])
    with pytest.raises(ValueError, match="positive definite"):
        sim_mvgpr(Omega=np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(ValueError, match="at least 2"):
        sim_mvgpr(M=1)
