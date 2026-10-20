import numpy as np
import pytest
import torch
from scipy import integrate, stats

from shrinkgpr._gpr_module import GPRModule, ldnorm
from shrinkgpr.flows import iaf, nsf_cl, sylvester
from shrinkgpr.kernels import kernel_matern_32, kernel_se

F64 = torch.float64


def make_module(with_mean=False, dtype=F64, flow=sylvester, kernel=kernel_se, n_layers=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(12, 3, generator=g, dtype=dtype)
    y = torch.sin(x[:, 0]) + 0.1 * torch.randn(12, generator=g, dtype=dtype)
    x_mean = torch.cat([torch.ones(12, 1, dtype=dtype), x[:, :1]], dim=1) if with_mean else None
    torch.manual_seed(seed)
    module = GPRModule(y, x, x_mean, n_layers=n_layers, flow_func=flow, kernel_func=kernel)
    return module.to(dtype)


@pytest.mark.parametrize("with_mean, d", [(False, 3 + 2), (True, 3 + 2 + 3)])
def test_latent_layout(with_mean, d):
    module = make_module(with_mean)
    assert module.d == d
    zk, log_det_J = module(module.gen_batch(50))
    assert zk.shape == (50, d) and log_det_J.shape == (50,)
    assert (zk[:, module.positive] > 0).all()
    params = module.unpack(zk)
    assert params["thetas"].shape == (50, 3)
    if with_mean:
        # betas are unconstrained and sit between tau and tau_mean
        assert not module.positive[5:7].any() and module.positive[7]
        assert (params["betas"] < 0).any()
        assert params["tau_mean"].shape == (50,)
    else:
        assert params["betas"] is None and module.positive.all()


@pytest.mark.parametrize("with_mean", [False, True])
@pytest.mark.parametrize("flow", [sylvester, iaf, nsf_cl])
def test_forward_log_det_matches_autograd_jacobian(with_mean, flow):
    module = make_module(with_mean, flow=flow)
    # Larger perturbations of stacked splines contract volume by ~e^-60, and
    # slogdet of the composite Jacobian is then inaccurate even in float64
    scale = 0.05 if flow is nsf_cl else 0.3
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn_like(p))
    z = 2 * module.gen_batch(5)
    _, log_det_J = module(z)
    J = torch.stack([
        torch.autograd.functional.jacobian(lambda r: module(r.unsqueeze(0))[0][0], zi) for zi in z
    ])
    tol = 1e-3 if flow is nsf_cl else 1e-6  # see test_flows for the spline tolerance
    torch.testing.assert_close(log_det_J, torch.linalg.slogdet(J).logabsdet, rtol=tol, atol=tol)


@pytest.mark.parametrize("with_mean", [False, True])
def test_ldnorm_matches_multivariate_normal(with_mean):
    torch.manual_seed(1)
    B, N = 4, 6
    x = torch.randn(N, 2, dtype=F64)
    K = kernel_se(torch.rand(B, 2, dtype=F64), torch.rand(B, dtype=F64) + 0.5, x)
    sigma2 = torch.rand(B, dtype=F64) + 0.1
    y = torch.randn(N, 1, dtype=F64)
    x_mean = torch.randn(N, 2, dtype=F64) if with_mean else None
    beta = torch.randn(B, 2, dtype=F64) if with_mean else None

    got = ldnorm(K, None, sigma2, y, x_mean, beta)
    cov = K + sigma2.view(B, 1, 1) * torch.eye(N, dtype=F64)
    loc = (x_mean @ beta.t()).t() if with_mean else torch.zeros(B, N, dtype=F64)
    mvn = torch.distributions.MultivariateNormal(loc, covariance_matrix=cov)
    expected = mvn.log_prob(y.squeeze(1)) + 0.5 * N * np.log(2 * np.pi)
    torch.testing.assert_close(got, expected)
    # Passing the Cholesky factor gives the same value
    torch.testing.assert_close(ldnorm(K, torch.linalg.cholesky(cov), sigma2, y, x_mean, beta), got)


def log_mixture_density(x, a, c, tau, normal):
    """log p(x | tau) from the hierarchy behind the triple gamma / NGG priors.

    xi = (c tau / a) G_a / G_c (a scaled beta prime). Given xi, beta ~ N(0, xi)
    for the NGG prior and theta = beta^2 ~ Gamma(1/2, rate 1/(2 xi)) for the
    triple gamma prior.
    """
    def integrand(u):
        xi = np.exp(u)
        if normal:
            cond = stats.norm.pdf(x, scale=np.sqrt(xi))
        else:
            cond = stats.gamma.pdf(x, 0.5, scale=2 * xi)
        return cond * stats.betaprime.pdf(xi, a, c, scale=c * tau / a) * xi

    val, _ = integrate.quad(integrand, -60, 60, limit=1000, epsabs=0, epsrel=1e-10)
    return np.log(val)


@pytest.mark.parametrize("a, c", [(0.5, 0.5), (0.2, 0.3), (1.0, 2.0)])
@pytest.mark.parametrize("prior", ["ltg", "ngg"])
def test_shrinkage_priors_match_hierarchy(prior, a, c):
    # Grid over both the argument and tau; differences cancel the normalizing
    # constant, which depends on neither
    pts = [(0.05, 0.5), (0.3, 1.0), (1.0, 2.0), (2.5, 0.7), (0.8, 3.0)]
    x = torch.tensor([[p[0]] for p in pts], dtype=F64)
    tau = torch.tensor([p[1] for p in pts], dtype=F64)
    fn = getattr(GPRModule, prior)
    got = fn(x, torch.tensor(a, dtype=F64), torch.tensor(c, dtype=F64), tau).squeeze(1).numpy()
    expected = np.array([log_mixture_density(xi, a, c, ti, prior == "ngg") for xi, ti in pts])
    np.testing.assert_allclose(got - got[0], expected - expected[0], atol=1e-6)


def test_f_and_exponential_priors_match_scipy():
    x = torch.tensor([0.1, 0.5, 2.0, 7.0], dtype=F64)
    d1, d2 = torch.tensor(1.0, dtype=F64), torch.tensor(3.0, dtype=F64)
    got = GPRModule.ldf(x, d1, d2).numpy()
    expected = stats.f.logpdf(x.numpy(), 1.0, 3.0)
    np.testing.assert_allclose(got - got[0], expected - expected[0])
    rate = torch.tensor(10.0, dtype=F64)
    np.testing.assert_allclose(
        GPRModule.lexp(x, rate).numpy(), stats.expon.logpdf(x.numpy(), scale=0.1)
    )


@pytest.mark.parametrize("with_mean", [False, True])
def test_elbo_is_sum_of_components(with_mean):
    module = make_module(with_mean)
    zk, log_det_J = module(module.gen_batch(8))
    params = module.unpack(zk)
    K = module.kernel_func(params["thetas"], params["tau"], module.x)
    expected = (module.log_likelihood(K, None, params).mean() + module.log_prior(params)
                + log_det_J.mean())
    torch.testing.assert_close(module.elbo(zk, log_det_J), expected)


@pytest.mark.parametrize("with_mean", [False, True])
@pytest.mark.parametrize("flow", [sylvester, iaf, nsf_cl])
@pytest.mark.parametrize("kernel", [kernel_se, kernel_matern_32])
def test_elbo_backprop_float32_on_device(with_mean, flow, kernel):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = make_module(with_mean, dtype=torch.float32, flow=flow, kernel=kernel).to(device)
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


def test_elbo_nan_raises():
    module = make_module()
    zk, log_det_J = module(module.gen_batch(4))
    with pytest.raises(RuntimeError, match="ELBO is NaN"):
        module.elbo(zk, log_det_J * float("nan"))
