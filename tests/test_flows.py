import pytest
import torch

from shrinkgpr.flows import IAF, NSFCL, Sylvester, iaf, nsf_cl, sylvester

FLOWS = [sylvester, iaf, nsf_cl]


def perturbed(flow_cls, d, seed=0, **kwargs):
    """Float64 flow with all parameters moved away from their (often trivial) init."""
    torch.manual_seed(seed)
    flow = flow_cls(d, **kwargs).double()
    with torch.no_grad():
        for p in flow.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return flow


def jacobians(flow, z):
    return torch.stack(
        [torch.autograd.functional.jacobian(lambda r: flow(r.unsqueeze(0))[0][0], zi) for zi in z]
    )


@pytest.mark.parametrize("flow_cls", FLOWS)
@pytest.mark.parametrize("d", [2, 3, 6])
def test_shapes(flow_cls, d):
    flow = flow_cls(d)
    zk, log_diag_j = flow(torch.randn(7, d))
    assert zk.shape == (7, d) and log_diag_j.shape == (7,)


@pytest.mark.parametrize(
    "flow_cls, kwargs",
    [
        (sylvester, {}),
        (sylvester, {"n_householder": 1}),
        (iaf, {}),
        (iaf, {"n_hidden": 2, "hidden_dim": 8}),
        (nsf_cl, {}),
        (nsf_cl, {"K": 4, "B": 2.0}),  # many points land in the identity tails
    ],
)
@pytest.mark.parametrize("d", [2, 5])
def test_log_det_matches_autograd_jacobian(flow_cls, kwargs, d):
    flow = perturbed(flow_cls, d, **kwargs)
    z = 2 * torch.randn(6, d, dtype=torch.float64)
    _, log_diag_j = flow(z)
    expected = torch.linalg.slogdet(jacobians(flow, z)).logabsdet
    # The 1e-8 guards in the spline denominators (kept from R) shift the
    # log-derivative by up to ~1e-4 in flat bins; without them it agrees to 1e-13
    tol = 1e-3 if flow_cls is nsf_cl else 1e-6
    torch.testing.assert_close(log_diag_j, expected, rtol=tol, atol=tol)


def test_iaf_jacobian_is_triangular_under_its_ordering():
    d = 5
    flow = perturbed(iaf, d, n_hidden=2, hidden_dim=12)
    J = jacobians(flow, torch.randn(4, d, dtype=torch.float64))
    deg = flow.perm
    # Output i may depend on input j only if j comes no later in the ordering
    forbidden = deg.unsqueeze(0) > deg.unsqueeze(1)
    assert (J[:, forbidden] == 0).all()
    assert (J[:, ~forbidden] != 0).all()


def test_iaf_rejects_bad_permutation():
    with pytest.raises(ValueError, match="permutation"):
        IAF(3, perm=[0, 0, 1])


def test_init_is_near_identity():
    z = torch.randn(5, 4)
    # IAF starts as z' = sigmoid(forget_bias) * z
    zk, ld = IAF(4, forget_bias=2)(z)
    sig = torch.sigmoid(torch.tensor(2.0))
    torch.testing.assert_close(zk, sig * z)
    torch.testing.assert_close(ld, 4 * torch.log(sig).expand(5))
    # The NSF linear map starts as a pure permutation
    flow = NSFCL(4)
    torch.testing.assert_close(flow.P @ flow.get_L() @ flow.get_U(), flow.P)
    assert (flow.P.sum(0) == 1).all() and (flow.P.sum(1) == 1).all()


def test_sylvester_default_householder():
    assert Sylvester(1).n_householder == 0
    assert Sylvester(3).n_householder == 2
    assert Sylvester(10).n_householder == 5
    zk, ld = Sylvester(1)(torch.randn(3, 1))
    assert zk.shape == (3, 1) and torch.isfinite(ld).all()


@pytest.mark.parametrize("flow_cls", FLOWS)
def test_random_init_follows_torch_seed(flow_cls):
    torch.manual_seed(42)
    a = flow_cls(5)
    torch.manual_seed(42)
    b = flow_cls(5)
    for (name, x), y in zip(a.state_dict().items(), b.state_dict().values(), strict=True):
        assert torch.equal(x, y), name


@pytest.mark.parametrize("flow_cls", FLOWS)
def test_float32_on_device_backprop(flow_cls):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    flow = flow_cls(6).to(device)
    zk, ld = flow(torch.randn(10, 6, device=device))
    (zk.sum() + ld.sum()).backward()
    assert zk.device.type == device and zk.dtype == torch.float32
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in flow.parameters())
