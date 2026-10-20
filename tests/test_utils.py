import numpy as np
import pytest
import torch
from scipy import stats

from shrinkgpr._utils import lightweight_ols, merge_args, res_protector, robust_chol


def test_lightweight_ols_matches_linregress():
    rng = np.random.default_rng(1)
    x = np.arange(1, 101)
    y = 0.01 * x + rng.normal(size=100)
    assert lightweight_ols(y, x) == pytest.approx(stats.linregress(x, y).pvalue, rel=1e-10)


def random_spd(batch, n, dtype=torch.float64):
    a = torch.randn(batch, n, n, dtype=dtype)
    return a @ a.transpose(1, 2) + n * torch.eye(n, dtype=dtype)


def test_robust_chol_matches_cholesky_on_spd():
    torch.manual_seed(0)
    A = random_spd(4, 6)
    L = robust_chol(A)
    assert torch.allclose(L, torch.linalg.cholesky(A))
    assert torch.allclose(robust_chol(A, upper=True), L.transpose(1, 2))


def test_robust_chol_handles_singular_and_indefinite():
    torch.manual_seed(0)
    v = torch.randn(3, 5, 2, dtype=torch.float32)
    singular = v @ v.transpose(1, 2)  # rank 2
    indefinite = singular - 1e-3 * torch.eye(5)
    for A in (singular, indefinite):
        L = robust_chol(A)
        assert torch.isfinite(L).all()
        assert torch.allclose(L @ L.transpose(1, 2), A, atol=0.05)


def test_res_protector_keeps_sign():
    x = torch.tensor([-1.0, -1e-9, 0.0, 1e-9, 2.0])
    out = res_protector(x, tol=1e-6)
    assert torch.equal(out, torch.tensor([-1.0, -1e-6, 1e-6, 1e-6, 2.0]))


def test_merge_args_warns_on_unknown_and_fills_defaults():
    defaults = {"lr": 1e-3, "weight_decay": 1e-3}
    with pytest.warns(UserWarning, match="lrr has been incorrectly named"):
        merged = merge_args(defaults, {"lrr": 1.0, "weight_decay": 0.0}, "optim_control")
    assert merged == {"lr": 1e-3, "weight_decay": 0.0}
    assert merge_args(defaults, None, "optim_control") == defaults
