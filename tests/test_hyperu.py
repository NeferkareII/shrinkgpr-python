import math

import mpmath as mp
import numpy as np
import pytest
import torch

from shrinkgpr._hyperu import SMALL_Z, log_hyperu, robust_hyperu

mp.mp.dps = 40

# (a, b) pairs as they arise from the priors: a_U = c + 0.5, b_U = 1.5 - a,
# plus b + 1 for the gradient
AB_PAIRS = [(1.0, 1.0), (1.0, 2.0), (1.0, 1.2), (2.0, 1.1), (0.6, 1.3), (1.0, 2.5),
            (1.0, 0.7), (1.0, 0.0), (1.0, -0.5), (3.0, 0.3)]


def exact_log_u(a, b, z):
    return float(mp.log(mp.hyperu(a, b, z)))


def exact_dlog_u(a, b, z):
    return float(mp.diff(lambda t: mp.log(mp.hyperu(a, b, t)), mp.mpf(z)))


@pytest.mark.parametrize("a, b", AB_PAIRS)
def test_log_u_matches_mpmath_away_from_zero(a, b):
    z = np.logspace(-8, 2, 21)
    f64 = {"dtype": torch.float64}
    got = log_hyperu(torch.tensor(a, **f64), torch.tensor(b, **f64), torch.tensor(z, **f64)).numpy()
    expected = [exact_log_u(a, b, zi) for zi in z]
    np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("a, b", AB_PAIRS)
def test_small_z_expansion_matches_mpmath(a, b):
    z = torch.tensor([1e-10, 1e-12], dtype=torch.float64)
    got = robust_hyperu(a, b, z).numpy()
    expected = [float(mp.hyperu(a, b, zi)) for zi in z.tolist()]
    # For 0 < b < 1 only the constant term is used, which is accurate to O(z^(1 - b))
    rtol = 2e-2 if 0 < b < 1 else 1e-8
    np.testing.assert_allclose(got, expected, rtol=rtol)


@pytest.mark.parametrize("a, b", [(1.0, 1.0), (1.0, 1.2), (2.0, 1.1), (1.0, 2.5), (1.0, 0.0)])
def test_accurate_on_both_sides_of_small_z_threshold(a, b):
    z = [SMALL_Z * (1 - 1e-6), SMALL_Z * (1 + 1e-6)]
    got = robust_hyperu(a, b, torch.tensor(z, dtype=torch.float64)).numpy()
    expected = [float(mp.hyperu(a, b, zi)) for zi in z]
    np.testing.assert_allclose(got, expected, rtol=1e-7)


@pytest.mark.parametrize("a, b", [(1.0, 1.0), (1.0, 0.3), (1.2, 1.4), (2.0, 2.5), (1.0, 0.0)])
def test_gradcheck(a, b):
    z = torch.tensor([1e-3, 0.1, 1.0, 5.0], dtype=torch.float64, requires_grad=True)
    a_t = torch.tensor(a, dtype=torch.float64)
    b_t = torch.tensor(b, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x: log_hyperu(a_t, b_t, x), (z,))


@pytest.mark.parametrize("a, b", [(1.0, 1.0), (1.0, 0.5), (1.0, 1.5)])
def test_gradient_for_small_z(a, b):
    # The gradient uses U(a + 1, b + 1, z): b = 0.5 exercises the 1 < b < 2 branch
    z0 = 1e-10
    z = torch.tensor([z0], dtype=torch.float64, requires_grad=True)
    log_hyperu(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64), z).sum().backward()
    # U(a, b, z) for 0 < b < 1 keeps only the constant term, accurate to O(z^(1 - b))
    rel = 10 * z0 ** (1 - b) if 0 < b < 1 else 1e-6
    assert z.grad.item() == pytest.approx(exact_dlog_u(a, b, z0), rel=rel)


def test_keeps_dtype_and_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = torch.tensor([1e-12, 0.5, 3.0], device=device, requires_grad=True)
    out = log_hyperu(torch.tensor(1.0, device=device), torch.tensor(1.0, device=device), z)
    assert out.dtype == torch.float32 and out.device == z.device
    out.sum().backward()
    assert torch.isfinite(z.grad).all()
    assert math.isfinite(out.sum().item())
