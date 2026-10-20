import numpy as np
import pytest
import torch

from shrinkgpr.kernels import (
    kernel_matern_12,
    kernel_matern_32,
    kernel_matern_52,
    kernel_se,
    sqdist,
)

KERNELS = [kernel_se, kernel_matern_12, kernel_matern_32, kernel_matern_52]


def numpy_kernel(kernel, theta, tau, x, x_star):
    """Direct per-sample formula for one latent draw."""
    diff = x_star[:, None, :] - x[None, :, :]
    sq = np.sum(theta * diff**2, axis=-1)
    if kernel is kernel_se:
        return np.exp(-0.5 * sq) / tau
    D = np.sqrt(sq + 1e-4)
    if kernel is kernel_matern_12:
        return np.exp(-D) / tau
    if kernel is kernel_matern_32:
        return (1 + np.sqrt(3) * D) * np.exp(-np.sqrt(3) * D) / tau
    return (1 + np.sqrt(5) * D + 5 / 3 * D**2) * np.exp(-np.sqrt(5) * D) / tau


@pytest.fixture
def inputs():
    g = torch.Generator().manual_seed(1)
    f64 = {"dtype": torch.float64, "generator": g}
    thetas = torch.rand(3, 2, **f64) * 2
    tau = torch.rand(3, **f64) + 0.5
    x = torch.randn(5, 2, **f64)
    x_star = torch.randn(4, 2, **f64)
    return thetas, tau, x, x_star


def test_sqdist_matches_scaled_distances(inputs):
    thetas, _, x, x_star = inputs
    D = sqdist(x, thetas, x_star).numpy()
    for s in range(thetas.shape[0]):
        diff = x_star.numpy()[:, None, :] - x.numpy()[None, :, :]
        np.testing.assert_allclose(D[s], np.sum(thetas[s].numpy() * diff**2, -1), atol=1e-10)


@pytest.mark.parametrize("kernel", KERNELS)
def test_kernel_matches_numpy(kernel, inputs):
    thetas, tau, x, x_star = inputs
    for xs in (None, x_star):
        K = kernel(thetas, tau, x, xs).numpy()
        xs_np = x.numpy() if xs is None else xs.numpy()
        assert K.shape == (3, xs_np.shape[0], 5)
        for s in range(3):
            expected = numpy_kernel(kernel, thetas[s].numpy(), tau[s].item(), x.numpy(), xs_np)
            np.testing.assert_allclose(K[s], expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("kernel", KERNELS)
def test_kernel_symmetric_psd(kernel, inputs):
    thetas, tau, x, _ = inputs
    K = kernel(thetas, tau, x)
    assert torch.allclose(K, K.transpose(1, 2))
    assert (torch.linalg.eigvalsh(K) > -1e-10).all()


@pytest.mark.parametrize("kernel", KERNELS)
def test_kernel_float32_on_device(kernel):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    thetas = torch.rand(10, 3, device=device)
    tau = torch.rand(10, device=device) + 0.5
    x = torch.randn(15, 3, device=device)
    K = kernel(thetas, tau, x, torch.randn(7, 3, device=device))
    assert K.shape == (10, 7, 15) and K.dtype == torch.float32 and K.device == x.device
    assert torch.isfinite(K).all()
