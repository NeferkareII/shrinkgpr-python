import numpy as np
import pytest
import torch

from shrinkgpr import kernel_se, sim_gpr, sim_mvgpr


def test_sim_gpr_columns_and_true_values():
    torch.manual_seed(0)
    sim = sim_gpr(N=40, d=4, d_mean=2, device="cpu")
    assert list(sim["data"].columns) == ["y", "x.1", "x.2", "x.3", "x.4", "x_mean.1", "x_mean.2"]
    assert sim["data"].shape == (40, 7)
    tv = sim["true_vals"]
    assert set(tv) == {"theta", "sigma2", "tau", "beta"}
    # perc_spars = 0.5: half of theta and beta are zero
    assert tv["theta"].shape == (4,) and (tv["theta"] == 0).sum() == 2
    assert tv["beta"].shape == (2,) and (tv["beta"] == 0).sum() == 1


def test_sim_gpr_single_columns_are_named_as_in_r():
    torch.manual_seed(0)
    sim = sim_gpr(N=10, d=1, d_mean=1, device="cpu")
    assert list(sim["data"].columns) == ["y", "x", "x_mean"]
    torch.manual_seed(0)
    assert list(sim_mvgpr(N=10, M=2, d=1, device="cpu")["data"].columns) == ["y.1", "y.2", "x"]


def test_sim_gpr_without_mean():
    torch.manual_seed(1)
    sim = sim_gpr(N=10, d=2, device="cpu")
    assert list(sim["data"].columns) == ["y", "x.1", "x.2"]
    assert "beta" not in sim["true_vals"]


def test_sim_gpr_is_reproducible_and_uses_given_values():
    torch.manual_seed(2)
    a = sim_gpr(N=15, d=2, d_mean=1, theta=[0.5, 0.0], beta=[2.0], device="cpu")
    torch.manual_seed(2)
    b = sim_gpr(N=15, d=2, d_mean=1, theta=[0.5, 0.0], beta=[2.0], device="cpu")
    np.testing.assert_array_equal(a["data"].to_numpy(), b["data"].to_numpy())
    np.testing.assert_array_equal(a["true_vals"]["theta"], [0.5, 0.0])
    np.testing.assert_array_equal(a["true_vals"]["beta"], [2.0])


def test_sim_gpr_follows_the_model():
    """Many draws at fixed covariates: y has mean x_mean beta and covariance K + sigma2 I."""
    torch.manual_seed(3)
    N = 4
    sims = [sim_gpr(N=N, d=2, d_mean=1, sigma2=0.2, tau=1.5, theta=[0.3, 0.1], beta=[1.0],
                    rho=0.5, device="cpu") for _ in range(4000)]
    # The covariates change with every draw, so compare residuals given the covariates
    resid, maha = [], []
    for s in sims:
        df = s["data"]
        x = torch.tensor(df[["x.1", "x.2"]].to_numpy(), dtype=torch.float64)
        K = kernel_se(torch.tensor([[0.3, 0.1]], dtype=torch.float64),
                      torch.tensor([1.5], dtype=torch.float64), x)[0] + 0.2 * torch.eye(N)
        r = df["y"].to_numpy() - df["x_mean"].to_numpy()
        resid.append(r)
        maha.append(r @ np.linalg.solve(K.numpy(), r))
    np.testing.assert_allclose(np.mean(resid), 0, atol=0.03)
    # Mahalanobis distances are chi-squared with N degrees of freedom
    assert np.mean(maha) == pytest.approx(N, rel=0.05)


@pytest.mark.parametrize("kwargs, match", [
    ({"N": 0}, "N"), ({"d": 1.5}, "d"), ({"d_mean": -1}, "d_mean"), ({"sigma2": 0}, "sigma2"),
    ({"perc_spars": 2}, "perc_spars"), ({"rho": -0.1}, "rho"), ({"theta": [1.0]}, "theta"),
    ({"d_mean": 2, "beta": [1.0]}, "beta"),
])
def test_sim_gpr_argument_checks(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sim_gpr(**{"device": "cpu", **kwargs})
