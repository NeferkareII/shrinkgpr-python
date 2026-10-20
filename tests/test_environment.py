import torch

import shrinkgpr


def test_package_imports():
    assert shrinkgpr.__version__


def test_torch_runs_on_available_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(3, 3, device=device)
    assert torch.isfinite(x @ x.T).all()
