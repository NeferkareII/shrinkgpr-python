import warnings

import numpy as np
import pandas as pd
import pytest
import torch

import shrinkgpr
from shrinkgpr import ShrinkGPR, ShrinkMVGPR, ShrinkMVTPR, ShrinkTPR, iaf

MODELS = [ShrinkGPR, ShrinkTPR, ShrinkMVGPR, ShrinkMVTPR]


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(11)
    n = 15
    out = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                        "f": list("abc") * (n // 3)})
    out["y"] = np.sin(2 * out["x1"]) + 0.1 * rng.normal(size=n)
    out["y2"] = np.cos(out["x2"]) + 0.1 * rng.normal(size=n)
    return out


def build(cls, df, **kwargs):
    kwargs.setdefault("device", "cpu")
    if cls in (ShrinkMVGPR, ShrinkMVTPR):
        return cls.from_formula("cbind(y, y2) ~ x1 + f", df, **kwargs)
    return cls.from_formula("y ~ x1 + f", df, formula_mean="~ x2", **kwargs)


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
def test_round_trip_predicts_the_same(df, cls, tmp_path):
    torch.manual_seed(0)
    model = build(cls, df, flow_func=iaf).fit(n_epochs=5, display_progress=False)
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = shrinkgpr.load(path)
    assert type(loaded) is cls and loaded.device == model.device
    np.testing.assert_array_equal(loaded.loss_stor_, model.loss_stor_)
    assert loaded.loss_ == model.loss_ and loaded.stop_reason_ == model.stop_reason_
    assert loaded.formula_ == model.formula_ and loaded.x_names_ == model.x_names_

    new = df.iloc[:4]
    for method in ("predict", "calc_pred_moments", "gen_posterior_samples"):
        args = () if method == "gen_posterior_samples" else (new,)
        torch.manual_seed(1)
        a = getattr(model, method)(*args, nsamp=3)
        torch.manual_seed(1)
        b = getattr(loaded, method)(*args, nsamp=3)
        if isinstance(a, dict):
            for k in a:
                np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))
        else:
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("cls", [ShrinkGPR, ShrinkMVTPR], ids=lambda c: c.__name__)
def test_training_continues_after_loading(df, cls, tmp_path):
    """fit, save, load, fit gives the same run as fitting twice in memory."""
    torch.manual_seed(0)
    reference = build(cls, df).fit(n_epochs=5, display_progress=False)
    path = tmp_path / "model.pt"
    reference.save(path)
    torch.manual_seed(2)
    reference.fit(n_epochs=4, display_progress=False)

    loaded = shrinkgpr.load(path)
    torch.manual_seed(2)
    loaded.fit(n_epochs=4, display_progress=False)
    np.testing.assert_array_equal(loaded.loss_stor_, reference.loss_stor_)
    assert loaded.loss_ == reference.loss_
    # The optimizer still updates the loaded model's parameters
    assert all(p is q for p, q in zip(loaded.optimizer_.param_groups[0]["params"],
                                      loaded.last_model_.parameters(), strict=True))


def test_unfitted_model_and_file_objects(df, tmp_path):
    import io

    model = ShrinkGPR(df["y"], df[["x1"]], device="cpu")
    buf = io.BytesIO()
    model.save(buf)
    buf.seek(0)
    loaded = shrinkgpr.load(buf)
    assert not loaded.is_fitted
    torch.manual_seed(0)
    loaded.fit(n_epochs=2, display_progress=False)
    assert loaded.loss_stor_.shape == (2,)


def test_load_rejects_other_files(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"weights": torch.zeros(2)}, path)
    with pytest.raises(TypeError, match="save"):
        shrinkgpr.load(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_handling(df, tmp_path):
    torch.manual_seed(0)
    model = build(ShrinkTPR, df, device="cuda").fit(n_epochs=3, display_progress=False)
    path = tmp_path / "model.pt"
    model.save(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        on_gpu = shrinkgpr.load(path)
    assert on_gpu.device.type == "cuda" and on_gpu.model_.x.is_cuda

    on_cpu = shrinkgpr.load(path, device="cpu")
    assert on_cpu.model_.x.device.type == "cpu" and on_cpu.last_model_.x.device.type == "cpu"
    assert all(v.device.type == "cpu" for v in on_cpu._best_state.values())
    on_cpu.fit(n_epochs=2, display_progress=False)
    assert on_cpu.loss_stor_.shape == (5,)


def test_cuda_model_loads_on_cpu_with_a_warning(df, tmp_path, monkeypatch):
    model = build(ShrinkGPR, df)
    model.device = torch.device("cuda")  # as if trained on CUDA
    path = tmp_path / "model.pt"
    model.save(path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="CUDA is not available"):
        loaded = shrinkgpr.load(path)
    assert loaded.device.type == "cpu"
