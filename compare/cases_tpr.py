"""Comparison cases for ShrinkTPR (R: TPR_class, shrinkTPR); R handlers in r/handlers_tpr.R.

R's ``TPR_class$predict`` draws its chi-squared variables with ``rgamma`` from
R's own generator, which PyTorch cannot reproduce. The cases record the draws
of the Python side and the R handlers substitute them for ``rgamma``.
"""

import numpy as np
import torch

from shrinkgpr._tpr_module import TPRModule, ldt, ldt_univariate
from shrinkgpr.kernels import kernel_se
from shrinkgpr.tpr import ShrinkTPR

from .numerics import FLOWS, KERNELS, Case, _grads, _perturb, _state


def _record_chi2(module):
    """Make ``module`` keep every chi-squared draw of predict() in the returned list."""
    draws = []
    original = module._draw_chi2

    def draw(df):
        out = original(df)
        draws.append(out)
        return out

    module._draw_chi2 = draw
    return draws


def build_cases(dtype):
    dt = torch.float64 if dtype == "float64" else torch.float32
    rng = np.random.default_rng(2027)

    def T(arr):
        return torch.as_tensor(np.asarray(arr, dtype=float), dtype=dt)

    cases = []

    # Gamma prior on nu - 2
    x_g = T(rng.uniform(0.01, 8, size=7))
    for alpha, beta in [(0.5, 2.0), (2.0, 0.5), (1.0, 1.0)]:
        cases.append(Case(f"tpr ldg alpha={alpha} beta={beta}", "tpr_ldg",
                          {"x": x_g, "alpha": alpha, "beta": beta},
                          {"ldg": TPRModule.ldg(x_g, T(alpha), T(beta))}))

    # Student-t likelihood
    B, N = 4, 7
    thetas = T(rng.uniform(0.01, 2, size=(B, 3)))
    tau = T(rng.uniform(0.3, 3, size=B))
    x = T(rng.normal(size=(N, 3)))
    K = kernel_se(thetas, tau, x)
    sigma2 = T(rng.uniform(0.05, 1, size=B))
    nu = T([2.3, 4.0, 11.0, 80.0])
    y_ld = T(rng.normal(size=(N, 1)))
    x_mean = T(rng.normal(size=(N, 2)))
    beta_ld = T(rng.normal(size=(B, 2)))
    for tag, xm, bt in [("", None, None), (" with mean", x_mean, beta_ld)]:
        cases.append(Case("tpr ldt" + tag, "tpr_ldt",
                          {"K": K, "sigma2": sigma2, "y": y_ld, "x_mean": xm, "beta": bt,
                           "nu": nu},
                          {"ll": ldt(K, None, sigma2, y_ld, nu, xm, bt)}))

    # Univariate predictive density; R takes pi as a float32 tensor
    y_t = T([[-1.5], [0.0], [0.4], [6.0]])
    mean_t, scale_t = T(rng.normal(size=B)), T(rng.uniform(0.1, 3, size=B))
    cases.append(Case("tpr dt", "tpr_dt",
                      {"y": y_t, "mean": mean_t, "scale": scale_t, "nu": nu},
                      {"dens": ldt_univariate(y_t, mean_t, scale_t, nu)}, tol="r_scalar"))

    # Full model: flows + softplus + ELBO + gradients, prediction, replayed training
    N, p = 12, 3
    x_tp = rng.normal(size=(N, p))
    y_tp = np.sin(2 * x_tp[:, 0]) + 0.1 * rng.standard_t(4, size=N)
    x_mean_tp = np.column_stack([np.ones(N), x_tp[:, 1]])
    x_new_tp = rng.normal(size=(5, p))
    x_mean_new_tp = np.column_stack([np.ones(5), x_new_tp[:, 1]])
    y_eval = np.array([-0.5, 0.0, 1.2])
    y_new_tp = np.sin(2 * x_new_tp[:, 0])
    model_specs = [
        ("sylvester", "kernel_se", False, {}, (0.5, 2.0)),
        ("iaf", "kernel_matern_32", True, {"hidden_dim": 16}, (1.5, 0.7)),
        ("nsf_cl", "kernel_matern_52", True, {"hidden_dim": 16, "K": 5}, (0.5, 2.0)),
        ("sylvester", "kernel_matern_12", False, {"n_householder": 2}, (2.0, 1.0)),
    ]
    for i, (fname, kname, with_mean, fargs, (nu_a, nu_b)) in enumerate(model_specs):
        inputs = {"y": y_tp, "x": x_tp, "x_mean": x_mean_tp if with_mean else None,
                  "a": 0.5, "c": 0.5, "a_mean": 0.5, "c_mean": 0.5, "sigma2_rate": 10.0,
                  "nu_alpha": nu_a, "nu_beta": nu_b, "n_layers": 3, "flow": fname,
                  "flow_args": fargs, "kernel": kname}
        label = f"{fname}, {kname}, {'mean' if with_mean else 'no mean'}"

        torch.manual_seed(300 + i)
        module = TPRModule(T(y_tp), T(x_tp), T(x_mean_tp) if with_mean else None,
                           nu_alpha=nu_a, nu_beta=nu_b, n_layers=3, flow_func=FLOWS[fname],
                           flow_args=fargs, kernel_func=KERNELS[kname]).to(dt)
        _perturb(module, 0.05, seed=300 + i)
        z_tp = module.gen_batch(10)
        zk, log_det_J = module(z_tp)
        elbo = module.elbo(zk, log_det_J)
        (-elbo).backward()
        cases.append(Case(f"tpr {label}", "tpr",
                          {**inputs, "state": _state(module), "z": z_tp},
                          {"zk": zk, "log_det_J": log_det_J, "elbo": elbo, **_grads(module)},
                          tol="r_scalar"))

        # Prediction from the same base samples; predict() draws eps from the
        # shared libtorch generator and its chi-squared variables are handed to R
        z_pred = module.gen_batch(6).detach()
        module.gen_batch = lambda n, z_pred=z_pred: z_pred
        chi2 = _record_chi2(module)
        xm_new = T(x_mean_new_tp) if with_mean else None
        means, scales, pred_nu = module.calc_pred_moments(T(x_new_tp), 6, xm_new)
        torch.manual_seed(31)
        draws = module.predict(T(x_new_tp), 6, xm_new)
        dens = module.eval_pred_dens(T(y_eval), T(x_new_tp[:1]), 6,
                                     None if xm_new is None else xm_new[:1], log=True)
        cases.append(Case(f"tpr pred {label}", "tpr_pred",
                          {**inputs, "state": _state(module), "z": z_pred, "x_new": x_new_tp,
                           "x_mean_new": x_mean_new_tp if with_mean else None,
                           "y_eval": y_eval, "seed": 31, "nsamp": 6, "chi2": chi2[0]},
                          {"means": means, "vars": scales, "nu": pred_nu, "predict": draws,
                           "chi2_df": pred_nu, "eval_pred_dens": dens},
                          tol="r_scalar"))

        # Posterior samples from the same model and base samples
        wrapper = ShrinkTPR(y_tp, x_tp, x_mean_tp if with_mean else None, n_layers=1,
                            device="cpu")
        wrapper.model_ = module
        post = wrapper.gen_posterior_samples(6)
        cases.append(Case(f"tpr posterior {label}", "tpr_posterior",
                          {**inputs, "state": _state(module), "z": z_pred, "nsamp": 6},
                          {k: np.asarray(v) for k, v in post.items()}, tol="r_scalar"))

        if i < 2:
            n_epochs = 30
            torch.manual_seed(400 + i)
            model = ShrinkTPR(y_tp, x_tp, x_mean_tp if with_mean else None, nu_alpha=nu_a,
                              nu_beta=nu_b, n_layers=3, flow_func=FLOWS[fname],
                              flow_args=fargs, kernel_func=KERNELS[kname], device="cpu")
            model._module = model._module.to(dt)
            state0 = {k: v.clone() for k, v in _state(model._module).items()}
            zs = [model._module.gen_batch(10) for _ in range(n_epochs)]
            replay = iter(zs)
            model._module.gen_batch = lambda n_latent, replay=replay: next(replay)
            model.fit(n_epochs=n_epochs, auto_stop=False, display_progress=False)
            final = {"state." + k: v for k, v in model.last_model_.named_parameters()}
            cases.append(Case(f"tpr train {label}", "tpr_train",
                              {**inputs, "state": state0, "zs": zs},
                              {"loss_stor": model.loss_stor_, "n_replayed": np.array(n_epochs),
                               **final},
                              tol="train"))

    # End to end through the public APIs with the same seed (float32 only, as
    # for ShrinkGPR); see numerics.py. With a mean, float32 rounding differences
    # grow quickly after ~45 epochs (to ~1e-2 by epoch 50, varying between
    # runs), so the runs are kept shorter than for ShrinkGPR.
    if dtype == "float32":
        n_epochs = 30
        for with_mean in (False, True):
            torch.manual_seed(7)
            model = ShrinkTPR(y_tp, x_tp, x_mean_tp if with_mean else None, device="cpu")
            model.fit(n_epochs=n_epochs, auto_stop=False, display_progress=False)
            chi2 = _record_chi2(model.model_)
            xm_new = x_mean_new_tp if with_mean else None
            xm_row = x_mean_new_tp[:1] if with_mean else None
            torch.manual_seed(11)
            moments = model.calc_pred_moments(x_new_tp, nsamp=20, x_mean_new=xm_new)
            torch.manual_seed(12)
            draws = model.predict(x_new_tp, nsamp=20, x_mean_new=xm_new)
            torch.manual_seed(13)
            lpds = model.lpds(y_new_tp[0], x_new_tp[:1], nsamp=20, x_mean_new=xm_row)
            cases.append(Case(f"tpr end to end shrinkTPR() {'mean' if with_mean else 'no mean'}",
                              "tpr_end_to_end",
                              {"y": y_tp, "x": x_tp, "seed": 7, "n_epochs": n_epochs,
                               "with_mean": with_mean, "x_new": x_new_tp, "y_new": y_new_tp,
                               "chi2": chi2[0]},
                              {"loss_stor": model.loss_stor_, "loss": np.array(model.loss_),
                               "means": moments["means"], "vars": moments["vars"],
                               "nu": moments["nu"], "predict": draws, "lpds": np.array(lpds)},
                              tol="train"))

    return cases
