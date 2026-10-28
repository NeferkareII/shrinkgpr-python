"""Comparison cases of the multivariate GPR (R handlers in r/handlers_mvgpr.R).

``build_model_cases`` is shared with cases_mvtpr.py: the R handlers take the
model class (``MVGPR`` or ``MVTPR``) as an input.
"""

import numpy as np
import pandas as pd
import torch

from shrinkgpr._mvgpr_module import MVGPRModule, ldnorm_multi, make_corr_chol
from shrinkgpr.mvgpr import ShrinkMVGPR

from .numerics import FLOWS, KERNELS, Case, _grads, _perturb, _state


def _data(rng, N, M, p):
    x = rng.normal(size=(N, p))
    y = np.column_stack([np.sin((j + 1) * x[:, 0]) + 0.3 * j * x[:, 1] for j in range(M)])
    return y + 0.1 * rng.normal(size=(N, M)), x


def elbo_parts(module, zk):
    """Likelihood, prior and the Jacobian terms of the ELBO."""
    params = module.unpack(zk)
    K = module.kernel_func(params["thetas"], params["tau"], module.x)
    omega = module.omega_factor(params)
    return {"likelihood": module.log_likelihood(K, omega["L_Om"], params).mean(),
            "prior": module.log_prior(params, omega),
            "logJ_D": omega["logJ_D"].mean(), "log_det_S": omega["log_det_S"].mean()}


def build_model_cases(dtype, model, module_cls, user_cls, extra=None, seed=0,
                      with_predict=True):
    """ELBO, parts, prediction, replayed training and end-to-end cases of one MV model.

    ``model`` is the R class name (``"MVGPR"`` or ``"MVTPR"``); ``extra`` are
    additional prior hyperparameters for both constructors.
    """
    dt = torch.float64 if dtype == "float64" else torch.float32
    rng = np.random.default_rng(seed)
    extra = extra or {}
    tag = model.lower()

    def T(arr):
        return torch.as_tensor(np.asarray(arr, dtype=float), dtype=dt)

    cases = []
    N, N_new = 12, 4
    specs = [
        ("sylvester", "kernel_se", 2, 2, {}),
        ("iaf", "kernel_matern_32", 3, 2, {"hidden_dim": 16}),
        ("nsf_cl", "kernel_matern_52", 2, 3, {"hidden_dim": 16, "K": 5}),
        ("sylvester", "kernel_matern_12", 4, 2, {"n_householder": 2}),
    ]
    for i, (fname, kname, M, p, fargs) in enumerate(specs):
        y, x = _data(rng, N, M, p)
        x_new = rng.normal(size=(N_new, p))
        y_eval = rng.normal(size=(3, M))
        inputs = {"model": model, "y": y, "x": x, "a": 0.5, "c": 0.5, "eta": 4.0, "a_Om": 0.5,
                  "c_Om": 0.5, "sigma2_rate": 10.0, **extra, "n_layers": 3, "flow": fname,
                  "flow_args": fargs, "kernel": kname}
        label = f"{fname}, {kname}, M={M}"

        torch.manual_seed(300 + i)
        module = module_cls(T(y), T(x), n_layers=3, flow_func=FLOWS[fname], flow_args=fargs,
                            kernel_func=KERNELS[kname], **extra).to(dt)
        _perturb(module, 0.05, seed=300 + i)
        z = module.gen_batch(10)
        zk, log_det_J = module(z)
        elbo = module.elbo(zk, log_det_J)
        (-elbo).backward()
        cases.append(Case(f"{tag} {label}", "mv_model",
                          {**inputs, "state": _state(module), "z": z},
                          {"zk": zk, "log_det_J": log_det_J, "elbo": elbo, **_grads(module)},
                          tol="r_scalar"))
        with torch.no_grad():
            parts = elbo_parts(module, zk)
        cases.append(Case(f"{tag} parts {label}", "mv_parts",
                          {**inputs, "state": _state(module), "zk": zk.detach()}, parts,
                          tol="r_scalar"))

        # Prediction with replayed base samples; predict() draws its noise from
        # the shared libtorch generator after the same seed
        z_pred = module.gen_batch(6).detach()
        module.gen_batch = lambda n, z_pred=z_pred: z_pred
        x_new_t = T(x_new)
        moments = module.calc_pred_moments(x_new_t, 6)
        dens = module.eval_pred_dens(T(y_eval), x_new_t[:1], 6, log=True)
        expected = {k: v for k, v in moments.items()}
        expected["eval_pred_dens"] = dens
        if with_predict:
            torch.manual_seed(31)
            expected["predict"] = module.predict(x_new_t, 6)
        cases.append(Case(f"{tag} pred {label}", "mv_pred",
                          {**inputs, "state": _state(module), "z": z_pred, "x_new": x_new,
                           "y_eval": y_eval, "seed": 31, "nsamp": 6,
                           "with_predict": with_predict},
                          expected, tol="r_scalar"))

        # Posterior samples from the same model and base samples
        wrapper = user_cls(y, x, n_layers=1, device="cpu", **extra)
        wrapper.model_ = module
        post = wrapper.gen_posterior_samples(6)
        cases.append(Case(f"{tag} posterior {label}", "mv_posterior",
                          {**inputs, "state": _state(module), "z": z_pred, "nsamp": 6},
                          {k: np.asarray(v) for k, v in post.items()}, tol="r_scalar"))

        if i < 2:
            n_epochs = 30
            torch.manual_seed(400 + i)
            fit = user_cls(y, x, n_layers=3, flow_func=FLOWS[fname], flow_args=fargs,
                           kernel_func=KERNELS[kname], device="cpu", **extra)
            fit._module = fit._module.to(dt)
            state0 = {k: v.clone() for k, v in _state(fit._module).items()}
            zs = [fit._module.gen_batch(10) for _ in range(n_epochs)]
            replay = iter(zs)
            fit._module.gen_batch = lambda n_latent, replay=replay: next(replay)
            fit.fit(n_epochs=n_epochs, auto_stop=False, display_progress=False)
            final = {"state." + k: v for k, v in fit.last_model_.named_parameters()}
            cases.append(Case(f"{tag} train {label}", "mv_train",
                              {**inputs, "state": state0, "zs": zs},
                              {"loss_stor": fit.loss_stor_, "n_replayed": np.array(n_epochs),
                               **final},
                              tol="train"))

    # End to end through the public APIs in float32: R torch and PyTorch share
    # libtorch's generators, and the Sylvester flow and gen_batch draw in the
    # same order, so the same seed gives the same run
    if dtype == "float32":
        y, x = _data(rng, N, 2, 2)
        x_new = rng.normal(size=(N_new, 2))
        y_new = rng.normal(size=(N_new, 2))
        n_epochs = 50
        torch.manual_seed(7)
        fit = user_cls(y, x, device="cpu", **extra)
        fit.fit(n_epochs=n_epochs, auto_stop=False, display_progress=False)
        torch.manual_seed(11)
        moments = fit.calc_pred_moments(x_new, nsamp=20)
        expected = {"loss_stor": fit.loss_stor_, "loss": np.array(fit.loss_), **moments}
        if with_predict:
            torch.manual_seed(12)
            expected["predict"] = fit.predict(x_new, nsamp=20)
        torch.manual_seed(13)
        expected["lpds"] = np.array(fit.lpds(y_new[0], x_new[:1], nsamp=20))
        # Marginal samples with fixed covariates (R names the covariates X1, X2)
        torch.manual_seed(15)
        marg = fit.gen_marginal_samples("x1", nsamp=4, fixed_x=pd.DataFrame(
            x_new[:1], columns=["x1", "x2"]), n_eval_points=6, display_progress=False)
        expected["marg"], expected["marg.grid"] = marg.mean_pred, marg.grid
        cases.append(Case(f"{tag} end to end shrink{model}()", "mv_end_to_end",
                          {"model": model, "y": y, "x": x, **extra, "seed": 7,
                           "n_epochs": n_epochs, "x_new": x_new, "y_new": y_new,
                           "with_predict": with_predict},
                          expected, tol="train"))
    return cases


def build_cases(dtype):
    dt = torch.float64 if dtype == "float64" else torch.float32
    rng = np.random.default_rng(2027)

    def T(arr):
        return torch.as_tensor(np.asarray(arr, dtype=float), dtype=dt)

    cases = []

    # Correlation Cholesky factor and its log-Jacobian, also with large inputs
    # that hit the clamps of the stick breaking
    for M in (2, 3, 4):
        P = M * (M - 1) // 2
        u = T(np.vstack([rng.normal(size=(4, P)), 6 * rng.normal(size=(2, P))]))
        L, logJ = make_corr_chol(u, M)
        cases.append(Case(f"mvgpr make_corr_chol M={M}", "mv_corr_chol",
                          {"model": "MVGPR", "M": M, "u": u}, {"L": L, "logJ": logJ},
                          tol="r_scalar"))

    # Matrix-normal likelihood from Cholesky factors
    for N, M in [(7, 2), (5, 4)]:
        a = rng.normal(size=(3, N, N))
        b = rng.normal(size=(3, M, M))
        L_K = torch.linalg.cholesky(T(a @ a.transpose(0, 2, 1) + N * np.eye(N)))
        L_Om = torch.linalg.cholesky(T(b @ b.transpose(0, 2, 1) + np.eye(M)))
        y = T(rng.normal(size=(N, M)))
        cases.append(Case(f"mvgpr ldnorm_multi N={N} M={M}", "mv_ldnorm",
                          {"L_K": L_K, "L_Om": L_Om, "y": y},
                          {"ll": ldnorm_multi(L_K, L_Om, y)}))

    return cases + build_model_cases(dtype, "MVGPR", MVGPRModule, ShrinkMVGPR, seed=1)
