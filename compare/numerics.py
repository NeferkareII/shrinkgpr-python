"""Function-by-function numerical comparison of the Python port with the R package.

Every case evaluates one function in Python, sends the same inputs (and, for
flows and models, the same parameter values) to R, and compares the outputs
elementwise. ``train`` cases replay identical base samples through both
training loops, so the loss trajectories must agree as well.

Usage: uv run python -m compare.numerics [--dtype float64|float32|both]
"""

import argparse
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from shrinkgpr import ShrinkGPR
from shrinkgpr._gpr_module import GPRModule, ldnorm
from shrinkgpr._hyperu import log_hyperu, robust_hyperu
from shrinkgpr._utils import lightweight_ols, res_protector, robust_chol
from shrinkgpr.flows import iaf, nsf_cl, sylvester
from shrinkgpr.kernels import (
    kernel_matern_12,
    kernel_matern_32,
    kernel_matern_52,
    kernel_se,
    sqdist,
)

from ._rbridge import REFERENCE_SHA, r_package_sha, run_r

KERNELS = {f.__name__: f for f in (kernel_se, kernel_matern_12, kernel_matern_32, kernel_matern_52)}
FLOWS = {"sylvester": sylvester, "iaf": iaf, "nsf_cl": nsf_cl}

# Elementwise check |py - r| <= atol * max|r| + rtol * |r|, per precision and case group.
# "r_scalar": R torch turns an R number on the left of an arithmetic operator into
# a float32 tensor (0.7 * x uses 0.69999998807907104 even for float64 x), e.g.
# `self$beta_sp * ...` in GPR_class or `1.1 * shift + tol` in robust_chol, so in
# float64 R is only accurate to ~1e-8 relative there.
TOLERANCES = {
    "float64": {"default": (1e-12, 1e-8), "r_scalar": (1e-6, 1e-6), "train": (1e-6, 1e-6)},
    "float32": {"default": (1e-5, 1e-4), "r_scalar": (1e-5, 1e-4), "train": (1e-3, 1e-3)},
}


@dataclass
class Case:
    name: str
    fn: str
    inputs: dict
    expected: dict
    tol: str = "default"
    notes: list = field(default_factory=list)


def _state(module):
    return {k: v for k, v in module.state_dict().items() if v is not None}


def _perturb(module, scale, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))


def _grads(module, prefix="grad."):
    return {prefix + k: p.grad for k, p in module.named_parameters()}


def plugin_names():
    return [p.stem.removeprefix("cases_")
            for p in sorted(Path(__file__).parent.glob("cases_*.py"))]


def build_cases(dtype, plugins=None):
    """All cases, or with ``plugins`` (e.g. ``["tpr"]``) only those of compare/cases_<name>.py."""
    if plugins is not None:
        cases = []
        for name in plugins:
            cases.extend(importlib.import_module(f"{__package__}.cases_{name}").build_cases(dtype))
        return cases

    dt = torch.float64 if dtype == "float64" else torch.float32
    rng = np.random.default_rng(2026)

    def T(arr):
        return torch.as_tensor(np.asarray(arr, dtype=float), dtype=dt)

    cases = []

    # log_hyperu and its small-z branches (z < 1e-9), for the (a, b) the priors use
    z = T([1e-12, 5e-10, 1e-6, 1e-3, 0.1, 1.0, 10.0, 100.0])
    for a, b in [(1.0, 1.0), (1.0, 1.2), (2.0, 1.1), (1.0, 2.0), (1.0, 2.5), (1.0, 0.7),
                 (1.0, 0.0), (1.0, -0.5)]:
        cases.append(Case(f"hyperu a={a} b={b}", "hyperu", {"a": a, "b": b, "z": z},
                          {"u": robust_hyperu(a, b, z)}))
    for a, b in [(1.0, 1.0), (1.0, 1.5), (0.6, 1.3)]:
        zg = T([1e-12, 1e-3, 0.5, 4.0]).requires_grad_(True)
        value = log_hyperu(T(a), T(b), zg)
        value.sum().backward()
        cases.append(Case(f"log_hyperu+grad a={a} b={b}", "log_hyperu",
                          {"a": a, "b": b, "z": zg.detach()},
                          {"value": value, "grad": zg.grad}))

    # Utilities
    y_ols = 0.01 * np.arange(1, 101) + rng.normal(size=100)
    cases.append(Case("lightweight_ols", "lightweight_ols",
                      {"y": y_ols, "x": np.arange(1.0, 101.0)},
                      {"p": np.array(lightweight_ols(y_ols, np.arange(1, 101)))}))
    a_mat = T(rng.normal(size=(3, 6, 6)))
    spd = a_mat @ a_mat.transpose(1, 2) + 6 * torch.eye(6, dtype=dt)
    v = T(rng.normal(size=(3, 5, 2)))
    singular = v @ v.transpose(1, 2)
    for name, A in [("spd", spd), ("singular", singular),
                    ("indefinite", singular - 1e-3 * torch.eye(5, dtype=dt))]:
        cases.append(Case(f"robust_chol {name}", "robust_chol", {"A": A}, {"L": robust_chol(A)},
                          tol="r_scalar"))
    xs = T([-2.0, -1e-9, 0.0, 3e-7, 1e-5, 4.0])
    for tol in (1e-6, 1e-4):
        cases.append(Case(f"res_protector tol={tol}", "res_protector", {"x": xs, "tol": tol},
                          {"out": res_protector(xs, tol=tol)}))

    # Kernels
    thetas = T(rng.uniform(0.01, 2, size=(4, 3)))
    tau = T(rng.uniform(0.3, 3, size=4))
    x = T(rng.normal(size=(7, 3)))
    x_star = T(rng.normal(size=(5, 3)))
    for xst in (None, x_star):
        tag = "" if xst is None else " x_star"
        cases.append(Case("sqdist" + tag, "sqdist", {"x": x, "thetas": thetas, "x_star": xst},
                          {"D": sqdist(x, thetas, xst)}))
        for kname, kfun in KERNELS.items():
            cases.append(Case(kname + tag, "kernel",
                              {"kernel": kname, "thetas": thetas, "tau": tau, "x": x,
                               "x_star": xst},
                              {"K": kfun(thetas, tau, x, xst)}))

    # Flows: same parameters in both languages, moved away from their init
    d = 5
    z_flow = T(1.5 * rng.normal(size=(8, d)))
    flow_specs = [
        ("sylvester", {}, 0.3),
        ("sylvester", {"n_householder": 2}, 0.3),
        ("iaf", {"hidden_dim": 16, "n_hidden": 2}, 0.3),
        ("nsf_cl", {"K": 6, "hidden_dim": 16}, 0.1),
        ("nsf_cl", {"K": 4, "B": 2.0, "hidden_dim": 16}, 0.1),
    ]
    for i, (fname, args, scale) in enumerate(flow_specs):
        torch.manual_seed(i)
        flow = FLOWS[fname](d, **args).to(dt)
        _perturb(flow, scale, seed=i)
        r_args = dict(args)
        if fname == "iaf":
            r_args["perm"] = (flow.perm + 1).tolist()  # R expects a 1-based permutation
        zk, log_diag_j = flow(z_flow)
        (zk.sum() + log_diag_j.sum()).backward()
        cases.append(Case(f"flow {fname} {args}", "flow",
                          {"flow": fname, "d": d, "args": r_args, "state": _state(flow),
                           "z": z_flow},
                          {"zk": zk, "log_diag_j": log_diag_j, **_grads(flow)}))

    # Priors
    theta = T(rng.uniform(1e-3, 3, size=(6, 3)))
    beta = T(rng.normal(size=(6, 3)))
    tau6 = T(rng.uniform(0.1, 4, size=6))
    for a_p, c_p in [(0.5, 0.5), (0.2, 0.3), (1.0, 2.0)]:
        at, ct = T(a_p), T(c_p)
        cases.append(Case(f"priors a={a_p} c={c_p}", "priors",
                          {"a": a_p, "c": c_p, "theta": theta, "beta": beta, "tau": tau6,
                           "rate": 10.0},
                          {"ltg": GPRModule.ltg(theta, at, ct, tau6),
                           "ngg": GPRModule.ngg(beta, at, ct, tau6),
                           "ldf": GPRModule.ldf(tau6, 2 * ct, 2 * at),
                           "lexp": GPRModule.lexp(tau6, T(10.0))}))

    # Likelihood
    K = kernel_se(thetas, tau, x)
    sigma2 = T(rng.uniform(0.05, 1, size=4))
    y_ld = T(rng.normal(size=(7, 1)))
    x_mean = T(rng.normal(size=(7, 2)))
    beta_ld = T(rng.normal(size=(4, 2)))
    for tag, xm, bt in [("", None, None), (" with mean", x_mean, beta_ld)]:
        cases.append(Case("ldnorm" + tag, "ldnorm",
                          {"K": K, "sigma2": sigma2, "y": y_ld, "x_mean": xm, "beta": bt},
                          {"ll": ldnorm(K, None, sigma2, y_ld, xm, bt)}))

    # Full model: flows + softplus + ELBO + gradients, and replayed training
    N, p = 12, 3
    x_gp = rng.normal(size=(N, p))
    y_gp = np.sin(2 * x_gp[:, 0]) + 0.1 * rng.normal(size=N)
    x_mean_gp = np.column_stack([np.ones(N), x_gp[:, 1]])
    x_new_gp = rng.normal(size=(5, p))
    x_mean_new_gp = np.column_stack([np.ones(5), x_new_gp[:, 1]])
    y_eval = np.array([-0.5, 0.0, 1.2])
    y_new_gp = np.sin(2 * x_new_gp[:, 0])
    model_specs = [
        ("sylvester", "kernel_se", False, {}),
        ("iaf", "kernel_matern_32", True, {"hidden_dim": 16}),
        ("nsf_cl", "kernel_matern_52", True, {"hidden_dim": 16, "K": 5}),
        ("sylvester", "kernel_matern_12", False, {"n_householder": 2}),
    ]
    for i, (fname, kname, with_mean, fargs) in enumerate(model_specs):
        inputs = {"y": y_gp, "x": x_gp, "x_mean": x_mean_gp if with_mean else None,
                  "a": 0.5, "c": 0.5, "a_mean": 0.5, "c_mean": 0.5, "sigma2_rate": 10.0,
                  "n_layers": 3, "flow": fname, "flow_args": fargs, "kernel": kname}
        label = f"{fname}, {kname}, {'mean' if with_mean else 'no mean'}"

        torch.manual_seed(100 + i)
        module = GPRModule(T(y_gp), T(x_gp), T(x_mean_gp) if with_mean else None,
                           n_layers=3, flow_func=FLOWS[fname], flow_args=fargs,
                           kernel_func=KERNELS[kname]).to(dt)
        _perturb(module, 0.05, seed=100 + i)
        z_gp = module.gen_batch(10)
        zk, log_det_J = module(z_gp)
        elbo = module.elbo(zk, log_det_J)
        (-elbo).backward()
        cases.append(Case(f"gpr {label}", "gpr",
                          {**inputs, "state": _state(module), "z": z_gp},
                          {"zk": zk, "log_det_J": log_det_J, "elbo": elbo, **_grads(module)},
                          tol="r_scalar"))

        # Prediction: same base samples; predict() then draws its noise from
        # the shared libtorch generator after the same seed
        z_pred = module.gen_batch(6).detach()
        module.gen_batch = lambda n, z_pred=z_pred: z_pred
        xm_new = T(x_mean_new_gp) if with_mean else None
        means, covs = module.calc_pred_moments(T(x_new_gp), 6, xm_new)
        torch.manual_seed(31)
        draws = module.predict(T(x_new_gp), 6, xm_new)
        dens = module.eval_pred_dens(T(y_eval), T(x_new_gp[:1]), 6,
                                     None if xm_new is None else xm_new[:1], log=True)
        cases.append(Case(f"pred {label}", "pred",
                          {**inputs, "state": _state(module), "z": z_pred, "x_new": x_new_gp,
                           "x_mean_new": x_mean_new_gp if with_mean else None,
                           "y_eval": y_eval, "seed": 31, "nsamp": 6},
                          {"means": means, "vars": covs, "predict": draws,
                           "eval_pred_dens": dens},
                          tol="r_scalar"))

        if i < 2:
            n_epochs = 30
            torch.manual_seed(200 + i)
            model = ShrinkGPR(y_gp, x_gp, x_mean_gp if with_mean else None, n_layers=3,
                              flow_func=FLOWS[fname], flow_args=fargs,
                              kernel_func=KERNELS[kname], device="cpu")
            model._module = model._module.to(dt)
            state0 = {k: v.clone() for k, v in _state(model._module).items()}
            zs = [model._module.gen_batch(10) for _ in range(n_epochs)]
            replay = iter(zs)
            model._module.gen_batch = lambda n_latent, replay=replay: next(replay)
            model.fit(n_epochs=n_epochs, auto_stop=False, display_progress=False)
            final = {"state." + k: v for k, v in model.last_model_.named_parameters()}
            cases.append(Case(f"train {label}", "train",
                              {**inputs, "state": state0, "zs": zs},
                              {"loss_stor": model.loss_stor_, "n_replayed": np.array(n_epochs),
                               **final},
                              tol="train"))

    # End to end through the public APIs: R torch and PyTorch share libtorch's
    # generators, and the Sylvester flow draws its initial values and base
    # samples in the same order as R, so the same seed gives the same run.
    # (iaf and nsf_cl draw permutations with R's sample.int instead.) R fits
    # in float32 only, through its formula interface.
    if dtype == "float32":
        n_epochs = 50
        for with_mean in (False, True):
            torch.manual_seed(7)
            model = ShrinkGPR(y_gp, x_gp, x_mean_gp if with_mean else None, device="cpu")
            model.fit(n_epochs=n_epochs, auto_stop=False, display_progress=False)
            xm_new = x_mean_new_gp if with_mean else None
            xm_row = x_mean_new_gp[:1] if with_mean else None
            torch.manual_seed(11)
            moments = model.calc_pred_moments(x_new_gp, nsamp=20, x_mean_new=xm_new)
            torch.manual_seed(12)
            draws = model.predict(x_new_gp, nsamp=20, x_mean_new=xm_new)
            torch.manual_seed(13)
            lpds = model.lpds(y_new_gp[0], x_new_gp[:1], nsamp=20, x_mean_new=xm_row)
            cases.append(Case(f"end to end shrinkGPR() {'mean' if with_mean else 'no mean'}",
                              "end_to_end",
                              {"y": y_gp, "x": x_gp, "seed": 7, "n_epochs": n_epochs,
                               "with_mean": with_mean, "x_new": x_new_gp, "y_new": y_new_gp},
                              {"loss_stor": model.loss_stor_, "loss": np.array(model.loss_),
                               "means": moments["means"], "vars": moments["vars"],
                               "predict": draws, "lpds": np.array(lpds)},
                              tol="train"))

    # Cases of further models live in compare/cases_<model>.py, each with a
    # build_cases(dtype) returning a list of Case; their R handlers live in
    # compare/r/handlers_<model>.R
    return cases + build_cases(dtype, plugin_names())


def _as_numpy(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().double().numpy()
    return np.asarray(v, dtype=float)


@dataclass
class Row:
    case: str
    output: str
    max_abs: float
    max_rel: float
    ok: bool
    message: str = ""


def compare(cases, r_results, dtype):
    rows = []
    for case in cases:
        r_out = r_results.get(case.name, {})
        if "error" in r_out:
            rows.append(Row(case.name, "-", np.nan, np.nan, False, f"R error: {r_out['error']}"))
            continue
        atol, rtol = TOLERANCES[dtype][case.tol]
        for key, py_val in case.expected.items():
            py = _as_numpy(py_val)
            if key not in r_out:
                rows.append(Row(case.name, key, np.nan, np.nan, False, "missing in R output"))
                continue
            r = np.asarray(r_out[key], dtype=float).reshape(py.shape) if np.size(
                r_out[key]) == py.size else None
            if r is None:
                rows.append(Row(case.name, key, np.nan, np.nan, False,
                                f"shape mismatch: py {py.shape}, R {np.shape(r_out[key])}"))
                continue
            diff = np.abs(py - r)
            nan_mismatch = np.isnan(py) != np.isnan(r)
            diff = np.where(np.isnan(py) & np.isnan(r), 0.0, diff)
            scale = np.nanmax(np.abs(r)) if np.isfinite(r).any() else 0.0
            ok = bool(not nan_mismatch.any()
                      and (diff <= atol * scale + rtol * np.abs(r)).all())
            max_abs = float(np.nanmax(diff))
            rows.append(Row(case.name, key, max_abs, max_abs / scale if scale > 0 else 0.0, ok,
                            "NaN mismatch" if nan_mismatch.any() else ""))
    return rows


def run_comparison(dtype, plugins=None):
    cases = build_cases(dtype, plugins)
    payload = {"dtype": dtype,
               "cases": [{"name": c.name, "fn": c.fn, "inputs": c.inputs} for c in cases]}
    return compare(cases, run_r("numerics.R", payload), dtype)


def summarize(rows, dtype, verbose=False):
    """One line per case (worst output); all outputs of failing cases."""
    groups = ", ".join(f"{k} {v}" for k, v in TOLERANCES[dtype].items())
    lines = [f"\n== {dtype}: pass if |py - R| <= atol*max|R| + rtol*|R|; (atol, rtol): {groups}",
             f"{'case':58s} {'outputs':>7s} {'max abs':>10s} {'abs/scale':>10s}  status"]
    by_case = {}
    for row in rows:
        by_case.setdefault(row.case, []).append(row)
    for case, case_rows in by_case.items():
        ok = all(r.ok for r in case_rows)
        worst_abs = max((r.max_abs for r in case_rows), default=np.nan)
        worst_rel = max((r.max_rel for r in case_rows), default=np.nan)
        lines.append(f"{case[:58]:58s} {len(case_rows):7d} {worst_abs:10.2e} {worst_rel:10.2e}  "
                     f"{'ok' if ok else 'FAIL'}")
        if verbose or not ok:
            for r in case_rows:
                if verbose or not r.ok:
                    lines.append(f"    {r.output:40s} {r.max_abs:10.2e} {r.max_rel:10.2e}  "
                                 f"{'ok' if r.ok else 'FAIL'} {r.message}")
    n_fail = sum(not r.ok for r in rows)
    lines.append(f"{len(by_case)} cases, {len(rows)} outputs, {n_fail} failing outputs")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtype", choices=["float64", "float32", "both"], default="both")
    parser.add_argument("--verbose", action="store_true", help="show every output")
    parser.add_argument("--plugin", action="append", choices=plugin_names(),
                        help="only run the cases of compare/cases_<PLUGIN>.py (repeatable)")
    args = parser.parse_args(argv)

    sha = r_package_sha()
    print(f"R shrinkGPR commit: {sha or 'unknown'} (reference {REFERENCE_SHA})")
    if not sha or not sha.startswith(REFERENCE_SHA):
        print("WARNING: the R package is not the reference version; see compare/README.md")

    failed = False
    for dtype in (["float64", "float32"] if args.dtype == "both" else [args.dtype]):
        rows = run_comparison(dtype, args.plugin)
        print(summarize(rows, dtype, args.verbose))
        failed |= not all(r.ok for r in rows)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
