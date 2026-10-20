"""Training speed of the Python port against the R package.

Both sides fit the same data with the same flow, device and number of epochs
(auto-stop off, no progress display), after one warm-up fit per flow and device.
The timing covers model construction and training, as a call to shrinkGPR() in
R does.

Usage: uv run python -m compare.benchmark [--sizes 100 1000] [--devices cpu cuda]
"""

import argparse
import sys
import time

import numpy as np
import torch

from shrinkgpr import ShrinkGPR, iaf, nsf_cl, sylvester

from ._rbridge import REFERENCE_SHA, r_package_sha, run_r

FLOWS = {"sylvester": sylvester, "iaf": iaf, "nsf_cl": nsf_cl}


def make_data(N, p=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=(N, p))
    y = np.sin(2 * x[:, 0]) + 0.1 * rng.normal(size=N)
    return y, x


def default_epochs(N):
    return 200 if N <= 200 else 50


def time_python(configs):
    warmed = set()
    results = []
    for cfg in configs:
        y, x = cfg["y"], cfg["x"]
        flow, device = FLOWS[cfg["flow"]], cfg["device"]
        if (cfg["flow"], device) not in warmed:
            ShrinkGPR(y, x, flow_func=flow, device=device).fit(n_epochs=5, display_progress=False)
            warmed.add((cfg["flow"], device))
        torch.manual_seed(1)
        start = time.perf_counter()
        model = ShrinkGPR(y, x, flow_func=flow, device=device)
        model.fit(n_epochs=cfg["n_epochs"], auto_stop=False, display_progress=False)
        if device == "cuda":
            torch.cuda.synchronize()
        results.append({"seconds": time.perf_counter() - start,
                        "epochs": len(model.loss_stor_), "loss": model.loss_})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1000])
    parser.add_argument("--flows", nargs="+", default=list(FLOWS), choices=list(FLOWS))
    parser.add_argument("--devices", nargs="+", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="epochs per fit (default: 200 for N <= 200, else 50)")
    args = parser.parse_args(argv)
    devices = args.devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])

    sha = r_package_sha()
    print(f"R shrinkGPR commit: {sha or 'unknown'} (reference {REFERENCE_SHA})")

    configs = []
    for N in args.sizes:
        y, x = make_data(N)
        for device in devices:
            for flow in args.flows:
                configs.append({"N": N, "flow": flow, "device": device, "y": y, "x": x,
                                "n_epochs": args.epochs or default_epochs(N)})

    py = time_python(configs)
    r = run_r("benchmark.R", {"configs": configs})

    print(f"Python: torch {torch.__version__}, {torch.get_num_threads()} CPU threads | "
          f"R: torch {r['torch_version']}, {int(r['threads'])} CPU threads")
    print(f"{'N':>6} {'device':>6} {'flow':>10} {'epochs':>6} {'Python it/s':>12} "
          f"{'R it/s':>9} {'Py/R':>6} {'best loss Py':>13} {'best loss R':>12}")
    for cfg, p, q in zip(configs, py, r["results"], strict=True):
        p_rate = p["epochs"] / p["seconds"]
        q_rate = float(q["epochs"]) / float(q["seconds"])
        epochs = f"{p['epochs']}" if p["epochs"] == int(q["epochs"]) else \
            f"{p['epochs']}/{int(q['epochs'])}"
        print(f"{cfg['N']:6d} {cfg['device']:>6} {cfg['flow']:>10} {epochs:>6} {p_rate:12.1f} "
              f"{q_rate:9.1f} {p_rate / q_rate:6.2f} {p['loss']:13.2f} {float(q['loss']):12.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
