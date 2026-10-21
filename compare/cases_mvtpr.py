"""Comparison cases of the multivariate TPR (R handlers in r/handlers_mvtpr.R).

R's predict() for MVTPR samples with mniw::rMT, which uses R's own random
number generator, so prediction draws are not compared; the predictive
moments and densities are.
"""

import numpy as np
import torch

from shrinkgpr._mvtpr_module import MVTPRModule, ldt_multi
from shrinkgpr.mvtpr import ShrinkMVTPR

from .cases_mvgpr import build_model_cases
from .numerics import Case


def build_cases(dtype):
    dt = torch.float64 if dtype == "float64" else torch.float32
    rng = np.random.default_rng(2028)

    def T(arr):
        return torch.as_tensor(np.asarray(arr, dtype=float), dtype=dt)

    cases = []

    # Matrix-t likelihood from Cholesky factors (R's MVTPR_class$ldt takes K
    # and sigma2, the handler passes K + sigma2 I with sigma2 = 0)
    for N, M in [(7, 2), (5, 4)]:
        a = rng.normal(size=(3, N, N))
        b = rng.normal(size=(3, M, M))
        K = T(a @ a.transpose(0, 2, 1) + N * np.eye(N))
        L_Om = torch.linalg.cholesky(T(b @ b.transpose(0, 2, 1) + np.eye(M)))
        y = T(rng.normal(size=(N, M)))
        nu = T([0.7, 3.0, 25.0])
        cases.append(Case(f"mvtpr ldt N={N} M={M}", "mv_ldt",
                          {"K": K, "L_Om": L_Om, "y": y, "nu": nu},
                          {"ll": ldt_multi(torch.linalg.cholesky(K), L_Om, y, nu)}))

    # Gamma prior on nu
    nu = T(rng.uniform(0.05, 30, size=8))
    for alpha, beta in [(0.5, 2.0), (2.0, 0.1)]:
        cases.append(Case(f"mvtpr ldg alpha={alpha} beta={beta}", "mv_ldg",
                          {"x": nu, "alpha": alpha, "beta": beta},
                          {"ldg": MVTPRModule.ldg(nu, T(alpha), T(beta))}))

    return cases + build_model_cases(dtype, "MVTPR", MVTPRModule, ShrinkMVTPR,
                                     extra={"nu_alpha": 0.5, "nu_beta": 2.0}, seed=2,
                                     with_predict=False)
