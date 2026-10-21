"""Gaussian process regression with hierarchical shrinkage priors."""

from .flows import IAF, NSFCL, Sylvester, iaf, nsf_cl, sylvester
from .gpr import ShrinkGPR
from .kernels import kernel_matern_12, kernel_matern_32, kernel_matern_52, kernel_se
from .mvgpr import ShrinkMVGPR
from .mvtpr import ShrinkMVTPR
from .sim_gpr import sim_gpr
from .sim_mvgpr import sim_mvgpr
from .tpr import ShrinkTPR

__version__ = "0.1.0.dev0"

__all__ = [
    "IAF",
    "NSFCL",
    "ShrinkGPR",
    "ShrinkMVGPR",
    "ShrinkMVTPR",
    "ShrinkTPR",
    "Sylvester",
    "iaf",
    "kernel_matern_12",
    "kernel_matern_32",
    "kernel_matern_52",
    "kernel_se",
    "nsf_cl",
    "sim_gpr",
    "sim_mvgpr",
    "sylvester",
]
