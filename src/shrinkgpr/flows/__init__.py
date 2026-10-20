"""Normalizing flows used to approximate the posterior.

Every flow is an ``nn.Module`` built as ``flow(d, **flow_args)`` whose forward
pass maps ``z`` (n x d) to ``(zk, log_diag_j)``, where ``log_diag_j`` is the
per-sample log absolute Jacobian determinant. The lowercase aliases match the
names of the R package and are what is passed as ``flow_func``.
"""

from .iaf import IAF
from .nsf import NSFCL
from .sylvester import Sylvester

sylvester = Sylvester
iaf = IAF
nsf_cl = NSFCL

__all__ = ["IAF", "NSFCL", "Sylvester", "iaf", "nsf_cl", "sylvester"]
