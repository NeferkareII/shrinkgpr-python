"""Running R scripts with JSON input and output."""

import glob
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

R_DIR = Path(__file__).parent / "r"
REFERENCE_SHA = "011c9ca"
# Separate R library holding the reference version of shrinkGPR (see README)
DEFAULT_R_LIB = Path.home() / ".venvs" / "shrinkgpr_rlib"


def find_rscript():
    """Path to Rscript: $RSCRIPT, then PATH, then the newest R under Program Files."""
    if os.environ.get("RSCRIPT"):
        return os.environ["RSCRIPT"]
    found = shutil.which("Rscript")
    if found:
        return found
    candidates = sorted(glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe"))
    return candidates[-1] if candidates else None


def r_env():
    env = dict(os.environ)
    if not env.get("SHRINKGPR_R_LIB") and DEFAULT_R_LIB.is_dir():
        env["SHRINKGPR_R_LIB"] = str(DEFAULT_R_LIB)
    return env


def r_package_sha():
    """Git commit of the R shrinkGPR the comparison will load (None if unknown)."""
    rscript = find_rscript()
    if rscript is None:
        return None
    code = ('lib <- Sys.getenv("SHRINKGPR_R_LIB"); if (nzchar(lib)) .libPaths(c(lib, .libPaths()));'
            'cat(tryCatch(packageDescription("shrinkGPR")$RemoteSha, error = function(e) ""))')
    out = subprocess.run([rscript, "-e", code], check=False, capture_output=True, text=True,
                         env=r_env())
    return out.stdout.strip() or None


def encode(value):
    """Convert tensors and arrays (also nested in dicts and lists) to tensor JSON."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().double().numpy()
    if isinstance(value, np.ndarray):
        return {"__tensor__": True, "shape": list(value.shape),
                "data": value.astype(float).ravel().tolist()}
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def decode(value):
    if isinstance(value, dict) and value.get("__tensor__"):
        data = value["data"]
        data = [data] if not isinstance(data, list) else data
        arr = np.array([np.nan if v is None else v for v in data], dtype=float)
        return arr.reshape(value["shape"])
    if isinstance(value, dict):
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def run_r(script, payload, timeout=1800):
    """Run ``compare/r/<script>`` on a JSON payload and return its decoded output."""
    rscript = find_rscript()
    if rscript is None:
        raise RuntimeError("Rscript not found; set the RSCRIPT environment variable")
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "input.json", Path(tmp) / "output.json"
        inp.write_text(json.dumps(encode(payload)))
        proc = subprocess.run([rscript, str(R_DIR / script), str(inp), str(out)],
                              check=False, capture_output=True, text=True, env=r_env(),
                              timeout=timeout)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"R script {script} failed:\n{proc.stderr[-3000:]}")
        return decode(json.loads(out.read_text()))
