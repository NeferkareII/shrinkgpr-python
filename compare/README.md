# Comparison with the R package

These scripts check the Python port against the R package shrinkGPR, function by function, and compare their speed.

## Setup

The comparison needs R with the packages `torch` (with its backend installed), `gsl`, `mniw`, `progress`, `rlang` and `jsonlite`. It also needs shrinkGPR at the reference commit (`REFERENCE_SHA` in `_rbridge.py`), installed into a separate library so your normal R library stays untouched:

```r
remotes::install_github("NeferkareII/shrinkGPR@2580f13",
                        lib = "C:/Users/knaus/.venvs/shrinkgpr_rlib",
                        dependencies = FALSE, upgrade = "never", force = TRUE)
```

The scripts put that library first on R's library path:
- **Library:** `$SHRINKGPR_R_LIB` if set, otherwise `~/.venvs/shrinkgpr_rlib` if it exists.
- **Rscript:** `$RSCRIPT`, then `Rscript` on the PATH, then the newest `C:\Program Files\R\R-*`.

After the reference commit changes, reinstall with the new hash and update `REFERENCE_SHA`.

## Numerics

```
uv run python -m compare.numerics                 # float64 and float32
uv run python -m compare.numerics --dtype float32 --verbose
uv run pytest -m r                                # the same check as a test
```

The cases of models beyond `ShrinkGPR` live in plugins:
- `compare/cases_<model>.py` defines `build_cases(dtype)`, which returns a list of `numerics.Case`.
- `compare/r/handlers_<model>.R` adds entries to the R `handlers` list. It can use the helpers in `common.R` and `numerics.R` (`D`, `S`, `new_gpr`, `GPR_replay`, `set_replay`, `load_state`, `grads_of`, …).
- `--plugin <model>` runs only that plugin's cases.

How it works:
- `numerics.py` evaluates each function in Python.
- It sends the same inputs to `r/numerics.R` as JSON. Flows and models also get the same parameter values, copied by name.
- It compares every output elementwise.

What it covers:
- **Building blocks:** `robust_hyperu` (including all small-z branches); `log_hyperu` and its gradient; `lightweight_ols`; `robust_chol`; `res_protector`; `sqdist` and the four kernels.
- **Flows:** `sylvester`, `iaf` and `nsf_cl`, both the forward pass and the parameter gradients.
- **Model pieces:** the priors, `ldnorm`, and the full model (flows, softplus, ELBO and every parameter gradient).
- **Training:**
  - *Replayed training.* Identical base samples are fed to R's own `shrinkGPR()` loop, through a `GPR_class` subclass passed as `cont_model`. The loss trajectories and final parameters must agree.
  - *End to end (float32).* `shrinkGPR()` and `ShrinkGPR(...).fit()` run with the same seed. R torch and PyTorch share libtorch's random number generators, and the Sylvester flow draws its random numbers in the same order in both, so the two runs coincide.

Two caveats for float64:
- **R isn't built for float64.** In float64 mode the harness sets R's default dtype and patches one cast in `robust_hyperu`: R converts the gsl result with `torch_tensor()`, which always creates float32.
- **R stays at about 1e-8 relative accuracy in some places.** R torch turns an R number on the left of an operator into a float32 tensor. `0.7 * x` uses 0.69999998807907104 even for float64 `x`, and `GPR_class` and `robust_chol` contain such expressions (`self$beta_sp * ...`, `1.1 * shift`). Those cases use a looser float64 tolerance (`r_scalar`).

## Speed

```
uv run python -m compare.benchmark                       # N = 100 and 1000, all flows, CPU and CUDA
uv run python -m compare.benchmark --sizes 500 --devices cuda --epochs 100
```

Both languages fit the same data with auto-stop off. The timing covers model construction and training, and comes after one warm-up fit per flow and device.
