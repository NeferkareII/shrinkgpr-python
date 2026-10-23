# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Python port of the R package **shrinkGPR** (https://github.com/NeferkareII/shrinkGPR, version 2.1.1). The package does variational inference for Gaussian process regression with hierarchical shrinkage priors, with the posterior approximated by normalizing flows trained in PyTorch.

- **Reference commit: `bb488f5`.** The Python code should reproduce the numerics of that commit. Read the R source at `https://raw.githubusercontent.com/NeferkareII/shrinkGPR/bb488f5/R/<file>.R`.
  - When the reference changes, update `REFERENCE_SHA` in `compare/_rbridge.py` and the install command in `compare/README.md`, and reinstall the R package into `~/.venvs/shrinkgpr_rlib`.
  - Also check `compare/r/handlers_*.R` for hand copies of R internals, such as `mv_elbo_parts`.
## Commands

The project uses uv with Python 3.15 (`.python-version`). The source layout is `src/shrinkgpr`.

```
uv sync                                         # install/update the environment from uv.lock
uv run pytest                                   # full test suite
uv run pytest -m "not slow"                     # skip long-running checks
uv run pytest tests/test_file.py::test_name     # single test
uv run ruff check .                             # lint
uv run pytest -m "not r"                        # skip the comparisons against R
uv run python -m compare.numerics               # function-by-function comparison with R
uv run python -m compare.benchmark              # training speed, Python vs R
uv run python docs/readme_figures.py            # rerun the README examples, regenerate docs/figures
uv run --group docs mkdocs serve                # documentation site at http://127.0.0.1:8000
uv run --group docs mkdocs build --strict       # build it into site/, failing on any warning
```

- **README.** `README.md` mirrors the R package's README. Its code blocks are copies of `docs/readme_figures.py`; when changing one, change the other and regenerate the figures.
- **Documentation site.** MkDocs with the Material theme (`mkdocs.yml`, pages in `docs/`).
  - The API reference is generated from the docstrings by mkdocstrings.
  - "Getting started" includes the README between its `docs-include-start`/`docs-include-end` markers.
  - `.github/workflows/docs.yml` builds it on every push and pull request, and publishes it to GitHub Pages from `main`. CI uses Python 3.13 with only the docs group, without torch.
  - PyYAML has no Python 3.15 wheels, so `[tool.uv.extra-build-variables]` builds it without its C extension.
- **Docstrings** are numpydoc-style and rendered as Markdown.
  - Cross-reference with mkdocstrings syntax, e.g. ``[`fit`][shrinkgpr.ShrinkGPR.fit]``, not Sphinx roles such as `:meth:`.
  - Math goes in `$...$` / `$$...$$` inside raw (`r"""`) docstrings.
  - Prose after a Parameters list belongs in a Notes section; otherwise the strict build fails.

- **Comparison with R.** `compare/` checks the port against the reference R package. It is installed into the separate library `~/.venvs/shrinkgpr_rlib`; see `compare/README.md`.
  - Every newly ported function should get a case in `compare/numerics.py` and a handler in `compare/r/numerics.R`.
  - The `r`-marked tests skip themselves when R or the reference version of the package is missing.

- **Exit codes after `.backward()` can't be trusted.** With torch 2.14.1+cu130 on Python 3.15, any process that calls `.backward()` crashes at interpreter shutdown (exit 139 / 0xC0000005), even after a successful run.
  - Read pytest's summary line rather than the exit code.
  - Run scripts with `PYTHONUNBUFFERED=1` or `python -u`, so buffered output isn't lost.
- **The venv lives outside OneDrive.** It is at `C:\Users\knaus\.venvs\shrinkgpr`, set through the user environment variable `UV_PROJECT_ENVIRONMENT`. Don't create a `.venv` inside the repository.
- **torch comes from the PyTorch CUDA 13.0 index** (`pytorch-cu130` in `pyproject.toml`). That index has Python 3.15 wheels and supports the RTX 5080 (Blackwell). Everything else comes from PyPI.

## Scope

Ported for `ShrinkGPR`, `ShrinkTPR`, `ShrinkMVGPR` and `ShrinkMVTPR`:
- constructing a model, from arrays or with `from_formula`, including the optional linear mean of the univariate models (`x_mean` / `formula_mean`);
- training with `.fit()`;
- prediction methods: `predict`, `calc_pred_moments`, `eval_pred_dens`, `lpds`;
- checking: `loss_stor_`, `gen_posterior_samples`, and `plot`, which draws boxplots of θ;
- marginal effects: `gen_marginal_samples` and the 1D/2D plots of its result;
- `save` and `load`;
- `sim_gpr` and `sim_mvgpr` (the ports of `simGPR` and `simMVGPR`), to simulate data for checks.

The fit depends on these R files:
- `GPR_class.R`;
- `shrinkGPR.R`, the fitting function;
- `kernel_funcs.R`;
- the TorchScript code in `shrinkGPR_package.R`;
- the flows `sylvester.R`, `IAF.R` and `NSF.R`;
- `log_hyperu.R`;
- `internal_utils.R`;
- the univariate branches of `pred_functions.R` and `plot_functions.R`.

The further models:
- `ShrinkTPR(ShrinkGPR)` in `tpr.py` / `_tpr_module.py`;
- `ShrinkMVGPR` in `mvgpr.py` / `_mvgpr_module.py`, and `ShrinkMVTPR(ShrinkMVGPR)` in `mvtpr.py` / `_mvtpr_module.py`. R has no linear mean for these, so there is no `x_mean`.

Shared code lives in `ShrinkBase` (`_base.py`: `fit`, `plot`, `gen_marginal_samples`, `save`, formula and posterior helpers; the module-level `load`), `_formula.py` (`FormulaDesign`) and `_marginal.py` (`MarginalSamples`). The R comparison cases of the further models are plugins (`compare/cases_<model>.py`, `compare/r/handlers_<model>.R`). `tests/test_models.py` checks the behaviour shared by all four models; `tests/test_marginal.py` and `tests/test_save_load.py` cover marginal samples and saving.

All of the R package is ported.

## Python API conventions

The port uses the common Python object style: construct the model, then fit it. It does not copy R's call-and-fit-at-once functions.

```python
model = ShrinkGPR(y=y, x=x, x_mean=None, a=0.5, c=0.5, kernel_func=kernel_se, flow_func=sylvester)
model.fit(n_epochs=1000)
samples = model.predict(x_new, nsamp=100)
```

- **Two classes.** `ShrinkGPR` (PEP 8 CapWords) is a plain Python class that *holds* the internal `nn.Module` (the port of `GPR_class`). It is not an `nn.Module` itself.
- **Use `.fit()`, never `.train()`.** `nn.Module.train()` already exists, so a `.train()` method would collide with it.
- **Constructor:** `ShrinkGPR(y, x, x_mean=None, *, a=0.5, c=0.5, a_mean=0.5, c_mean=0.5, sigma2_rate=10, kernel_func=kernel_se, n_layers=10, flow_func=sylvester, flow_args=None, device=None)`.
  - Accepts numpy arrays, torch tensors or pandas objects.
  - DataFrame/Series column names are kept for labels (`theta_<col>`, `beta_<col>`).
  - Validates the arguments and builds the model, but does not train it.
- **`ShrinkGPR.from_formula("y ~ x1 + x2", data=df, formula_mean="~ x1", **kwargs)`:**
  - builds the design matrices with `formulaic`;
  - stores the model specs, so `x_new` DataFrames are turned into the same columns at prediction time;
  - the formula is available as `formula_`, which is None for models built from arrays.

  `ShrinkMVGPR.from_formula("cbind(y1, y2) ~ x1 + x2", data=df, **kwargs)` takes R's `cbind` response, or formulaic's `y1 + y2`, and has no `formula_mean`. `ShrinkTPR` and `ShrinkMVTPR` inherit `from_formula`.
- **`fit(n_epochs=1000, n_latent=10, auto_stop=True, display_progress=True, optim_control=None)`** trains the model and returns `self`.
  - Calling `fit()` again continues from the last model and optimizer state. This replaces R's `cont_model`.
- **Fitted state** uses scikit-learn-style attribute names with a trailing underscore: `model_` (best model), `loss_`, `loss_stor_`, `last_model_`, `optimizer_`.
- **Prediction and checking are methods:**
  - `predict(x_new=None, nsamp=100)`;
  - `calc_pred_moments(x_new=None, nsamp=100)`;
  - `eval_pred_dens(y_eval, x_new, nsamp=100, log=False)`;
  - `lpds(y_new, x_new, nsamp=100)`;
  - `gen_posterior_samples(nsamp=1000)`, which returns a dict: `thetas` and `betas` as DataFrames, the scalar parameters as arrays of shape `(nsamp,)`, and `Omega` as `(nsamp, M, M)`;
  - `plot(nsamp=1000, ax=None, **kwargs)`, which draws with matplotlib and returns the `Axes`;
  - `gen_marginal_samples(to_eval, nsamp=200, fixed_x=None, n_eval_points=None, eval_range=None, display_progress=True)`, which returns a `MarginalSamples` (see below);
  - `save(file)`, with the module-level `shrinkgpr.load(file, device=None)`.

  `x_new=None` means the training data. For a model built with `from_formula`, `x_new` is a DataFrame holding the variables of both formulas, and `x_mean_new` is not used.
- **Models built from arrays with a mean** also take the keyword `x_mean_new`. It is required whenever `x_new` is given. DataFrame columns are matched by the names seen in training. With several covariates, a 1-D array counts as one row.
- **`eval_pred_dens` and `lpds` take exactly one test row**, as in R. `eval_pred_dens` accepts several `y_eval` values.
- **Later models** follow the same pattern: `ShrinkTPR` (a subclass of `ShrinkGPR`), `ShrinkMVGPR` and `ShrinkMVTPR`.

The rest of this file describes R reference behavior. R names such as `shrinkGPR(...)`, `cont_model` or `LPDS` refer to the R source, not to the Python API.

## Architecture (as in the R reference)

**Model class (`GPR_class`, an `nn.Module`).** It holds the data (`y`, `x`, optional `x_mean`) and the prior hyperparameters as buffers, plus `n_layers` flow layers.

- `gen_batch(n)` draws N(0, I) samples of dimension `d`.
- `forward(z)` runs them through the flows, adding up each layer's `log_diag_j`, and then applies the positivity transform.
- `elbo(zk, log_det_J)` returns the mean log-likelihood plus the mean log-prior plus the mean log|J|.

**Python port (`_gpr_module.py`).** `GPRModule` applies softplus through a boolean `positive` mask, which is true everywhere except on the β columns. Subclasses extend it like this:
- set `n_extra` to append trailing positive parameters (ν for TPR);
- override the hooks `unpack`, `log_likelihood` and `log_prior`.

**Latent vector layout.** Columns are 0-based here; R uses 1-based indexing.
- Without a mean: `[θ_1..θ_d, σ², τ]`, so `d_latent = d + 2`.
- With a mean: `[θ_1..θ_d, σ², τ, β_1..β_p, τ_mean]`, so `d_latent = d + p + 3`.
- Softplus (`beta=0.7, threshold=20`) is applied to θ, σ², τ and τ_mean. β stays unconstrained.
- The log-Jacobian of each softplus is `beta_sp * (z - softplus(z))`.

**Priors.** These are unnormalized log densities.
- θ_j | τ ~ triple gamma: `-0.5 log τ - 0.5 log θ + log U(c+0.5, 1.5-a, a·θ/(2cτ))`.
- β_k | τ_mean ~ normal-gamma-gamma (NGG): `-0.5 log τ + log U(c+0.5, 1.5-a, a·β²/(2cτ))`.
- τ ~ F(2c, 2a) and τ_mean ~ F(2c_mean, 2a_mean), via `ldf(x, d1, d2)`.
- σ² ~ Exp(`sigma2_rate`).
- Before the kernel and likelihood are evaluated, θ, τ (tol 1e-4) and σ² go through `res_protector_autograd`: a sign-preserving `|x| ≥ tol` clamp with default tol 1e-6.

**`log_hyperu`.** This is a custom `torch.autograd.Function`.
- The forward pass computes log U(a, b, z). Its main path calls `scipy.special.hyperu` on the CPU, the equivalent of `gsl::hyperg_U`.
- For |z| < 1e-9 it uses small-z asymptotic branches chosen by the value of b (see `robust_hyperu`).
- The backward pass returns a gradient only for z: `dlogU/dz = -a · U(a+1, b+1, z) / U(a, b, z)`.

**Kernels.** Each kernel takes `(thetas [S×d], tau [S], x [N×d], x_star=None)` and returns `S×N×N`, or `S×N_new×N` when `x_star` is given.
- The squared distance is computed on θ-scaled inputs: `cdist(x·√θ, …)²`.
- SE: `(1/τ)·exp(-D/2)`.
- Matérn ½, 3/2 and 5/2 use `D = sqrt(sqdist + 1e-4)`.
- The TorchScript functions in `shrinkGPR_package.R` (`sqdist`, the kernels, `ldnorm`, `sylvester_full`) become plain Python torch functions.

**Likelihood (`ldnorm`).** It computes `-0.5 logdet(K + σ²I) - 0.5 yᵀ(K + σ²I)⁻¹y`, batched over samples, with y de-meaned by `x_mean @ β` when there is a mean. If the plain Cholesky fails, the ELBO is recomputed with `robust_chol`, which tries in turn:
1. symmetrize;
2. `cholesky_ex`;
3. 3 jitter attempts (`tol·mean(diag)`, ×100 each time);
4. an eigenvalue shift;
5. one more aggressive shift;
6. raise an error.

**Flows.** Every flow is an `nn.Module` with `forward(z) -> (zk, log_diag_j)` and receives `d` from the model. The extra `flow_args` are merged with the flow's defaults, and unknown names trigger a warning.
- `sylvester(d, n_householder=min(5, d-1))`: Householder Q, triangular R1/R2, tanh.
- `iaf(d, hidden_dim=max(64, 2d), n_hidden=1, forget_bias=2, perm=None)`: masked MADE layers with a zero-initialized output layer; σ = sigmoid(s + bias), z' = σz + (1-σ)m.
- `nsf_cl(d, K=10, B=10, hidden_dim=max(64, 2d))`: a P·L·U linear layer followed by two rational-quadratic spline coupling steps.
  - R adds 1 to the `searchsorted` result and clamps to `[1, K]` because R torch's `gather` is 1-based.
  - In Python, clamp to `[0, K-1]` and gather at `idx` and `idx + 1`.
- Random flow permutations use R's `sample.int` in the reference. In Python, draw them from torch's RNG.
- In Python the flows are the classes `Sylvester`, `IAF` and `NSFCL` in `shrinkgpr.flows`. The lowercase aliases `sylvester`, `iaf` and `nsf_cl` keep the R names for `flow_func`. `IAF(perm=...)` takes a 0-based permutation of `range(d)`.

**Training loop (R `shrinkGPR`, Python `ShrinkGPR.fit`).**
- Optimizer: Adam with `lr=1e-3` and `weight_decay=1e-3`, overridable through `optim_control`.
- Each epoch: `z = gen_batch(n_latent)`, then `loss = -elbo`, then a step.
- `loss_stor` records every epoch's loss. The model with the best loss so far is kept as a deep copy.
- `auto_stop`: every 100 epochs, an OLS slope is fitted to the last 100 losses. Training stops if its p-value exceeds 0.2 (two-sided t-test, see `lightweight_ols`).
- An interrupt or error during training returns the model trained so far.
- R returns `model` (the best model), `loss`, `loss_stor`, `last_model`, `optimizer` and `model_internals`, and continues training through `cont_model`. In Python these become fitted attributes and repeated `.fit()` calls; see "Python API conventions".
- The device defaults to CUDA when available.

**Formula interface (`ShrinkGPR.from_formula`, using `formulaic`).**
- The covariance formula builds the design matrix *without* an intercept, so a factor gets one column per level. formulaic codes factors, interactions and the column order as R's `model.matrix` does; `compare/numerics.py` checks this (`design` cases).
- The mean formula keeps its intercept, which is renamed `Intercept`.
- NA values are rejected, in both the array and the formula path.
- Prediction rebuilds the design matrices from new data with the stored model specs, the equivalent of R's `terms`/`xlevels`. Unseen factor levels raise, as in R; formulaic on its own would only warn and code them as zeros.
- Transforms use Python syntax: `np.log(x)` where R has `log(x)`.

**Prediction (`GPR_class` methods, run without gradients).**
- `calc_pred_moments(x_new, nsamp, x_mean_new)`:
  - computes `L = chol(K + σ²I)`;
  - mean = `K*ᵀ(K + σ²I)⁻¹y`, plus `x_mean_new β`;
  - variance = `K** - vᵀv + σ²I`;
  - returns `means` with shape `nsamp × N_new` and `vars` with shape `nsamp × N_new × N_new`.
- `predict` draws `mean + chol(var)·ε`.
- `eval_pred_dens` takes one test row. It averages normal densities over the posterior samples with log-mean-exp.
- `LPDS` (Python: `lpds`) is the log of `eval_pred_dens` evaluated at the observed y.

**`gen_posterior_samples`** pushes `gen_batch(nsamp)` through the flows (`forward`) of the best model.
- GPR and TPR: `thetas` (columns `theta_<term>`), `sigma2`, `tau`; with a mean also `betas` (`beta_<name>`) and `tau_mean`; TPR adds `nu` (latent + 2). These are the raw transformed latents, without zero-guards.
- MVGPR and MVTPR: `thetas`, `tau`, `sigma2`, `tau_Om` and `Omega` (built from the diagonal-biased factor `L_Om`); MVTPR adds `nu`. θ, τ (tol 1e-4) and σ² (tol 1e-6) are zero-guarded.
- `plot` draws boxplots of `thetas`, with vertical labels as R's `las = 2`.

**`gen_marginal_samples`** sweeps one or two covariates over a grid, `n_eval_points` points (default 100 for one covariate, 30 for two) over `eval_range` (default: the training range).
- For each of the `nsamp` draws, the other covariates come from `fixed_x` (its first row; the swept covariates may be left out, as in R) or from a random training row, and the predictive mean of one posterior draw (`calc_pred_moments(..., 1)`) is evaluated on the grid.
- `to_eval` names raw variables: the columns of `x` and `x_mean` for array models, the data variables of the formulas for formula models. They must be numeric. Covariance and mean columns with the same name are one covariate; if their values differ, the model refuses.
- The result `MarginalSamples(mean_pred, grid, to_eval, response)` has `mean_pred` of shape `nsamp × n` (1D) or `nsamp × n × n` (2D, `[i, j, k]` at `(grid1[j], grid2[k])`), with a trailing `M` axis for MV models. `grid` is an array, or a pair for 2D.
- `MarginalSamples.plot()` dispatches to `plot_1d` (nested quantile bands, `probs` in pairs, and the median line) or `plot_2d` (a 3D surface of the median), one panel per response.

**`save` / `load`.** `save` pickles the whole model object (fitted state, optimizer, formula specs) with `torch.save`. `load` maps to the CPU, then moves to `device`, which defaults to the saved device (falling back to the CPU with a warning when CUDA is missing). `load` only accepts files written by `save`, warns if they come from another shrinkgpr version, and must only be used on trusted files.

## Deliberate deviations from the R reference

- **Continued training keeps the best model from earlier `fit()` calls.** With `cont_model`, R resets the best model to the first epoch of the new run, even if that epoch is worse.
- **`loss_stor_` accumulates across `fit()` calls and holds only the epochs that ran.** R's `loss_stor` covers one call and is padded with NA. Auto-stop still looks only at the current call's losses.
- **Prediction outputs always have full shapes:** `nsamp × N_new` and `nsamp × N_new × N_new`. R's `$squeeze()` drops dimensions when `nsamp = 1` or `N_new = 1`.
- **Errors during `fit()`.** An error before any finite loss exists is re-raised. Later errors and `KeyboardInterrupt` warn and keep the model trained so far, as in R.
- **Factor columns are named by formulaic,** e.g. `theta_f[a]` where R writes `theta_fa`. Both name θ by design-matrix columns (R via `model_internals$x_names` since `2580f13`), so only the format differs, and the comparison checks values only.
- **`gen_posterior_samples` returns arrays of shape `(nsamp,)`** for scalar parameters, and DataFrames for `thetas` and `betas`, where R returns `nsamp × 1` matrices.
- **Random draws come from torch's RNG.** This covers TPR's chi-squared draws in `predict` (R uses `rgamma`), MVTPR's `predict` (R uses `mniw::rMT`) and everything in `sim_gpr` and `sim_mvgpr`. The comparison feeds R the TPR draws; MVTPR draws match R only in distribution.
- **Marginal samples.**
  - The 2D plot is a matplotlib 3D surface, where R uses plotly.
  - `plot_2d(response=None)` draws one surface per response in one figure. R (since `bb488f5`) draws one response per call, chosen with its 1-based `response`; pass a 0-based index or a name to `plot_2d` to do the same.
  - The random training rows come from torch's RNG; with `fixed_x` the results match R.
- **Saved files are pickles, not R's zip format,** and the two can't read each other's files.
- **MVTPR training keeps the partial model on errors.** R's `shrinkMVTPR` has its `tryCatch` commented out.
- **Stricter MV input checks:** `a_Om`, `c_Om`, `nu_alpha` and `nu_beta` must be positive, and `y` needs at least 2 columns (R gives NaN with one).
- **R behaviour worth knowing, ported as is.**
  - Training settings: the MV fits use Adam with `lr=1e-4` and clip the gradient norm to 0.5 (`ShrinkBase._adam_defaults`, `_max_grad_norm`).
  - Degrees of freedom: in TPR and MVTPR, ν = softplus output + 2, with the Gamma(`nu_alpha`, `nu_beta`) prior on ν − 2. Prediction uses ν̂ = ν + N (Shah et al. 2014).
  - The matrix t follows the Shah convention, V ~ IW(Ω, ν + M − 1), so each row is t with ν degrees of freedom. This holds for the likelihood `ldt_multi`, for `eval_pred_dens` and for `mniw::dMT`.
    - `mniw::rMT` and the port's `sample_matrix_t` use the MNIW convention instead, V ~ IW(Ω, ν).
    - So `predict` passes them ν̂ + M − 1.
  - Zero-guards:
    - TPR's ELBO doesn't guard θ, σ² or τ against zero.
    - The MVGPR σ² is clamped at 1e-4 in the ELBO but at 1e-6 in prediction. MVTPR uses 1e-4 in both (`_pred_sigma2_tol`).
- **`robust_hyperu` with b == 0:** this uses Kummer's transformation `U(a,0,z) = z·U(a+1,2,z)`. `scipy.special.hyperu` is inaccurate at b = 0 for z < 1e-3, whereas gsl, used by R, is not.

## R → Python porting pitfalls

- **Negative indices:** R torch reads them from the end, like Python, so `zk[, -1]` is the last column.
- **Slices:** R ranges such as `a:b` include both ends and are 1-based.
- **Gamma parameters:** R's `rgamma(shape, rate)` takes a rate; numpy's gamma takes `scale = 1/rate`.
- **Deprecated Cholesky:** `torch.cholesky` in the TorchScript code is deprecated. Use `torch.linalg.cholesky` or `torch.linalg.cholesky_ex`.

## Test coverage to mirror (`tests/testthat/test_shrinkGPR.R`)

The R tests fit on 15 training rows with `n_epochs=10`. Scenarios cross:
- `auto_stop` on/off;
- the SE and Matérn 3/2 kernels;
- all three flows;
- `display_progress` toggled;
- with and without `formula_mean = ~ x1 + x2`.

They check these outputs (Python method names):
- `predict`: 100 × N_test;
- `calc_pred_moments`: `means` 100 × N, `vars` 100 × N × N;
- `lpds`: a scalar;
- `eval_pred_dens`: one value per evaluation point;
- `gen_posterior_samples`: the names above.

They also check that training can continue. In Python that means calling `.fit()` again, also after `save` and `load`.
