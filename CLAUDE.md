# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Python port of the R package **shrinkGPR** (https://github.com/NeferkareII/shrinkGPR, version 2.1.1). The package does variational inference for Gaussian process regression with hierarchical shrinkage priors, with the posterior approximated by normalizing flows trained in PyTorch.

- **Reference commit: `9ff1d31`.** The Python code should reproduce the numerics of that commit. Read the R source at `https://raw.githubusercontent.com/NeferkareII/shrinkGPR/9ff1d31/R/<file>.R`.
## Commands

The project uses uv with Python 3.15 (`.python-version`). The source layout is `src/shrinkgpr`.

```
uv sync                                         # install/update the environment from uv.lock
uv run pytest                                   # full test suite
uv run pytest -m "not slow"                     # skip long-running checks
uv run pytest tests/test_file.py::test_name     # single test
uv run ruff check .                             # lint
```

- **The venv lives outside OneDrive.** It is at `C:\Users\knaus\.venvs\shrinkgpr`, set through the user environment variable `UV_PROJECT_ENVIRONMENT`. Don't create a `.venv` inside the repository.
- **torch comes from the PyTorch CUDA 13.0 index** (`pytorch-cu130` in `pyproject.toml`). That index has Python 3.15 wheels and supports the RTX 5080 (Blackwell). Everything else comes from PyPI.

## Current scope: univariate `shrinkGPR` only

Port these first:
- constructing a `ShrinkGPR` model, from arrays or with `from_formula`, including the optional linear mean (`x_mean` / `formula_mean`);
- training with `.fit()`;
- prediction methods: `predict`, `calc_pred_moments`, `eval_pred_dens`, `lpds`;
- checking: `loss_stor_`, `gen_posterior_samples`, and `plot`, which draws boxplots of θ;
- `sim_gpr` (the port of `simGPR`), to simulate data for checks.

The fit depends on these R files:
- `GPR_class.R`;
- `shrinkGPR.R`, the fitting function;
- `kernel_funcs.R`;
- the TorchScript code in `shrinkGPR_package.R`;
- the flows `sylvester.R`, `IAF.R` and `NSF.R`;
- `log_hyperu.R`;
- `internal_utils.R`;
- the univariate branches of `pred_functions.R` and `plot_functions.R`.

Deferred until later:
- `shrinkTPR`, `shrinkMVGPR`, `shrinkMVTPR`;
- `gen_marginal_samples` and its plots;
- save/load.

Design the GPR model class so TPR can later subclass it, as `TPR_class` inherits from `GPR_class` in R.

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
  - stores the model specs, so `x_new` DataFrames are turned into the same columns at prediction time.
- **`fit(n_epochs=1000, n_latent=10, auto_stop=True, display_progress=True, optim_control=None)`** trains the model and returns `self`.
  - Calling `fit()` again continues from the last model and optimizer state. This replaces R's `cont_model`.
- **Fitted state** uses scikit-learn-style attribute names with a trailing underscore: `model_` (best model), `loss_`, `loss_stor_`, `last_model_`, `optimizer_`.
- **Prediction and checking are methods:**
  - `predict(x_new=None, nsamp=100)`;
  - `calc_pred_moments(x_new=None, nsamp=100)`;
  - `eval_pred_dens(y_eval, x_new, nsamp=100, log=False)`;
  - `lpds(y_new, x_new, nsamp=100)`;
  - `gen_posterior_samples(nsamp=1000)`;
  - `plot(nsamp=1000)`.

  `x_new=None` means the training data. For a model built with `from_formula`, `x_new` is a DataFrame.
- **Later models** follow the same pattern: `ShrinkTPR` (a subclass of `ShrinkGPR`), `ShrinkMVGPR` and `ShrinkMVTPR`.

The rest of this file describes R reference behavior. R names such as `shrinkGPR(...)`, `cont_model` or `LPDS` refer to the R source, not to the Python API.

## Architecture (as in the R reference)

**Model class (`GPR_class`, an `nn.Module`).** It holds the data (`y`, `x`, optional `x_mean`) and the prior hyperparameters as buffers, plus `n_layers` flow layers.

- `gen_batch(n)` draws N(0, I) samples of dimension `d`.
- `forward(z)` runs them through the flows, adding up each layer's `log_diag_j`, and then applies the positivity transform.
- `elbo(zk, log_det_J)` returns the mean log-likelihood plus the mean log-prior plus the mean log|J|.

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

**Training loop (R `shrinkGPR`, Python `ShrinkGPR.fit`).**
- Optimizer: Adam with `lr=1e-3` and `weight_decay=1e-3`, overridable through `optim_control`.
- Each epoch: `z = gen_batch(n_latent)`, then `loss = -elbo`, then a step.
- `loss_stor` records every epoch's loss. The model with the best loss so far is kept as a deep copy.
- `auto_stop`: every 100 epochs, an OLS slope is fitted to the last 100 losses. Training stops if its p-value exceeds 0.2 (two-sided t-test, see `lightweight_ols`).
- An interrupt or error during training returns the model trained so far.
- R returns `model` (the best model), `loss`, `loss_stor`, `last_model`, `optimizer` and `model_internals`, and continues training through `cont_model`. In Python these become fitted attributes and repeated `.fit()` calls; see "Python API conventions".
- The device defaults to CUDA when available.

**Formula interface (`ShrinkGPR.from_formula`, using `formulaic`).**
- The covariance formula builds the design matrix *without* an intercept.
- The mean formula keeps its intercept, which is renamed `Intercept`.
- NA values are rejected, in both the array and the formula path.
- Prediction rebuilds the design matrices from new data with the stored model specs, the equivalent of R's `terms`/`xlevels`.

**Prediction (`GPR_class` methods, run without gradients).**
- `calc_pred_moments(x_new, nsamp, x_mean_new)`:
  - computes `L = chol(K + σ²I)`;
  - mean = `K*ᵀ(K + σ²I)⁻¹y`, plus `x_mean_new β`;
  - variance = `K** - vᵀv + σ²I`;
  - returns `means` with shape `nsamp × N_new` and `vars` with shape `nsamp × N_new × N_new`.
- `predict` draws `mean + chol(var)·ε`.
- `eval_pred_dens` takes one test row. It averages normal densities over the posterior samples with log-mean-exp.
- `LPDS` (Python: `lpds`) is the log of `eval_pred_dens` evaluated at the observed y.

**`gen_posterior_samples`** pushes `gen_batch(nsamp)` through the flows (`forward`). It returns `thetas` (columns `theta_<term>`), `sigma2`, `tau`, and with a mean also `betas` (`beta_<name>`) and `tau_mean`.

## Deliberate deviations from the R reference

- **`robust_hyperu`, small-z branch for 1 < b < 2:** the constant term is `Γ(1-b)/Γ(a-b+1)`, which is negative here. R computes `exp(-lgamma(1-b) - lgamma(a-b+1))`, which is wrong by up to ~13% against mpmath.
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

They also check that training can continue. In Python that means calling `.fit()` again; save/load is deferred.
