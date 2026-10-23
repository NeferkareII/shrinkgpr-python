# Differences from R

The numerics of `shrinkgpr` follow the R package
[shrinkGPR](https://github.com/NeferkareII/shrinkGPR): the repository's
`compare/` directory checks the port against R function by function, and full
training runs with the Sylvester flow agree under the same seed. The interface follows Python
conventions instead of R's.

## Interface

| R | Python |
|---|---|
| `shrinkGPR(y ~ x1 + x2, data = df)` | `ShrinkGPR.from_formula("y ~ x1 + x2", data=df).fit()`, or `ShrinkGPR(y, x).fit()` |
| `res$loss_stor`, `res$model`, `res$last_model`, `res$optimizer` | `model.loss_stor_`, `model.model_`, `model.last_model_`, `model.optimizer_` |
| `shrinkGPR(..., cont_model = res)` | call `model.fit()` again |
| `predict(res, newdata = df_new)` | `model.predict(df_new)` |
| `calc_pred_moments`, `eval_pred_dens`, `LPDS`, `gen_posterior_samples`, `gen_marginal_samples` | methods of the model; `LPDS` is `lpds` |
| `save_shrinkGPR(res, file)`, `load_shrinkGPR(file)` | `model.save(file)`, `shrinkgpr.load(file)` |
| `simGPR`, `simMVGPR` | `sim_gpr`, `sim_mvgpr` |
| `optim_control` of the fit function | `optim_control` of `fit` |

## Formulas

- Formulas are strings, parsed by [formulaic](https://matthewwardrop.github.io/formulaic/).
  Factors, interactions and the column order are coded as in R's
  `model.matrix`, but transforms use Python syntax: `np.log(x)` instead of
  `log(x)`.
- Factor columns are named the formulaic way, e.g. `theta_f[a]` where R writes
  `theta_fa`.
- Multivariate responses can be written as in R, `cbind(y1, y2) ~ ...`, or as
  `y1 + y2 ~ ...`.

## Results

- **Shapes.** Prediction outputs always have shape `nsamp × N_new` (and
  `nsamp × N_new × N_new` for covariances), with a trailing `M` axis for the
  multivariate models; R drops dimensions of length one.
  Scalar parameters from `gen_posterior_samples` are arrays of shape `(nsamp,)`
  and `thetas`/`betas` are DataFrames, where R returns `nsamp × 1` matrices.
- **Random numbers.** All draws come from torch's random number generator. This
  covers the Student-t draws of `predict`, the simulators, the random
  training rows of `gen_marginal_samples` and the random permutations of the
  `IAF` and `NSFCL` flows, where R uses `sample.int`. They match R in
  distribution, not draw for draw.
- **Continued training.** `loss_stor_` accumulates over all calls to `fit` and
  holds only the epochs that ran. The best model is kept across calls, where R
  restarts it with each continuation.
- **Errors.** An error before any finite loss is raised. Later errors and
  interrupts warn and keep the model trained so far, as in R, also for
  `ShrinkMVTPR`.
- **Input checks.** The multivariate models also require positive `a_Om`,
  `c_Om`, `nu_alpha` and `nu_beta`, and at least two responses.

## Plots and files

- Plots use matplotlib. The 2D marginal plot is a static 3D surface where R uses
  plotly; for multivariate models it draws all responses at once unless
  `response` picks one. `response` is a 0-based index or a response name,
  where R's is 1-based: R's `response = 2` is `response=1` in Python.
- Saved models are pickles, written with `torch.save`. They can't be read by R,
  nor R's files by Python. Only load files you trust.
