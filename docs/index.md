# shrinkgpr

**Normalizing flows for hierarchical shrinkage in Gaussian process regression models.**

`shrinkgpr` is the Python port of the R package
[shrinkGPR](https://github.com/NeferkareII/shrinkGPR). It fits Gaussian and
Student-t process regression models with hierarchical shrinkage priors on the
kernel parameters, so that covariates without an effect are pulled out of the
model. The posterior is approximated by normalizing flows, trained by
variational inference in PyTorch, with GPU acceleration through CUDA.

## Models

| Model | Response | Likelihood |
|---|---|---|
| [`ShrinkGPR`][shrinkgpr.ShrinkGPR] | one | Gaussian process, optional linear mean |
| [`ShrinkTPR`][shrinkgpr.ShrinkTPR] | one | Student-t process, optional linear mean |
| [`ShrinkMVGPR`][shrinkgpr.ShrinkMVGPR] | several | matrix normal |
| [`ShrinkMVTPR`][shrinkgpr.ShrinkMVTPR] | several | matrix Student-t |

All four share the same workflow:

```python
from shrinkgpr import ShrinkGPR

model = ShrinkGPR.from_formula("y ~ x1 + x2", data=df).fit()   # or ShrinkGPR(y, x)

model.plot()                                   # posterior of the inverse length scales
draws = model.predict(df_new, nsamp=1000)      # posterior predictive draws
marginal = model.gen_marginal_samples("x1")    # effect of x1, with credible bands
marginal.plot()
```

## Where to go next

- [Getting started](getting-started.md) walks through simulated examples, from
  fitting to prediction, for univariate and multivariate responses.
- [Model](model.md) describes the model, the priors, the kernels and the flows.
- [Differences from R](differences.md) lists where the Python interface and
  results differ from the R package.
- The **API reference** documents every class, method and function.
