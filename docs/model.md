# Model

This page describes the models, priors and inference of `shrinkgpr`. It follows
the documentation of the R package; the notation is the same.

## Gaussian process regression

Given $N$ observations with $d$-dimensional covariates $x_i$,
[`ShrinkGPR`][shrinkgpr.ShrinkGPR] fits

$$
y_i = f(x_i) + \varepsilon_i, \qquad \varepsilon_i \overset{iid}{\sim} \mathcal{N}(0, \sigma^2),
$$

$$
f \sim \mathcal{GP}\left(\mu(\cdot),\; k(\cdot, \cdot\,;\, \theta, \tau)\right),
$$

so that $Y \sim \mathcal{N}(\mu,\, K + \sigma^2 I)$, where $K_{ij} = k(x_i, x_j;\, \theta, \tau)$.
The default squared exponential kernel is

$$
k(x, x';\, \theta, \tau) = \frac{1}{\tau} \exp\left(-\frac{1}{2} \sum_{j=1}^d \theta_j (x_j - x'_j)^2\right),
$$

where $\theta_j \ge 0$ are inverse squared length scales, and the output
variance is $1/\tau$. A covariate whose $\theta_j$ is shrunk to zero has no
influence on the covariance. The other [kernels](#kernels) use the same
parameters. $\tau$ is also the global shrinkage parameter of the
[prior](#priors) on the $\theta_j$, so the two roles are tied: a small $\tau$
means both strong shrinkage of the $\theta_j$ and a large output variance.

The mean is zero by default. With covariates for a linear mean (`x_mean`, or
`formula_mean` in [`from_formula`][shrinkgpr.ShrinkGPR.from_formula]), it
becomes $\mu(x_i) = x_{\mu,i}^\top \beta$.

## Student-t process regression

[`ShrinkTPR`][shrinkgpr.ShrinkTPR] replaces the Gaussian process by a
Student-t process (Shah et al. 2014): any finite collection of function values
has a joint multivariate t distribution with $\nu$ degrees of freedom. Unlike in
the Gaussian process model, the noise is added directly to the kernel, so

$$
Y \sim t_N\left(\nu,\, \mu,\, K(\theta, \tau) + \sigma^2 I\right).
$$

Following Shah et al. (2014), the third argument is the covariance of $Y$,
not the scale matrix of the usual parameterization; the scale matrix is
$\frac{\nu - 2}{\nu} (K + \sigma^2 I)$. The heavier tails make the model more robust to outliers.

## Multivariate responses

For $M$ responses, [`ShrinkMVGPR`][shrinkgpr.ShrinkMVGPR] models the response
matrix $Y \in \mathbb{R}^{N \times M}$ as matrix normal,

$$
Y \sim \mathcal{MN}_{N,M}\left(0,\; K(\theta, \tau) + \sigma^2 I_N,\; \Omega\right),
\qquad \text{i.e.} \qquad
\mathrm{vec}(Y) \sim \mathcal{N}_{NM}\left(0,\; \Omega \otimes (K + \sigma^2 I_N)\right).
$$

$\Omega$ is the $M \times M$ covariance between the responses, parameterized as
$\Omega = S D S$ with a correlation matrix $D$ and
$S = \mathrm{diag}(s_1, \ldots, s_M)$. For identifiability, the product of the
$s_m$ is fixed to 1. Two safeguards keep training stable, so the
parameterization holds only approximately at its edges: each free $\log s_m$
is bounded smoothly to $(-4, 4)$, and the diagonal of the Cholesky factor
$S L_D$ of $\Omega$ (with $D = L_D L_D^\top$) is kept above $0.03$ by a
softplus, which changes it noticeably only below about $0.3$. The ELBO
includes the log-Jacobian of that softplus,
$\sum_m \log \mathrm{sigmoid}\left(10\, (\ell_{mm} - 0.03)\right)$ for the
diagonal entries $\ell_{mm}$ of $S L_D$, which acts as a soft penalty on small
diagonal entries.

[`ShrinkMVTPR`][shrinkgpr.ShrinkMVTPR] is the matrix Student-t counterpart,

$$
Y \sim \mathcal{MT}_{N,M}\left(\nu,\; 0,\; K(\theta, \tau) + \sigma^2 I_N,\; \Omega\right),
$$

in the convention of Shah et al. (2014): $Y \mid V \sim
\mathcal{MN}_{N,M}(0, K + \sigma^2 I_N, V)$ with
$V \sim \mathcal{IW}(\Omega, \nu + M - 1)$, so each row of $Y$ is multivariate t
with $\nu$ degrees of freedom. Here $\Omega$ is a scale matrix rather than the
covariance:

$$
\mathrm{Cov}(\mathrm{vec}(Y)) = \frac{1}{\nu - 2}\, \Omega \otimes (K + \sigma^2 I_N).
$$

The multivariate models have no linear mean.

## Priors

The triple gamma prior (Cadonna et al. 2020) shrinks the inverse length
scales of irrelevant covariates towards zero. It is a gamma distribution
mixed over local scales $\lambda_j$ with an F distribution:

$$
\theta_j \mid \tau, \lambda_j \sim \mathcal{G}\left(\frac{1}{2},\, \frac{1}{2 \tau \lambda_j}\right), \qquad
\lambda_j \sim F(2a, 2c), \qquad
\tau \sim F(2c, 2a).
$$

Gamma distributions $\mathcal{G}(\cdot, \cdot)$ are given by shape and rate.
$a$ controls the spike at zero and $c$ the
tail decay. The local scales are integrated out analytically, so the model
works with the marginal density

$$
p(\theta_j \mid \tau) \propto \tau^{-1/2}\, \theta_j^{-1/2}\,
U\!\left(c + \tfrac{1}{2},\, \tfrac{3}{2} - a,\, \frac{a\, \theta_j}{2 c \tau}\right),
$$

where $U$ is the confluent hypergeometric function of the second kind.

With a linear mean, the coefficients get the same prior in its normal form, a
normal-gamma-gamma prior:

$$
\beta_k \mid \tau_\mu, \lambda_{\mu,k} \sim \mathcal{N}\left(0,\, \tau_\mu \lambda_{\mu,k}\right), \qquad
\lambda_{\mu,k} \sim F(2a_\mu, 2c_\mu), \qquad
\tau_\mu \sim F(2c_\mu, 2a_\mu),
$$

with marginal density
$p(\beta_k \mid \tau_\mu) \propto \tau_\mu^{-1/2}\,
U\!\left(c_\mu + \tfrac{1}{2},\, \tfrac{3}{2} - a_\mu,\, \frac{a_\mu \beta_k^2}{2 c_\mu \tau_\mu}\right)$.

The noise variance has an exponential prior with rate $\sigma^2_\mathrm{rate}$,

$$
\sigma^2 \sim \mathrm{Exp}(\sigma^2_\mathrm{rate}).
$$

The Student-t models add a prior on the degrees of freedom,

$$
\nu - 2 \sim \mathcal{G}(\nu_\alpha, \nu_\beta),
$$

where the shift by 2 keeps $\nu > 2$, so the process variance is finite.

The multivariate models add priors on $\Omega = S D S$:

$$
\begin{gathered}
D \sim \mathrm{LKJ}(\eta), \qquad
s_m \mid \tau_\Omega, \lambda_{\Omega,m} \sim \mathcal{G}\left(\frac{1}{2},\, \frac{1}{2 \tau_\Omega \lambda_{\Omega,m}}\right), \\
\lambda_{\Omega,m} \sim F(2a_\Omega, 2c_\Omega), \qquad
\tau_\Omega \sim F(2c_\Omega, 2a_\Omega),
\end{gathered}
$$

for $m = 1, \ldots, M$. Since the product of the scales is fixed to 1, only
$s_1, \ldots, s_{M-1}$ are free; the density above is evaluated at all $M$
scales and enters the objective as an unnormalized density on the free ones.
The LKJ prior is uniform over correlation matrices for
$\eta = 1$ and concentrates near the identity as $\eta$ grows.

All densities enter the objective without their normalizing constants.

| Python argument | Symbol |
|---|---|
| `a`, `c` | $a$, $c$ |
| `a_mean`, `c_mean` | $a_\mu$, $c_\mu$ |
| `sigma2_rate` | $\sigma^2_\mathrm{rate}$ |
| `nu_alpha`, `nu_beta` | $\nu_\alpha$, $\nu_\beta$ |
| `eta` | $\eta$ |
| `a_Om`, `c_Om` | $a_\Omega$, $c_\Omega$ |

## Inference

The posterior of all parameters is approximated by a normalizing flow
$q_\phi$: base samples $z \sim \mathcal{N}(0, I)$ pass through a stack of
`n_layers` invertible [flows](#flows), and a softplus transform maps the
positive parameters to $(0, \infty)$. The multivariate models start from a
shifted and scaled base distribution instead: the components for $\Omega$ are
scaled by $0.2/\sqrt{q}$, with $q$ the number of them, so $\Omega$ starts near
the identity, and those for $\theta$ are shifted by $-2$, so the $\theta_j$
start small. [`fit`][shrinkgpr.ShrinkGPR.fit] trains
the flow parameters $\phi$ with Adam to maximize the evidence lower bound
(ELBO),

$$
\mathrm{ELBO}(\phi) = \mathbb{E}_{q_\phi}\left[\log p(Y \mid \vartheta) + \log p(\vartheta) - \log q_\phi(\vartheta)\right],
$$

estimated in each epoch from `n_latent` draws. The loss, stored in
`loss_stor_`, is the negative ELBO up to an additive constant: the dropped
normalizing constants depend on the data and the prior hyperparameters, so
losses are only comparable between fits of the same model to the same data
with the same hyperparameters. With `auto_stop=True`, training
stops early once the loss shows no significant trend over the last 100
epochs. The model with the lowest loss is kept as `model_`.

Predictions average over draws from $q_\phi$: for each draw, the predictive
distribution at new inputs is the conditional Gaussian (or Student-t)
distribution given the training data.

## Kernels

| Kernel | $k(x, x')$ |
|---|---|
| [`kernel_se`][shrinkgpr.kernels.kernel_se] | $\frac{1}{\tau} \exp\left(-\frac{r^2}{2}\right)$ |
| [`kernel_matern_12`][shrinkgpr.kernels.kernel_matern_12] | $\frac{1}{\tau} \exp(-D)$ |
| [`kernel_matern_32`][shrinkgpr.kernels.kernel_matern_32] | $\frac{1}{\tau} (1 + \sqrt{3} D) \exp(-\sqrt{3} D)$ |
| [`kernel_matern_52`][shrinkgpr.kernels.kernel_matern_52] | $\frac{1}{\tau} (1 + \sqrt{5} D + \frac{5}{3} D^2) \exp(-\sqrt{5} D)$ |

Here $r^2 = \sum_j \theta_j (x_j - x'_j)^2$ and $D = \sqrt{r^2 + 10^{-4}}$.

### Custom kernels

Any function with the same signature can be passed as `kernel_func`:

```python
def kernel(thetas, tau, x, x_star=None):
    ...
```

- `thetas` has shape `(n_latent, d)`, `tau` shape `(n_latent,)`, `x` shape
  `(N, d)` and `x_star`, if given, shape `(N_new, d)`.
- It returns a tensor of shape `(n_latent, N, N)` for `x_star=None`, and
  `(n_latent, N_new, N)` otherwise.
- The matrices must be positive semi-definite. Use torch operations only, so
  the kernel runs on the CPU and the GPU and gradients flow through it.
- To [`save`][shrinkgpr.ShrinkGPR.save] the model, the kernel must be
  picklable: define it with `def` at the top level of a module, not as a
  lambda. A function defined in a script or notebook can only be loaded where
  it is defined again.

[`sqdist`][shrinkgpr.kernels.sqdist] computes the theta-scaled squared
distances, as the built-in kernels do. For example, a rational quadratic
kernel:

```python
from shrinkgpr import ShrinkGPR
from shrinkgpr.kernels import sqdist

def kernel_rq(thetas, tau, x, x_star=None, alpha=2.0):
    r2 = sqdist(x, thetas, x_star)
    return (1 + r2 / (2 * alpha)) ** -alpha / tau[:, None, None]

model = ShrinkGPR(y, x, kernel_func=kernel_rq)
```

## Flows

| Flow | Reference |
|---|---|
| [`Sylvester`][shrinkgpr.flows.Sylvester] (`sylvester`, the default) | van den Berg et al. (2018) |
| [`IAF`][shrinkgpr.flows.IAF] (`iaf`) | Kingma et al. (2016) |
| [`NSFCL`][shrinkgpr.flows.NSFCL] (`nsf_cl`) | Durkan et al. (2019) |

The flow is chosen with `flow_func`, and its options are passed as
`flow_args`, e.g. `ShrinkGPR(y, x, flow_func=iaf, flow_args={"hidden_dim": 32})`.

### Custom flows

A flow is a `torch.nn.Module` that the model builds once per layer as
`flow_func(d, **flow_args)`, where `d` is the number of latent parameters. Its
`forward(z)` takes `z` of shape `(n_latent, d)` and returns a tuple
`(zk, log_diag_j)`: the transformed samples, of the same shape, and the log
absolute determinant of the Jacobian, of shape `(n_latent,)`. See
[`Sylvester`][shrinkgpr.flows.Sylvester] for an example.

- The first parameter of `flow_func` must be named `d`.
- Options must be explicit keyword parameters. The model reads them from the
  signature, so options taken through `**kwargs` are dropped with a warning.
- An option without a default must be given in `flow_args`.
- As for kernels, the flow class must be picklable to save the model.

## References

- Cadonna, A., Frühwirth-Schnatter, S., & Knaus, P. (2020). Triple the gamma —
  A unifying shrinkage prior for variance and variable selection in sparse
  state space and TVP models. *Econometrics*, 8(2), 20.
- Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). Neural spline
  flows. *Advances in Neural Information Processing Systems*, 32.
- Kingma, D. P., Salimans, T., Jozefowicz, R., Chen, X., Sutskever, I., &
  Welling, M. (2016). Improved variational inference with inverse
  autoregressive flow. *Advances in Neural Information Processing Systems*, 29.
- Shah, A., Wilson, A., & Ghahramani, Z. (2014). Student-t processes as
  alternatives to Gaussian processes. In *Artificial Intelligence and
  Statistics* (pp. 877–885). PMLR.
- van den Berg, R., Hasenclever, L., Tomczak, J. M., & Welling, M. (2018).
  Sylvester normalizing flows for variational inference. In *Proceedings of the
  Conference on Uncertainty in Artificial Intelligence (UAI)*.
