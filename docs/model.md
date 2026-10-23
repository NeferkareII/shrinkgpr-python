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

where $\theta_j \ge 0$ are inverse squared length scales and $\tau > 0$ is the
output scale. A covariate whose $\theta_j$ is shrunk to zero has no influence
on the covariance. The other [kernels](#kernels) use the same parameters.

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

The heavier tails make the model more robust to outliers.

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
$s_m$ is fixed to 1.

[`ShrinkMVTPR`][shrinkgpr.ShrinkMVTPR] is the matrix Student-t counterpart,

$$
Y \sim \mathcal{MT}_{N,M}\left(\nu,\; 0,\; K(\theta, \tau) + \sigma^2 I_N,\; \Omega\right),
$$

in the convention of Shah et al. (2014), where each row of $Y$ is multivariate
t with $\nu$ degrees of freedom.

The multivariate models have no linear mean.

## Priors

The triple gamma (TG) prior (Cadonna et al. 2020) shrinks the inverse length
scales of irrelevant covariates towards zero:

$$
\theta_j \mid \tau \sim \mathrm{TG}(a, c, \tau), \quad j = 1, \ldots, d, \qquad
\tau \sim F(2c, 2a), \qquad
\sigma^2 \sim \mathrm{Exp}(\sigma^2_\mathrm{rate}).
$$

$a$ controls the spike at zero and $c$ the tail decay. With a linear mean, the
coefficients get a normal-gamma-gamma (NGG) prior:

$$
\beta_k \mid \tau_\mu \sim \mathrm{NGG}(a_\mu, c_\mu, \tau_\mu), \qquad
\tau_\mu \sim F(2c_\mu, 2a_\mu).
$$

The Student-t models add a prior on the degrees of freedom,

$$
\nu - 2 \sim \mathrm{Gamma}(\nu_\alpha, \nu_\beta),
$$

where the shift by 2 keeps $\nu > 2$, so the process variance is finite.

The multivariate models add priors on $\Omega = S D S$:

$$
D \sim \mathrm{LKJ}(\eta), \qquad
s_m \mid \tau_\Omega \sim \mathrm{TG}(a_\Omega, c_\Omega, \tau_\Omega), \quad m = 1, \ldots, M, \qquad
\tau_\Omega \sim F(2c_\Omega, 2a_\Omega).
$$

The LKJ prior is uniform over correlation matrices for $\eta = 1$ and
concentrates near the identity as $\eta$ grows.

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
positive parameters to $(0, \infty)$. [`fit`][shrinkgpr.ShrinkGPR.fit] trains
the flow parameters $\phi$ with Adam to maximize the evidence lower bound
(ELBO),

$$
\mathrm{ELBO}(\phi) = \mathbb{E}_{q_\phi}\left[\log p(Y \mid \vartheta) + \log p(\vartheta) - \log q_\phi(\vartheta)\right],
$$

estimated in each epoch from `n_latent` draws. With `auto_stop=True`, training
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

[`sqdist`][shrinkgpr.kernels.sqdist] computes the theta-scaled squared
distances, as the built-in kernels do. For example, a rational quadratic
kernel:

```python
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
