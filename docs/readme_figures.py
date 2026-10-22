"""Run the examples of README.md and save its figures to docs/figures.

    PYTHONUNBUFFERED=1 uv run python docs/readme_figures.py

The code blocks here match those in README.md; keep them in sync.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from shrinkgpr import ShrinkGPR, ShrinkMVGPR, ShrinkMVTPR, ShrinkTPR, sim_gpr, sim_mvgpr

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


def save(name, size=(7, 4.5)):
    plt.gcf().set_size_inches(*size)
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", dpi=96)
    plt.close("all")
    print("saved", name)


print("torch.cuda.is_available():", torch.cuda.is_available())

# Univariate response models

torch.manual_seed(117)
sim = sim_gpr(N=25, theta=[10, 0, 0], d_mean=0, sigma2=0.1)

model = ShrinkGPR.from_formula("y ~ .", data=sim["data"]).fit(display_progress=False)
model_t = ShrinkTPR.from_formula("y ~ .", data=sim["data"]).fit(display_progress=False)

plt.plot(model.loss_stor_)
save("loss")

model.plot()
save("thetas")

samples = model.gen_posterior_samples(nsamp=1000)

fig, axes = plt.subplots(1, 2)
samples["thetas"]["theta_x.1"].plot.kde(ax=axes[0], title=r"Posterior distribution of $\theta_1$")
pd.Series(samples["sigma2"]).plot.kde(ax=axes[1], title=r"Posterior distribution of $\sigma^2$")
save("posterior", (10, 4))

x1 = sim["data"]["x.1"]
newdata = pd.DataFrame({"x.1": np.linspace(x1.min(), x1.max(), 200), "x.2": 0.0, "x.3": 0.0})
preds = model.calc_pred_moments(newdata, nsamp=1000)

preds_samples = model.predict(newdata, nsamp=1000)

q05, q50, q95 = np.quantile(preds["means"], [0.05, 0.5, 0.95], axis=0)

plt.scatter(x1, sim["data"]["y"], color="black")
plt.plot(newdata["x.1"], q50, color="forestgreen", linewidth=2)
plt.fill_between(newdata["x.1"], q05, q95, color="skyblue", alpha=0.3, linewidth=0)
plt.xlabel("x1")
plt.ylabel("y")
save("predictions")

marginal_samples = model.gen_marginal_samples("x.1", fixed_x={"x.2": 0, "x.3": 0},
                                              display_progress=False)
marginal_samples.plot()
save("marginal")

# Multivariate response models

torch.manual_seed(42)
sim_mv = sim_mvgpr(N=200, M=3, d=3, theta=[3, 0, 0], sigma2=0.1)

model_mv = ShrinkMVGPR.from_formula("cbind(y.1, y.2, y.3) ~ .", data=sim_mv["data"]).fit(
    display_progress=False, optim_control={"lr": 1e-3})
model_mv_t = ShrinkMVTPR.from_formula("cbind(y.1, y.2, y.3) ~ .", data=sim_mv["data"]).fit(
    display_progress=False)

plt.plot(model_mv.loss_stor_)
save("loss_mv")

model_mv.plot()
save("thetas_mv")

samples_mv = model_mv.gen_posterior_samples(nsamp=1000)

fig, axes = plt.subplots(1, 2)
samples_mv["thetas"]["theta_x.1"].plot.kde(ax=axes[0],
                                           title=r"Posterior distribution of $\theta_1$")
pd.Series(samples_mv["sigma2"]).plot.kde(ax=axes[1], title=r"Posterior distribution of $\sigma^2$")
save("posterior_mv", (10, 4))

marg = model_mv.gen_marginal_samples("x.1", fixed_x={"x.2": 0, "x.3": 0},
                                     display_progress=False)
marg.plot()
save("marginal_mv", (7, 7))
