"""Marginal predictions over a grid of one or two covariates (port of gen_marginal_samples
and its plot methods in R/pred_functions.R and R/plot_functions.R)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class MarginalSamples:
    """Posterior predictive means over a grid of one or two covariates.

    Returned by ``gen_marginal_samples``.

    Attributes
    ----------
    mean_pred : numpy.ndarray
        Predictive means, one posterior draw per row: shape (nsamp, n) for one
        covariate and (nsamp, n, n) for two, where ``mean_pred[i, j, k]`` is
        at ``(grid[0][j], grid[1][k])``. Multivariate models add a trailing
        axis of length M.
    grid : numpy.ndarray or tuple of two numpy.ndarray
        The evaluation points of the swept covariate(s).
    to_eval : list of str
        The swept covariates.
    response : list of str
        Response name(s).
    """

    mean_pred: np.ndarray
    grid: object
    to_eval: list
    response: list

    @property
    def M(self):
        """Number of responses."""
        return len(self.response)

    def _by_response(self):
        """``mean_pred`` with a trailing response axis also for univariate models."""
        return self.mean_pred if self.M > 1 else self.mean_pred[..., None]

    def plot(self, **kwargs):
        """Plot the marginal predictions.

        One covariate: median and credible bands over the grid
        (:meth:`plot_1d`). Two covariates: surface of the median
        (:meth:`plot_2d`). Keyword arguments are passed on.
        """
        return self.plot_1d(**kwargs) if len(self.to_eval) == 1 else self.plot_2d(**kwargs)

    def plot_1d(self, probs=(0.025, 0.25, 0.75, 0.975), ax=None, shade_color="skyblue",
                line_color="black", xlabel=None, ylabel=None):
        """Median and nested credible bands of the predictions over the grid.

        As ``plot.mcmc.tvp`` in R's shrinkTVP, which the R package uses: by
        default 95% and 50% bands, shaded, with the median as a line.

        Parameters
        ----------
        probs : sequence of float
            Quantiles; the outermost and innermost pairs form the bands.
        ax : matplotlib.axes.Axes or array of Axes, optional
            Axes to draw on, one per response; new ones by default.
        shade_color, line_color : color
            Colors of the bands and the median.
        xlabel, ylabel : str, optional
            Default to the covariate and the response names.

        Returns
        -------
        matplotlib.axes.Axes, or an array of Axes for several responses
        """
        import matplotlib.pyplot as plt

        if len(self.to_eval) != 1:
            raise ValueError("plot_1d needs marginal samples of a single covariate")
        probs = np.sort(np.asarray(probs, dtype=float))
        if probs.ndim != 1 or probs.size < 2 or probs.size % 2 or not (
                (probs > 0) & (probs < 1)).all():
            raise ValueError("'probs' must be an even number of quantiles in (0, 1)")

        if ax is None:
            _, axes = plt.subplots(self.M, 1, squeeze=False, sharex=True)
            axes = axes[:, 0]
        else:
            axes = np.atleast_1d(ax)
            if axes.size != self.M:
                raise ValueError(f"'ax' must hold {self.M} Axes, one per response")

        draws = self._by_response()
        n_bands = probs.size // 2
        for m, a in enumerate(axes):
            q = np.quantile(draws[..., m], probs, axis=0)
            for k in range(n_bands):
                # Inner bands are drawn on top and more opaque
                a.fill_between(self.grid, q[k], q[-1 - k], color=shade_color,
                               alpha=0.7 * (k + 1) / n_bands, linewidth=0)
            a.plot(self.grid, np.median(draws[..., m], axis=0), color=line_color)
            a.set_ylabel(ylabel if ylabel is not None else self.response[m])
        axes[-1].set_xlabel(xlabel if xlabel is not None else self.to_eval[0])
        return axes[0] if self.M == 1 else axes

    def plot_2d(self, ax=None, response=None, **kwargs):
        """3D surface of the median prediction over the grid.

        R draws an interactive plotly surface; this draws a matplotlib 3D
        surface instead.

        Parameters
        ----------
        ax : Axes3D or array of Axes3D, optional
            Axes to draw on, one per plotted response; new ones by default.
        response : int or str, optional
            The response to plot, by 0-based index or name, as ``response`` in
            R (which is 1-based). By default every response gets its own surface.
        **kwargs
            Passed to ``Axes3D.plot_surface``, e.g. ``cmap`` or ``alpha``.

        Returns
        -------
        Axes3D, or an array of Axes3D for several responses
        """
        import matplotlib.pyplot as plt

        if len(self.to_eval) != 2:
            raise ValueError("plot_2d needs marginal samples of two covariates")
        if response is None:
            responses = list(range(self.M))
        elif isinstance(response, str):
            if response not in self.response:
                raise ValueError(f"Unknown response {response!r}; one of {self.response}")
            responses = [self.response.index(response)]
        elif isinstance(response, (int, np.integer)) and not isinstance(response, bool) \
                and 0 <= response < self.M:
            responses = [int(response)]
        else:
            raise ValueError(f"'response' must be a response name or an index in [0, {self.M})")

        n = len(responses)
        if ax is None:
            fig = plt.figure(figsize=(5 * n, 4))
            axes = np.array([fig.add_subplot(1, n, k + 1, projection="3d") for k in range(n)])
        else:
            axes = np.atleast_1d(ax)
            if axes.size != n:
                raise ValueError(f"'ax' must hold {n} Axes, one per plotted response")

        g1, g2 = np.meshgrid(self.grid[0], self.grid[1], indexing="ij")
        median = np.median(self._by_response(), axis=0)
        kwargs.setdefault("cmap", "viridis")
        for m, a in zip(responses, axes, strict=True):
            a.plot_surface(g1, g2, median[..., m], **kwargs)
            a.set_xlabel(self.to_eval[0])
            a.set_ylabel(self.to_eval[1])
            a.set_zlabel(self.response[m])
        return axes[0] if n == 1 else axes
