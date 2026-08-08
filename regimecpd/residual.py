"""The within-regime residual: what is left of a channel once its operating context is accounted for.

Segmentation says which context a sample was recorded in. This module says what "normal for that context"
was, and subtracts it. The output is the object arm B of the central comparison runs on, while arm A runs
the same detector on the raw channels.

Two residual models, in increasing strength:

``"zscore"``
    Per regime, subtract that regime's baseline mean and divide by its baseline standard deviation. This
    is condition normalisation in its plainest form: a loaded reading is compared against loaded normal.

``"linear"``
    Per regime, regress each monitored channel on the context variables and subtract the prediction. This
    additionally removes the variation WITHIN a regime that is still explained by context, which matters
    whenever the regime is coarse (six clusters standing in for a continuum of grades).

.. rubric:: Leakage, which is the way this stage actually fails

The model is fitted on a **healthy baseline** and applied to the monitored record. Fit it on the record
being monitored and the fault is folded into the definition of normal, so the residual has the fault
partly subtracted out of it. Nothing raises, the residual looks well behaved, and detection performance
degrades for a reason invisible in every output. :func:`make_arms` exists so the baseline and the
monitored record are two separate arguments and the split is impossible to leave implicit.

.. rubric:: Samples that cannot be residualised become NaN

A regime with too few baseline samples cannot support a mean, let alone a standard deviation, and a
sample labelled ``-1`` has no fitted local model at all. Those samples are ``NaN`` in the residual rather
than filled with a pooled or nearest-regime estimate. NaN keeps the time axis intact (so time-based
windows and exposure stay correct) and it is visible: every detector in this package treats NaN as
"statistic undefined here" and refuses to alarm on it, and ``Residual.regimes.coverage`` reports how much
of the record was usable. Filling instead would manufacture a plausible number for the one situation
where the method has nothing to say.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .regime import DiscreteRegimes, KMeansRegimes
from .types import RegimeLabels, Residual, Series

__all__ = ["RegimeResidualizer", "make_arms"]


def _design(context: np.ndarray) -> np.ndarray:
    """Context with an intercept column appended."""
    return np.hstack([context, np.ones((len(context), 1))])


@dataclass
class RegimeResidualizer:
    """Fit "normal for this context" on a baseline, then subtract it from a monitored record.

    Parameters
    ----------
    method
        ``"zscore"`` or ``"linear"``. See the module docstring.
    min_samples
        Baseline samples a regime needs before its statistics are trusted. A regime below this is marked
        unusable and its samples come back as NaN. The default of 30 is a convention, not a theorem; the
        point is that some floor exists, because a standard deviation from four points is noise.
    ridge
        Ridge term for the ``"linear"`` least squares, which keeps the fit finite when context columns
        are collinear inside a regime. That happens constantly: within one operating regime the context
        variables barely move, which is what made it a regime.
    """

    method: str = "zscore"
    min_samples: int = 30
    ridge: float = 1e-6
    names_: tuple = ()
    mean_: dict = field(default_factory=dict)
    scale_: dict = field(default_factory=dict)
    coef_: dict = field(default_factory=dict)
    usable_: set = field(default_factory=set)

    def fit(self, baseline: Series, labels: RegimeLabels,
            context: "np.ndarray | None" = None) -> "RegimeResidualizer":
        """Fit on a HEALTHY baseline. Read the module docstring on leakage before using this."""
        if self.method not in {"zscore", "linear"}:
            raise ValueError(f"unknown method {self.method!r}; use 'zscore' or 'linear'")
        if len(labels.labels) != baseline.n:
            raise ValueError(f"{len(labels.labels)} labels for {baseline.n} baseline samples")
        if self.method == "linear" and context is None:
            raise ValueError("method='linear' needs context; there is nothing to regress on without it")

        self.names_ = baseline.names
        self.mean_, self.scale_, self.coef_, self.usable_ = {}, {}, {}, set()
        ctx = None if context is None else np.atleast_2d(np.asarray(context, dtype=float))
        if ctx is not None and ctx.shape[0] != baseline.n:
            ctx = ctx.T
        if ctx is not None and ctx.shape[0] != baseline.n:
            raise ValueError(f"context has {ctx.shape[0]} rows for {baseline.n} baseline samples")

        for k in range(labels.n_regimes):
            member = labels.labels == k
            n_member = int(np.count_nonzero(member))
            if n_member < self.min_samples:
                continue

            xk = baseline.x[member]
            if self.method == "zscore":
                self.mean_[k] = xk.mean(axis=0)
                residual = xk - self.mean_[k]
            else:
                design = _design(ctx[member])
                gram = design.T @ design + self.ridge * np.eye(design.shape[1])
                self.coef_[k] = np.linalg.solve(gram, design.T @ xk)
                residual = xk - design @ self.coef_[k]
                self.mean_[k] = residual.mean(axis=0)
                residual = residual - self.mean_[k]

            spread = residual.std(axis=0)
            # A channel that never moves inside a regime carries no information there. Dividing by its
            # spread would turn rounding noise into an infinite residual, which then dominates every
            # multivariate statistic downstream.
            self.scale_[k] = np.where(spread > 0, spread, 1.0)
            self.usable_.add(k)

        if not self.usable_:
            raise ValueError(
                f"no regime reached min_samples={self.min_samples}; "
                f"either the baseline is too short or n_regimes is too high for it"
            )
        return self

    def transform(self, monitored: Series, labels: RegimeLabels,
                  context: "np.ndarray | None" = None) -> Residual:
        """Residualise a monitored record against the fitted baseline."""
        if not self.usable_:
            raise RuntimeError("RegimeResidualizer.transform called before fit")
        if monitored.names != self.names_:
            raise ValueError(f"fitted on channels {self.names_}, asked to transform {monitored.names}")
        if len(labels.labels) != monitored.n:
            raise ValueError(f"{len(labels.labels)} labels for {monitored.n} samples")

        ctx = None
        if context is not None:
            ctx = np.atleast_2d(np.asarray(context, dtype=float))
            if ctx.shape[0] != monitored.n:
                ctx = ctx.T
        if self.method == "linear" and ctx is None:
            raise ValueError("method='linear' needs context at transform time too")

        r = np.full_like(monitored.x, np.nan, dtype=float)
        out_labels = labels.labels.copy()

        for k in sorted(self.usable_):
            member = labels.labels == k
            if not np.any(member):
                continue
            xk = monitored.x[member]
            if self.method == "zscore":
                centred = xk - self.mean_[k]
            else:
                centred = xk - _design(ctx[member]) @ self.coef_[k] - self.mean_[k]
            r[member] = centred / self.scale_[k]

        # A sample in a regime that never became usable is as unresidualisable as one in no regime at
        # all, so it is reported the same way rather than left looking assigned.
        unusable = ~np.isin(labels.labels, list(self.usable_))
        out_labels[unusable] = -1

        return Residual(
            monitored.t, r, monitored.names,
            RegimeLabels(out_labels, labels.n_regimes, labels.method,
                         {**labels.meta, "usable_regimes": sorted(self.usable_),
                          "residual_method": self.method}),
            monitored.unit_id,
        )

    def fit_transform(self, series: Series, labels: RegimeLabels,
                      context: "np.ndarray | None" = None) -> Residual:
        """Fit and transform the SAME record.

        Legitimate for inspecting the residual on a baseline that is known healthy. It is **not** how a
        monitored record is residualised: doing that leaks the fault into the definition of normal. Use
        :func:`make_arms` for the monitored case, where the split is an argument rather than a habit.
        """
        return self.fit(series, labels, context).transform(series, labels, context)


def make_arms(baseline: Series, monitored: Series,
              context_names: "tuple[str, ...] | list[str]",
              n_regimes: "int | None" = None,
              monitor_names: "tuple[str, ...] | list[str] | None" = None,
              method: str = "zscore",
              discrete: bool = False,
              min_samples: int = 30,
              seed: int = 0) -> "tuple[Series, Series, RegimeLabels]":
    """Build both arms of the central comparison from one baseline and one monitored record.

    Returns ``(raw_arm, residual_arm, labels)``. Both arms are plain :class:`~regimecpd.types.Series`
    over the same channels and the same time axis, so the identical detector runs on each and cannot tell
    them apart.

    This function exists because the comparison is easy to get subtly wrong in ways that all favour the
    residual arm: fitting the regime model on the monitored record, using different channels in the two
    arms, or calibrating on different data. Here the baseline is an argument, the monitored record is an
    argument, and the channel set is shared by construction.

    Parameters
    ----------
    baseline
        A healthy record used to fit the regime model and the residual model. Nothing from the monitored
        record informs either.
    monitored
        The record to score. Both returned arms are over this record.
    context_names
        Channels treated as operating context. These are what regimes are built from.
    n_regimes
        Number of clusters for the discovered route. Ignored when ``discrete`` is true, where the number
        of regimes is however many distinct context combinations the baseline contained.
    monitor_names
        Channels to monitor. Defaults to every channel that is not context: you do not monitor the thing
        you are conditioning on, since its residual is zero by construction.
    discrete
        Use the observed-regime route (distinct context combinations) instead of clustering. Running both
        and comparing measures the price of not being told the regime.

    Notes
    -----
    The raw arm is the monitored channels untouched, deliberately **not** standardised. Standardising it
    would be a partial regime correction (a global mean is a one-regime model) and would blur the very
    contrast being measured. If a detector needs a scale, it must establish one from the baseline itself,
    the same way it would in a deployment that had never heard of this package.
    """
    context_names = tuple(context_names)
    monitor_names = tuple(monitor_names) if monitor_names is not None else tuple(
        nm for nm in monitored.names if nm not in context_names
    )
    if not monitor_names:
        raise ValueError("no channels left to monitor once context channels were removed")

    base_ctx = baseline.select(context_names).x
    mon_ctx = monitored.select(context_names).x

    if discrete:
        segmenter = DiscreteRegimes().fit(base_ctx)
    else:
        if n_regimes is None:
            raise ValueError("n_regimes is required for the clustered route")
        segmenter = KMeansRegimes(n_regimes=n_regimes, seed=seed).fit(base_ctx)

    base_labels = segmenter.label(base_ctx)
    mon_labels = segmenter.label(mon_ctx)

    base_mon = baseline.select(monitor_names)
    raw_arm = monitored.select(monitor_names)

    resid = RegimeResidualizer(method=method, min_samples=min_samples)
    resid.fit(base_mon, base_labels, base_ctx if method == "linear" else None)
    residual = resid.transform(raw_arm, mon_labels, mon_ctx if method == "linear" else None)

    return raw_arm, residual.as_series(), residual.regimes
