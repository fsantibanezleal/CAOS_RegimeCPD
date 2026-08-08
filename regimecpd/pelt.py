"""PELT: exact multiple-changepoint segmentation in linear expected time.

This rung answers a **different question** from every other rung in the package. BOCPD and the control
charts answer "is something changing now"; PELT answers "looking back over the whole record, when did the
changes actually happen". Both are worth having and they are not substitutes.

.. rubric:: The reason this module returns segments and NOT a Detection

PELT is retrospective. It sees the entire record, including everything after the onset, before deciding
where the onset was. Feeding its output into an online detection metric would score a method that had
access to the future, and it would score spectacularly: detection delay near zero, false alarms near
zero, because it is not detecting anything, it is describing a record it has already read.

That is leakage, and it is easy to commit by accident because the output shape is so close to a detection.
So this module deliberately provides no ``detect`` and returns no
:class:`~regimecpd.types.Detection`. It returns changepoint indices, and
:func:`segmentation_error` scores them against a known onset in a way that keeps the retrospective nature
visible.

.. rubric:: What it is for here

Two jobs, both honest.

1. **Scoring onset-time error against synthetic ground truth.** When the true onset is known by
   construction, PELT gives the best retrospective estimate of where it was, which is the right yardstick
   for "how close did we get" as opposed to "how fast did we notice".
2. **Regime segmentation.** A record can be cut into operating regimes retrospectively, as an alternative
   to clustering the context, and the two can be compared.

.. rubric:: The algorithm

Optimal partitioning minimises

.. math:: \\sum_{i=1}^{m+1} \\left[ \\mathcal{C}(y_{\\tau_{i-1}+1:\\tau_i}) \\right] + \\beta m

over all segmentations, which is :math:`O(n^2)` by dynamic programming. PELT adds a pruning step: a
candidate :math:`\\tau` can be discarded permanently once

.. math:: F(\\tau) + \\mathcal{C}(y_{\\tau+1:t}) + K > F(t)

where :math:`K` satisfies :math:`\\mathcal{C}(y_{u+1:v}) + \\mathcal{C}(y_{v+1:w}) + K \\le
\\mathcal{C}(y_{u+1:w})`. For the costs used here :math:`K = 0`.

**The result is exact, not approximate.** That is the paper's contribution and it is the thing worth
testing: :func:`optimal_partition` implements the unpruned quadratic version, and the test suite asserts
the two return identical segmentations on random data. A pruning bug would otherwise show up as slightly
different changepoints, which looks like a tuning difference rather than an error.

Source: Killick, R., Fearnhead, P., Eckley, I. A. "Optimal Detection of Changepoints With a Linear
Computational Cost." *Journal of the American Statistical Association* **107**(500), 1590-1598 (2012).
doi:10.1080/01621459.2012.737745
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .types import Series

__all__ = ["PELT", "optimal_partition", "segmentation_error"]

_TINY = 1e-12


def _cumulative(x: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Prefix sums of the values and of their squares, so any segment cost is O(1)."""
    zero = np.zeros((1, x.shape[1]))
    return (np.vstack([zero, np.cumsum(x, axis=0)]),
            np.vstack([zero, np.cumsum(x ** 2, axis=0)]))


def _make_cost(x: np.ndarray, kind: str):
    """Return a segment cost function ``cost(s, t)`` for the half-open interval ``[s, t)``.

    ``"meanvar"`` is the Gaussian negative log-likelihood with both mean and variance free per segment,
    up to constants:

    .. math:: \\mathcal{C} = n \\log \\hat{\\sigma}^2

    ``"mean"`` holds the variance fixed at its whole-record estimate and lets only the mean move, giving
    the residual sum of squares. Use it when the variance genuinely is stable and only the level moves:
    ``"meanvar"`` will otherwise happily explain a level shift as a variance change, which is a real way
    to get a segmentation that fits well and means nothing.
    """
    cs, css = _cumulative(x)
    d = x.shape[1]
    total_var = np.maximum(x.var(axis=0), _TINY)

    if kind == "meanvar":
        def cost(s: int, t: int) -> float:
            n = t - s
            if n <= 0:
                return 0.0
            seg_sum = cs[t] - cs[s]
            seg_sq = css[t] - css[s]
            var = np.maximum((seg_sq - seg_sum ** 2 / n) / n, _TINY)
            return float(n * np.sum(np.log(var)))
        return cost, 2 * d

    if kind == "mean":
        def cost(s: int, t: int) -> float:
            n = t - s
            if n <= 0:
                return 0.0
            seg_sum = cs[t] - cs[s]
            seg_sq = css[t] - css[s]
            rss = np.maximum(seg_sq - seg_sum ** 2 / n, 0.0)
            return float(np.sum(rss / total_var))
        return cost, d

    raise ValueError(f"unknown cost {kind!r}; use 'meanvar' or 'mean'")


@dataclass
class PELT:
    """Exact multiple-changepoint segmentation with pruning.

    Parameters
    ----------
    penalty
        Cost of adding a changepoint. ``None`` uses BIC: ``p * log(n)`` with ``p`` the parameters a
        segment carries (``2d`` for ``"meanvar"``, ``d`` for ``"mean"``).

        This is the single most consequential setting and it is **not** a nuisance parameter to leave at
        the default. It sets how many changepoints come back, and BIC is a general-purpose choice that
        knows nothing about the machine. Sweep it and look at the curve before trusting a count.
    min_size
        Minimum segment length. Must be at least 2 for ``"meanvar"``, which cannot estimate a variance
        from one point.
    cost
        ``"meanvar"`` or ``"mean"``. See :func:`_make_cost`.
    """

    penalty: float | None = None
    min_size: int = 2
    cost: str = "meanvar"

    def __post_init__(self) -> None:
        if self.cost not in {"meanvar", "mean"}:
            raise ValueError(f"unknown cost {self.cost!r}; use 'meanvar' or 'mean'")
        if self.min_size < 1:
            raise ValueError(f"min_size must be at least 1, got {self.min_size}")
        if self.cost == "meanvar" and self.min_size < 2:
            raise ValueError("cost='meanvar' needs min_size of at least 2 to estimate a variance")

    def _prepare(self, series: Series) -> "tuple[np.ndarray, object, float]":
        x = series.x
        complete = np.all(np.isfinite(x), axis=1)
        if not complete.any():
            raise ValueError("the series has no complete samples to segment")
        # Incomplete samples are DROPPED rather than filled, and the mapping back to original indices is
        # kept, so a reported changepoint always refers to a real observation.
        xc = x[complete]
        cost, params = _make_cost(xc, self.cost)
        beta = self.penalty if self.penalty is not None else params * math.log(max(len(xc), 2))
        return complete, cost, beta

    def segment(self, series: Series) -> np.ndarray:
        """Changepoint indices into ``series``, ascending, excluding the record endpoints.

        A returned index ``i`` means a new segment BEGINS at sample ``i``.
        """
        complete, cost, beta = self._prepare(series)
        original = np.flatnonzero(complete)
        n = len(original)

        f = np.full(n + 1, np.inf)
        f[0] = -beta
        previous = np.zeros(n + 1, dtype=int)
        candidates = [0]

        for t in range(self.min_size, n + 1):
            best, best_tau = np.inf, 0
            for tau in candidates:
                if t - tau < self.min_size:
                    continue
                value = f[tau] + cost(tau, t) + beta
                if value < best:
                    best, best_tau = value, tau
            if not np.isfinite(best):
                continue
            f[t] = best
            previous[t] = best_tau

            # The pruning step, and the whole reason this is linear rather than quadratic: a candidate
            # that cannot beat the current optimum even with a zero-length remainder can never beat it
            # later either, so it is discarded permanently rather than re-examined at every future t.
            candidates = [tau for tau in candidates
                          if t - tau < self.min_size or f[tau] + cost(tau, t) <= f[t]]
            candidates.append(t)

        points = []
        t = n
        while t > 0:
            tau = previous[t]
            if tau > 0:
                points.append(tau)
            t = tau
        return original[sorted(points)] if points else np.empty(0, dtype=int)


def optimal_partition(series: Series, penalty: "float | None" = None,
                      min_size: int = 2, cost: str = "meanvar") -> np.ndarray:
    """The unpruned quadratic optimal partitioning, for verifying that PELT's pruning is exact.

    Not for production use: it is :math:`O(n^2)` and returns the same answer. It exists so the exactness
    claim can be TESTED rather than trusted, because a pruning bug produces slightly different
    changepoints, which reads as a tuning difference rather than as an error.
    """
    x = series.x
    complete = np.all(np.isfinite(x), axis=1)
    if not complete.any():
        raise ValueError("the series has no complete samples to segment")
    original = np.flatnonzero(complete)
    xc = x[complete]
    cost_fn, params = _make_cost(xc, cost)
    n = len(xc)
    beta = penalty if penalty is not None else params * math.log(max(n, 2))

    f = np.full(n + 1, np.inf)
    f[0] = -beta
    previous = np.zeros(n + 1, dtype=int)
    for t in range(min_size, n + 1):
        for tau in range(0, t - min_size + 1):
            if not np.isfinite(f[tau]):
                continue
            value = f[tau] + cost_fn(tau, t) + beta
            if value < f[t]:
                f[t] = value
                previous[t] = tau

    points = []
    t = n
    while t > 0:
        tau = previous[t]
        if tau > 0:
            points.append(tau)
        t = tau
    return original[sorted(points)] if points else np.empty(0, dtype=int)


def segmentation_error(changepoints: np.ndarray, t: np.ndarray,
                       true_onset: float) -> "tuple[float, int]":
    """Onset-time error and the number of changepoints returned, as a pair.

    Returns ``(error, n_changepoints)`` where ``error`` is the absolute time difference between
    ``true_onset`` and the NEAREST changepoint, or ``inf`` when nothing was found.

    **The count is returned alongside the error on purpose, and both must be reported together.** Taking
    the nearest changepoint is optimistic: a segmentation that cuts the record into fifty pieces will
    almost always have one landing near the true onset, and reporting only the error would present that
    as an excellent result. "Located the onset to within 3 samples" and "declared 47 other changepoints"
    are the same measurement, and quoting the first without the second is the way this metric gets
    gamed, including accidentally.
    """
    if len(changepoints) == 0:
        return float("inf"), 0
    times = np.asarray(t, dtype=float)[np.asarray(changepoints, dtype=int)]
    return float(np.min(np.abs(times - true_onset))), int(len(changepoints))
