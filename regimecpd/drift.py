"""Streaming drift detectors: ADWIN and KSWIN.

These belong to a different family from the control charts. A control chart compares the present against
a **fixed baseline** established once by ``fit``. A drift detector compares the recent past against the
less recent past, carrying no baseline at all, and adapts as the machine changes.

That difference matters for a fleet. A truck that is rebuilt, re-shod or reassigned to a different pit has
legitimately changed, and a fixed baseline will alarm on it forever. A drift detector reports the change
once and then accepts the new normal, which is either exactly right or exactly wrong depending on whether
the change was a repair or a fault. Neither family subsumes the other, and running both is informative
precisely because they disagree about permanent changes.

.. rubric:: Why ADWIN earns a place

Most drift detectors are heuristics with a tuning knob. ADWIN carries **rigorous bounds on both the false
positive and the false negative rate**, governed by a single confidence parameter, which makes it the one
detector in this package that arrives with a guarantee attached rather than a threshold to sweep. That is
worth having next to conformal calibration as an independent route to the same property.

Sources:

- Bifet, A., Gavalda, R. "Learning from Time-Changing Data with Adaptive Windowing."
  *Proceedings of the 2007 SIAM International Conference on Data Mining (SDM)*, pp. 443-448.
- Raab, C., Heusinger, M., Schleif, F.-M. "Reactive Soft Prototype Computing for Concept Drift Streams."
  *Neurocomputing* **416**, 340-351 (2020), which introduces KSWIN.
  **UNVERIFIED:** the KSWIN reference was not opened against a primary source during the research pass
  behind this package, and the implementation follows the standard formulation rather than a specific
  paper's notation.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .types import Detection, Series

__all__ = ["ADWIN", "KSWIN", "ks_two_sample_pvalue"]


@dataclass
class ADWIN:
    """Adaptive windowing: keep a window, and cut it wherever two halves differ significantly.

    The window grows while the stream is stationary. Whenever some split point divides it into two
    sub-windows whose means differ by more than the Hoeffding-style bound

    .. math::
        \\epsilon_{\\mathrm{cut}} = \\sqrt{\\frac{2}{m}\\,\\sigma^2_W \\ln\\frac{2}{\\delta'}}
        + \\frac{2}{3m}\\ln\\frac{2}{\\delta'}, \\qquad
        \\frac{1}{m} = \\frac{1}{n_0 - 1} + \\frac{1}{n_1 - 1}

    the older part is dropped. ``delta`` is the confidence: the paper bounds the false positive rate by it.

    .. rubric:: What is implemented here, and what is not

    This is the **direct** ADWIN, which examines every split point of an explicit window. The paper's
    ADWIN2 keeps an exponential-histogram summary so that memory is logarithmic and the amortised cost
    per item is much lower.

    ADWIN2 is **not** implemented, and that is stated rather than glossed. The direct version returns the
    same cuts (it is the definition ADWIN2 approximates efficiently), so nothing about detection quality
    is lost, but the cost per item grows with the window. ``max_window`` bounds it. On a record of a few
    tens of thousands of samples this is fine; on a genuinely unbounded stream it is not, and ADWIN2 would
    be the thing to add.

    The theorem statement and the ADWIN2 complexity bounds were **not verified against the primary source**
    during the research pass behind this package (the PDF fetch returned compressed binary), so the
    guarantee is described here as the paper is generally reported rather than quoted.

    Parameters
    ----------
    delta
        Confidence parameter. Smaller means fewer false detections and slower reaction.
    min_sub
        Minimum samples on each side of a candidate split. Below about 5 the bound is loose enough that
        the detector fires on noise.
    max_window
        Cap on retained samples, which bounds the per-item cost of the direct algorithm.
    """

    delta: float = 0.002
    min_sub: int = 5
    max_window: int = 2000
    grace: int = 10
    _window: deque = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")
        if self.min_sub < 2:
            raise ValueError(f"min_sub must be at least 2, got {self.min_sub}")

    def reset(self) -> None:
        self._window.clear()

    def _cut_threshold(self, n0: int, n1: int, variance: float) -> float:
        n = n0 + n1
        # The harmonic term is the paper's 1/m; using n0 and n1 directly instead would understate the
        # bound for a lopsided split, which is exactly where a spurious cut is easiest to make.
        m = 1.0 / (1.0 / max(n0 - 1, 1) + 1.0 / max(n1 - 1, 1))
        delta_prime = self.delta / max(n, 1)
        log_term = math.log(2.0 / delta_prime)
        return math.sqrt(2.0 / m * variance * log_term) + 2.0 / (3.0 * m) * log_term

    def update(self, value: float) -> bool:
        """Add one observation. Returns True when a change was detected at this step."""
        self._window.append(float(value))
        while len(self._window) > self.max_window:
            self._window.popleft()

        n = len(self._window)
        if n < 2 * self.min_sub or n < self.grace:
            return False

        window = np.fromiter(self._window, dtype=float, count=n)
        variance = float(window.var())
        prefix = np.concatenate([[0.0], np.cumsum(window)])

        detected = False
        # Search from the OLDEST split inward: the paper drops the older part, so finding the earliest
        # valid cut removes the most stale data. Searching from the newest end would shave one sample at
        # a time and leave most of the outdated window in place.
        for split in range(self.min_sub, n - self.min_sub + 1):
            mean0 = prefix[split] / split
            mean1 = (prefix[n] - prefix[split]) / (n - split)
            if abs(mean0 - mean1) > self._cut_threshold(split, n - split, variance):
                for _ in range(split):
                    self._window.popleft()
                detected = True
                break
        return detected

    def detect(self, series: Series) -> Detection:
        """Run per channel and report the number of channels flagging a cut at each step.

        The statistic is a small integer count rather than a continuous score, which makes ADWIN a poor
        fit for the alarm-budget curve: there are only ``d + 1`` distinct thresholds, so its curve has
        ``d + 1`` points where the others have hundreds. That is a real limitation of putting a detector
        with a built-in decision rule onto a common budget, and it is the reason ``delta`` should be swept
        instead when ADWIN is being compared against anything.
        """
        detectors = [ADWIN(self.delta, self.min_sub, self.max_window, self.grace)
                     for _ in range(series.d)]
        flags = np.zeros((series.n, series.d))
        for i in range(series.n):
            row = series.x[i]
            for j in range(series.d):
                if np.isfinite(row[j]):
                    flags[i, j] = float(detectors[j].update(row[j]))
                else:
                    flags[i, j] = np.nan
        with np.errstate(invalid="ignore"):
            statistic = np.where(np.all(np.isnan(flags), axis=1), np.nan, np.nansum(flags, axis=1))
        return Detection(series.t, statistic, "adwin",
                         {"per_channel": flags, "names": series.names, "delta": self.delta})


def ks_two_sample_pvalue(a: np.ndarray, b: np.ndarray) -> "tuple[float, float]":
    """Two-sample Kolmogorov-Smirnov statistic and its asymptotic p-value.

    Returns ``(D, p)``. The p-value uses the standard asymptotic series

    .. math:: Q(\\lambda) = 2\\sum_{j=1}^{\\infty} (-1)^{j-1} e^{-2j^2\\lambda^2}

    evaluated at :math:`\\lambda = (\\sqrt{n_e} + 0.12 + 0.11/\\sqrt{n_e})\\,D`. Implemented here rather
    than imported so the core stays numpy-only, and verified in the tests against `scipy.stats.ks_2samp`
    where scipy happens to be available.
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0

    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / n_a
    cdf_b = np.searchsorted(b, grid, side="right") / n_b
    d = float(np.max(np.abs(cdf_a - cdf_b)))

    n_e = n_a * n_b / (n_a + n_b)
    lam = (math.sqrt(n_e) + 0.12 + 0.11 / math.sqrt(n_e)) * d
    if lam <= 0:
        return d, 1.0
    total = 0.0
    for j in range(1, 101):
        term = 2.0 * (-1) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam)
        total += term
        if abs(term) < 1e-12:
            break
    return d, float(min(max(total, 0.0), 1.0))


@dataclass
class KSWIN:
    """Kolmogorov-Smirnov windowing: compare a recent sample against a reservoir of older data.

    Where ADWIN watches the **mean**, KSWIN compares whole **distributions**. That is the reason to carry
    both: a fault that changes the spread or the shape of a channel while leaving its mean alone is
    invisible to ADWIN and to every control chart centred on the mean, and it is exactly the kind of thing
    a developing mechanical fault does before it moves an average.

    Parameters
    ----------
    window
        Total retained samples. The most recent ``recent`` of them are the test sample; the rest are the
        reference reservoir.
    recent
        Size of the recent test sample.
    alpha
        Significance for the KS test. Note this is a **per-step** significance with no multiple-testing
        correction, so the realised false-alarm rate over a long record is far higher than ``alpha``. That
        is inherent to the method rather than a defect here, and it is the reason the reported statistic
        is the KS distance rather than a flag: the calibration layer sets the operating point.
    """

    window: int = 200
    recent: int = 50
    alpha: float = 0.005
    _buffer: deque = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        if self.recent < 5:
            raise ValueError(f"recent must be at least 5, got {self.recent}")
        if self.window <= self.recent:
            raise ValueError(f"window ({self.window}) must exceed recent ({self.recent})")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")

    def reset(self) -> None:
        self._buffer.clear()

    def update(self, value: float) -> "tuple[float, float]":
        """Add one observation, returning ``(D, p)`` for the current comparison."""
        self._buffer.append(float(value))
        while len(self._buffer) > self.window:
            self._buffer.popleft()
        if len(self._buffer) < self.window:
            return float("nan"), float("nan")
        data = np.fromiter(self._buffer, dtype=float, count=len(self._buffer))
        return ks_two_sample_pvalue(data[: -self.recent], data[-self.recent:])

    def detect(self, series: Series) -> Detection:
        """Run per channel; the statistic is the maximum KS distance across channels.

        The KS distance rather than the p-value, because the distance is oriented so larger means more
        anomalous and is bounded in [0, 1], while a p-value would need flipping and saturates at the
        interesting end.
        """
        detectors = [KSWIN(self.window, self.recent, self.alpha) for _ in range(series.d)]
        distances = np.full((series.n, series.d), np.nan)
        pvalues = np.full((series.n, series.d), np.nan)
        for i in range(series.n):
            row = series.x[i]
            for j in range(series.d):
                if np.isfinite(row[j]):
                    distances[i, j], pvalues[i, j] = detectors[j].update(row[j])
        statistic = np.full(series.n, np.nan)
        defined = ~np.all(np.isnan(distances), axis=1)
        if defined.any():
            statistic[defined] = np.nanmax(distances[defined], axis=1)
        return Detection(series.t, statistic, "kswin", {
            "per_channel": distances, "pvalues": pvalues, "names": series.names,
            "alpha": self.alpha, "window": self.window, "recent": self.recent,
        })
