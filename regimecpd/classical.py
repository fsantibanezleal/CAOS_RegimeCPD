"""Classical control-chart detectors: Shewhart, CUSUM, EWMA, Page-Hinkley.

**These are the baseline this package claims to beat, so they are implemented properly.** A strawman
baseline invalidates every comparison built on top of it, and the temptation to leave the incumbent
slightly broken is strong precisely because nobody checks a number that came out the way it was expected
to. Each detector here follows its primary source, is tuned on the baseline rather than on the record
being scored, and gets the same alarm budget as every method it is compared against.

There is a second reason to take them seriously: on a real fleet, a fixed threshold is what is actually
running. A planner comparing this package against their current system is comparing it against these.

.. rubric:: Shared design

Every detector follows the same three-step contract, and the steps are separate calls on purpose:

1. ``fit(baseline)`` establishes per-channel location and scale from a HEALTHY record.
2. ``detect(series)`` returns a :class:`~regimecpd.types.Detection` carrying a continuous statistic.
3. The threshold is applied later, by the calibration layer, against a stated false-alarm budget.

Statistics are oriented so larger is always more anomalous, and they are **not** thresholded internally.
A detector that returns a boolean cannot be placed on a common alarm budget, and comparing methods at
their own favourite thresholds is the easiest way to produce a benchmark that means nothing.

.. rubric:: Multivariate reduction

Each detector runs per channel and the reported statistic is the **maximum across channels**: the machine
is out of control if any channel is. The per-channel statistics are kept in ``Detection.meta`` so
:meth:`~regimecpd.types.Detection.channel_attribution` can say which channel drove a given excursion.

Taking the maximum is deliberate rather than convenient. Averaging across channels dilutes a single-channel
fault by the number of healthy channels, so a twelve-channel machine would need a fault twelve times
larger to reach the same statistic. That is the wrong behaviour for onset detection, where the earliest
evidence is usually one channel moving.

.. rubric:: NaN

A residual carries NaN where the sample could not be residualised, and every recursive detector here must
survive that without corrupting its state. The rule is uniform: a NaN input leaves the recursion untouched
and emits NaN for that sample. Treating NaN as zero would feed a fabricated in-control observation into
the accumulator, which is exactly the sort of quiet fill this package refuses elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scaling import robust_scale
from .types import Detection, Series

__all__ = ["Shewhart", "CUSUM", "EWMA", "PageHinkley"]


@dataclass
class _BaselineScaler:
    """Per-channel location and scale, estimated once on a healthy baseline."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    names_: tuple = ()

    def _fit(self, baseline: Series) -> None:
        x = baseline.x
        finite = np.isfinite(x)
        if not finite.any():
            raise ValueError("the baseline contains no finite samples")
        # nan-aware, because a baseline may itself be a residual with unassignable stretches.
        self.mean_ = np.nanmean(np.where(finite, x, np.nan), axis=0)
        spread = np.nanstd(np.where(finite, x, np.nan), axis=0)
        # A channel that does not meaningfully move in the baseline cannot be standardised. Scale 1
        # leaves it in raw units rather than amplifying floating-point noise into an enormous
        # z-score that would dominate the across-channel maximum. See regimecpd.scaling.
        self.scale_ = robust_scale(spread, self.mean_)
        self.names_ = baseline.names

    def _z(self, series: Series) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("detect called before fit")
        if series.names != self.names_:
            raise ValueError(f"fitted on channels {self.names_}, asked to detect on {series.names}")
        return (series.x - self.mean_) / self.scale_


def _finalise(series: Series, per_channel: np.ndarray, method: str, meta: dict) -> Detection:
    """Reduce per-channel statistics to one, keeping the per-channel detail for attribution."""
    statistic = np.full(len(per_channel), np.nan)
    # Rows where every channel is undefined are left NaN without calling nanmax on them at all. Calling
    # it and suppressing the warning would work, but a suppressed warning is a habit that eventually
    # hides a real one.
    defined = ~np.all(np.isnan(per_channel), axis=1)
    if defined.any():
        statistic[defined] = np.nanmax(per_channel[defined], axis=1)
    return Detection(series.t, statistic, method,
                     {**meta, "per_channel": per_channel, "names": series.names})


@dataclass
class Shewhart(_BaselineScaler):
    """Fixed threshold on the standardised deviation: the incumbent, and the honest reference line.

    The statistic is :math:`\\max_j |x_{t,j} - \\mu_j| / \\sigma_j`, so a threshold of 3 reproduces the
    familiar 3-sigma chart exactly and any other threshold is a point on the same curve.

    Shewhart has **no memory**, which is its real weakness for this problem rather than a technicality.
    Degradation onset is a small persistent shift, and a chart that looks at one sample at a time needs
    that shift to be large before any single sample crosses. CUSUM and EWMA exist because of this.

    The second weakness matters more here and is the reason this package exists: on a load-varying
    machine the in-control distribution is a MIXTURE over operating regimes, so a single
    :math:`\\mu, \\sigma` pair fits none of them. Whether that actually costs false alarms is measured,
    not assumed.

    Source: Shewhart, W. A. *Economic Control of Quality of Manufactured Product* (1931).
    """

    def fit(self, baseline: Series) -> "Shewhart":
        self._fit(baseline)
        return self

    def detect(self, series: Series) -> Detection:
        return _finalise(series, np.abs(self._z(series)), "shewhart", {})


@dataclass
class CUSUM(_BaselineScaler):
    """Two-sided cumulative sum: the workhorse for the small persistent shift onset actually looks like.

    Per channel, on standardised observations :math:`z_t`:

    .. math::
        S^+_t = \\max(0,\\; S^+_{t-1} + z_t - k), \\qquad
        S^-_t = \\max(0,\\; S^-_{t-1} - z_t - k)

    and the channel statistic is :math:`\\max(S^+_t, S^-_t)`.

    Parameters
    ----------
    k
        Reference value in standard deviations, conventionally **half** the shift you want to detect
        quickly. The default 0.5 targets a one-sigma shift. It is not a free knob: too small and the
        accumulator drifts upward on noise alone, too large and it never accumulates.

    The reset at zero is what makes this work. Without it the sum is a random walk that wanders away from
    the origin and eventually crosses any threshold on noise alone, so a plain cumulative sum is not a
    detector at all.

    Source: Page, E. S. "Continuous Inspection Schemes." *Biometrika* **41**(1-2), 100-115 (1954).
    doi:10.1093/biomet/41.1-2.100
    """

    k: float = 0.5

    def fit(self, baseline: Series) -> "CUSUM":
        self._fit(baseline)
        return self

    def detect(self, series: Series) -> Detection:
        z = self._z(series)
        n, d = z.shape
        out = np.full((n, d), np.nan)
        pos = np.zeros(d)
        neg = np.zeros(d)
        for i in range(n):
            row = z[i]
            ok = np.isfinite(row)
            if not ok.any():
                continue  # state carried, statistic undefined here
            pos = np.where(ok, np.maximum(0.0, pos + row - self.k), pos)
            neg = np.where(ok, np.maximum(0.0, neg - row - self.k), neg)
            out[i] = np.where(ok, np.maximum(pos, neg), np.nan)
        return _finalise(series, out, "cusum", {"k": self.k})


@dataclass
class EWMA(_BaselineScaler):
    """Exponentially weighted moving average, with the exact time-varying control limit.

    Per channel, on standardised observations:

    .. math::
        w_t = \\lambda z_t + (1 - \\lambda) w_{t-1}, \\qquad
        \\sigma_{w_t}^2 = \\frac{\\lambda}{2 - \\lambda}\\left[1 - (1-\\lambda)^{2t}\\right]

    and the channel statistic is :math:`|w_t| / \\sigma_{w_t}`.

    The time-varying variance is not a refinement to skip. The asymptotic form
    :math:`\\lambda / (2-\\lambda)` **overstates** the spread of the early samples, so a chart using it
    from the start is too insensitive for the first few dozen observations. On a fleet where records are
    short and restarts are common, that dead zone at the beginning of every record is a real loss of
    detection, and it is invisible unless you look for it.

    Parameters
    ----------
    lam
        Smoothing constant in (0, 1]. Small values give long memory and catch small shifts slowly; 1
        reduces exactly to Shewhart. This is the direct detection-delay against false-alarm-rate trade,
        which is why it belongs on the alarm-budget curve rather than at one chosen operating point.

    Source: Roberts, S. W. "Control Chart Tests Based on Geometric Moving Averages." *Technometrics*
    **1**(3), 239-250 (1959). doi:10.1080/00401706.1959.10489860. Originally the geometric moving average
    chart; the EWMA name came later.
    """

    lam: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 < self.lam <= 1.0:
            raise ValueError(f"lam must be in (0, 1], got {self.lam}")

    def fit(self, baseline: Series) -> "EWMA":
        self._fit(baseline)
        return self

    def detect(self, series: Series) -> Detection:
        z = self._z(series)
        n, d = z.shape
        out = np.full((n, d), np.nan)
        w = np.zeros(d)
        # Counted per channel, because a channel that was NaN for a stretch has genuinely seen fewer
        # observations and its control limit must reflect that rather than the wall-clock index.
        seen = np.zeros(d, dtype=int)
        asym = self.lam / (2.0 - self.lam)
        for i in range(n):
            row = z[i]
            ok = np.isfinite(row)
            if not ok.any():
                continue
            w = np.where(ok, self.lam * row + (1.0 - self.lam) * w, w)
            seen = seen + ok
            var = asym * (1.0 - (1.0 - self.lam) ** (2 * np.maximum(seen, 1)))
            out[i] = np.where(ok & (seen > 0), np.abs(w) / np.sqrt(var), np.nan)
        return _finalise(series, out, "ewma", {"lam": self.lam})


@dataclass
class PageHinkley(_BaselineScaler):
    """Page-Hinkley: a sequential test for a change in mean, cheap enough for the streaming lane.

    Per channel, tracking the cumulative deviation from an allowed drift :math:`\\delta` and its running
    extreme:

    .. math::
        m^+_t = \\sum_{i \\le t} (z_i - \\delta), \\quad M^+_t = \\min_{i \\le t} m^+_i, \\quad
        \\mathrm{PH}^+_t = m^+_t - M^+_t

    with the mirrored form for a downward change, and the channel statistic the larger of the two.

    Parameters
    ----------
    delta
        Tolerated drift in standard deviations. Deviations smaller than this do not accumulate, which is
        what keeps the statistic from climbing on noise.

    **This is a CUSUM variant, not an independent rung**, and it is labelled that way here rather than
    presented as a fourth opinion. Page-Hinkley and CUSUM come from the same 1954 construction and agree
    closely on most data. Counting them as two independent classical methods would inflate any claim of
    the form "we beat N baselines".

    Source: Hinkley, D. V. (1971), building directly on Page (1954). The exact Hinkley reference was
    **not verified against a primary source** in the research pass behind this package, so it is recorded
    here as UNVERIFIED and the implementation follows the standard formulation given above rather than a
    specific paper's notation.
    """

    delta: float = 0.005

    def fit(self, baseline: Series) -> "PageHinkley":
        self._fit(baseline)
        return self

    def detect(self, series: Series) -> Detection:
        z = self._z(series)
        n, d = z.shape
        out = np.full((n, d), np.nan)
        m_up = np.zeros(d)
        min_up = np.zeros(d)
        m_dn = np.zeros(d)
        min_dn = np.zeros(d)
        for i in range(n):
            row = z[i]
            ok = np.isfinite(row)
            if not ok.any():
                continue
            m_up = np.where(ok, m_up + row - self.delta, m_up)
            min_up = np.where(ok, np.minimum(min_up, m_up), min_up)
            m_dn = np.where(ok, m_dn - row - self.delta, m_dn)
            min_dn = np.where(ok, np.minimum(min_dn, m_dn), min_dn)
            out[i] = np.where(ok, np.maximum(m_up - min_up, m_dn - min_dn), np.nan)
        return _finalise(series, out, "page-hinkley", {"delta": self.delta})
