"""How a detector is scored: detection delay, false alarms per unit time, and the alarm budget.

Accuracy is the wrong metric for this problem and it is worth being explicit about why. Onset events are
rare, so a detector that never fires scores extremely well on accuracy while being worthless. What a
maintenance planner actually trades is: how many nuisance call-outs per truck per month am I willing to
absorb, and given that budget, how early do I hear about a real problem. Those two quantities are what
this module computes.

Three choices here are load-bearing.

**Alarms are counted as EVENTS, not as samples above a line.** A statistic that sits above the threshold
for two hundred samples is one alarm that a human acknowledges once, not two hundred alarms. Counting
rising edges is what makes "false alarms per truck-month" mean the thing an operator experiences. Counting
samples instead inflates every rate by the dwell time of the statistic, and it inflates a smooth method
more than a spiky one, which quietly rigs the comparison.

**An excursion that starts before the onset is a false alarm, even if it is still going when the onset
arrives.** Detection is the first rising edge at or after the onset. A detector that is alarming
continuously from the start of the record has not detected anything; it has one false alarm and a miss.
The alternative convention (any alarm covering the onset counts as a detection) rewards a detector for
being permanently on, which is the degenerate solution this whole package exists to avoid.

**Intervals are bootstrapped over UNITS, never over samples.** Samples within one truck are strongly
dependent, so resampling them produces intervals that are far too narrow and a benchmark that looks much
more certain than it is. Resampling whole units is the honest unit of independence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Detection

__all__ = [
    "UnitOutcome",
    "UnitScore",
    "FleetScore",
    "rising_edges",
    "score_unit",
    "score_fleet",
    "candidate_thresholds",
    "alarm_budget_curve",
    "threshold_for_budget",
    "bootstrap_ci",
]


def rising_edges(mask: np.ndarray) -> np.ndarray:
    """Indices where ``mask`` transitions from not-alarming to alarming.

    A mask that is already True at index 0 counts as an edge there: the record opens mid-excursion, and
    that is an alarm the operator sees.
    """
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return np.empty(0, dtype=int)
    prev = np.concatenate(([False], m[:-1]))
    return np.flatnonzero(m & ~prev)


@dataclass(frozen=True)
class UnitOutcome:
    """One unit's detector output paired with the ground truth it is scored against.

    ``onset_t is None`` means the unit is healthy for its whole record, so every alarm on it is false and
    its entire duration counts as healthy exposure. Healthy units are not filler: they are where the
    false-alarm rate is actually measured, and a benchmark run only on faulty units cannot report one.
    """

    detection: Detection
    onset_t: float | None = None
    unit_id: str = "unit"

    @property
    def healthy_exposure(self) -> float:
        """Time during which any alarm would be false, in the units of ``t``."""
        t = self.detection.t
        if t.size < 2:
            return 0.0
        if self.onset_t is None:
            return float(t[-1] - t[0])
        return float(max(0.0, min(self.onset_t, t[-1]) - t[0]))

    @property
    def is_faulty(self) -> bool:
        """True when this unit has a true onset inside its record."""
        return self.onset_t is not None


@dataclass(frozen=True)
class UnitScore:
    """The outcome of scoring one unit at one threshold."""

    unit_id: str
    n_false_alarms: int
    healthy_exposure: float
    detected: bool
    delay: float | None
    is_faulty: bool


@dataclass(frozen=True)
class FleetScore:
    """Fleet-level performance at one threshold. The row of an alarm-budget curve."""

    threshold: float
    false_alarms_per_unit_time: float
    n_false_alarms: int
    healthy_exposure: float
    detection_rate: float
    n_detected: int
    n_faulty: int
    median_delay: float | None
    mean_delay: float | None

    def per(self, time_units: float) -> float:
        """The false-alarm rate expressed per ``time_units`` of ``t``.

        For hourly samples, ``per(730.0)`` gives false alarms per unit-month. The conversion lives here so
        a caller never has to remember which direction to multiply.
        """
        return self.false_alarms_per_unit_time * time_units


def score_unit(outcome: UnitOutcome, threshold: float) -> UnitScore:
    """Score one unit at one threshold, using the event and ordering conventions in the module docstring."""
    det = outcome.detection
    edges = rising_edges(det.alarms(threshold))
    edge_times = det.t[edges] if edges.size else np.empty(0)

    if outcome.onset_t is None:
        return UnitScore(outcome.unit_id, int(edge_times.size), outcome.healthy_exposure,
                         False, None, is_faulty=False)

    before = edge_times < outcome.onset_t
    after = edge_times[~before]
    delay = float(after[0] - outcome.onset_t) if after.size else None
    return UnitScore(outcome.unit_id, int(np.count_nonzero(before)), outcome.healthy_exposure,
                     detected=after.size > 0, delay=delay, is_faulty=True)


def score_fleet(outcomes: "list[UnitOutcome]", threshold: float) -> FleetScore:
    """Pool per-unit scores into the two numbers a planner decides on.

    The false-alarm rate is pooled (total events over total healthy exposure) rather than averaged over
    units, so a unit with a short record cannot carry the same weight as one observed for a year.
    """
    scores = [score_unit(o, threshold) for o in outcomes]
    n_fa = sum(s.n_false_alarms for s in scores)
    exposure = sum(s.healthy_exposure for s in scores)
    faulty = [s for s in scores if s.is_faulty]
    detected = [s for s in faulty if s.detected]
    delays = np.array([s.delay for s in detected], dtype=float) if detected else np.empty(0)

    return FleetScore(
        threshold=float(threshold),
        false_alarms_per_unit_time=(n_fa / exposure) if exposure > 0 else float("nan"),
        n_false_alarms=n_fa,
        healthy_exposure=exposure,
        detection_rate=(len(detected) / len(faulty)) if faulty else float("nan"),
        n_detected=len(detected),
        n_faulty=len(faulty),
        median_delay=float(np.median(delays)) if delays.size else None,
        mean_delay=float(np.mean(delays)) if delays.size else None,
    )


def candidate_thresholds(outcomes: "list[UnitOutcome]", n: int = 200) -> np.ndarray:
    """A grid of thresholds spanning the pooled statistics, ascending.

    Quantiles of the pooled values rather than a linear span, because detection statistics are heavily
    right-skewed and a linear grid spends nearly all of its points in a region where nothing happens.
    """
    pooled = np.concatenate([o.detection.statistic for o in outcomes]) if outcomes else np.empty(0)
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return np.empty(0)
    qs = np.linspace(0.0, 1.0, max(2, n))
    grid = np.unique(np.quantile(pooled, qs))
    # One step above the maximum, so the "never fires" end of the curve is representable.
    return np.append(grid, np.nextafter(grid[-1], np.inf))


def alarm_budget_curve(outcomes: "list[UnitOutcome]",
                       thresholds: "np.ndarray | None" = None,
                       n: int = 200) -> "list[FleetScore]":
    """Detection performance across the whole range of thresholds.

    This is the comparison instrument. Two methods reported at their own preferred operating points say
    nothing; the same two methods read off at a FIXED false-alarm rate say everything.
    """
    th = candidate_thresholds(outcomes, n) if thresholds is None else np.asarray(thresholds, dtype=float)
    return [score_fleet(outcomes, float(t)) for t in th]


def threshold_for_budget(outcomes: "list[UnitOutcome]", target_rate: float,
                         n: int = 400) -> "float | None":
    """The most sensitive threshold whose false-alarm rate still fits inside ``target_rate``.

    **The event-counted false-alarm rate is NOT monotone in the threshold**, and this function exists in
    the shape it does because of that. Counting excursions rather than samples makes the rate unimodal:

    * at a very HIGH threshold nothing ever crosses, so the rate is zero;
    * in the MIDDLE the statistic crosses in and out repeatedly, so the rate peaks;
    * at a very LOW threshold the statistic sits above the line for the entire record, which is ONE
      excursion, so the rate collapses back towards zero.

    That last region is a trap. A detector pinned permanently above its threshold reports a superb
    false-alarm rate and detects nothing at all, because under the ordering convention in this module its
    single excursion begins before every onset and is therefore false. A naive scan upward from the
    bottom of the grid returns exactly that point, and the resulting benchmark row reads as a
    spectacularly cheap detector.

    So the search descends from the "never fires" end and stops at the first threshold that breaks the
    budget, returning the lowest threshold in the region CONNECTED to never-firing. That region is the
    one where lowering the bar genuinely buys sensitivity.

    Returns ``None`` when even the top of the grid fails the budget, which is a real outcome rather than
    an error: a detector can be unable to operate at a given budget at all, and quietly returning the
    maximum threshold would disguise "cannot operate" as "operates and detects nothing".
    """
    chosen: float | None = None
    for score in reversed(alarm_budget_curve(outcomes, n=n)):
        rate = score.false_alarms_per_unit_time
        if not np.isfinite(rate) or rate > target_rate:
            break
        chosen = score.threshold
    return chosen


def bootstrap_ci(outcomes: "list[UnitOutcome]",
                 statistic: "callable",
                 n_boot: int = 1000,
                 alpha: float = 0.05,
                 seed: int = 0) -> "tuple[float, float, float]":
    """Point estimate and a percentile interval, resampling UNITS with replacement.

    Returns ``(point, lo, hi)``. Resampling units rather than samples is the whole reason this function
    exists: samples inside one unit are dependent, and a sample bootstrap would report an interval
    perhaps an order of magnitude too narrow.

    ``statistic`` maps a list of outcomes to a float. Replicates that come back non-finite are dropped
    rather than propagated, because a resample can legitimately contain no faulty unit at all and so have
    no defined detection rate; the count that survived is not hidden, it is recoverable from the interval
    being wide.
    """
    point = float(statistic(outcomes))
    if not outcomes:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(outcomes))
    reps = []
    for _ in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        value = statistic([outcomes[i] for i in pick])
        if np.isfinite(value):
            reps.append(float(value))
    if not reps:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(reps, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi)
