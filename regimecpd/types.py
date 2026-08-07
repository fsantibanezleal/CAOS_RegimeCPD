"""The data contract: what a series, a regime labelling, a residual and a detection are.

These types exist so that every rung of the ladder consumes and produces the same shapes. The whole
point of this package is a head-to-head comparison in which only ONE thing differs between two arms
(raw channels versus within-regime residuals), and that comparison is only trustworthy if both arms
travel through identical structures.

Two decisions here are load-bearing and are not conventions borrowed from elsewhere.

**Time is explicit and may be irregular.** Fleet telemetry is irregular, multi-rate and gappy: a truck
parked for a shift produces no readouts, and two trucks do not share a clock. Carrying ``t`` rather than
assuming a fixed sample interval means windows can be expressed in time rather than in samples, and it
means the false-alarm rate can be reported per unit of TIME, which is the only form a maintenance
planner can act on. A package that indexes by sample silently reports a different false-alarm rate for
a truck that idles.

**The decision statistic is separated from the alarm.** A detector returns a continuous statistic; the
threshold is applied later, by the calibration layer. Detectors that bake in a threshold cannot be put
on a common alarm budget, and comparing methods at their own favourite thresholds is the single easiest
way to produce a meaningless benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Series", "RegimeLabels", "Residual", "Detection", "Attribution"]


def _as_2d(x: np.ndarray) -> np.ndarray:
    """A univariate series is a d=1 multivariate series, not a special case."""
    a = np.asarray(x, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


@dataclass(frozen=True)
class Series:
    """A multivariate time series from one unit (one truck, one engine, one machine).

    Parameters
    ----------
    t
        Sample times, shape ``(n,)``, strictly increasing. Any unit (seconds, hours, cycles) as long as
        one series is internally consistent; every rate this package reports is expressed per unit of
        ``t``, so the caller chooses what "per truck-month" means by choosing the unit of ``t``.
    x
        Channel values, shape ``(n, d)``. A 1-D array is accepted and treated as ``d = 1``.
    names
        Channel names, length ``d``. Names are not decoration: attribution output is only actionable if
        it can say "strut pressure", and the whole product argument rests on being able to.
    unit_id
        Identifier of the unit this series came from. Carried because intervals must be bootstrapped
        over UNITS rather than over samples, and that is impossible once unit identity is lost.
    """

    t: np.ndarray
    x: np.ndarray
    names: tuple[str, ...]
    unit_id: str = "unit"

    def __post_init__(self) -> None:
        t = np.asarray(self.t, dtype=float)
        x = _as_2d(self.x)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "names", tuple(self.names))

        if t.ndim != 1:
            raise ValueError(f"t must be 1-D, got shape {t.shape}")
        if len(t) != len(x):
            raise ValueError(f"t has {len(t)} samples but x has {len(x)}")
        if len(self.names) != x.shape[1]:
            raise ValueError(f"{len(self.names)} names for {x.shape[1]} channels")
        if len(t) > 1 and not np.all(np.diff(t) > 0):
            raise ValueError("t must be strictly increasing; sort the series before constructing it")

    @property
    def n(self) -> int:
        """Number of samples."""
        return len(self.t)

    @property
    def d(self) -> int:
        """Number of channels."""
        return self.x.shape[1]

    @property
    def duration(self) -> float:
        """Elapsed time covered, in the units of ``t``. Zero for a series of fewer than two samples."""
        return float(self.t[-1] - self.t[0]) if self.n > 1 else 0.0

    def channel(self, name: str) -> np.ndarray:
        """The named channel, so callers index by meaning rather than by remembered column order."""
        try:
            return self.x[:, self.names.index(name)]
        except ValueError:
            raise KeyError(f"no channel {name!r}; have {self.names}") from None

    def select(self, names: "tuple[str, ...] | list[str]") -> "Series":
        """A view restricted to the named channels, preserving their given order."""
        idx = [self.names.index(nm) for nm in names]
        return Series(self.t, self.x[:, idx], tuple(names), self.unit_id)


@dataclass(frozen=True)
class RegimeLabels:
    """Which operating regime each sample belongs to.

    ``labels[i] == -1`` marks a sample that could not be assigned, which happens when a sample falls
    outside every regime seen during fitting. That is deliberately NOT folded into a nearest-regime
    guess: a truck operating in a context the baseline never saw is a case where the residual is not
    trustworthy, and hiding it inside the closest cluster is how a regime-conditional method quietly
    becomes a regime-conditional method with an unmodelled failure case.
    """

    labels: np.ndarray
    n_regimes: int
    method: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", np.asarray(self.labels, dtype=int))
        if self.labels.ndim != 1:
            raise ValueError(f"labels must be 1-D, got shape {self.labels.shape}")
        if self.n_regimes < 1:
            raise ValueError(f"n_regimes must be at least 1, got {self.n_regimes}")

    @property
    def unassigned(self) -> np.ndarray:
        """Boolean mask of samples no regime claimed."""
        return self.labels < 0

    @property
    def coverage(self) -> float:
        """Fraction of samples that were assigned a regime. A low value invalidates the residual."""
        return float(np.mean(~self.unassigned)) if len(self.labels) else 0.0


@dataclass(frozen=True)
class Residual:
    """What is left of a series after its operating regime is accounted for.

    This is the object the central comparison turns on. Arm A of the comparison runs a detector on a
    ``Series``; arm B runs the identical detector on a ``Residual``. Keeping the regime labels attached
    means a result can always be traced back to the segmentation that produced it.
    """

    t: np.ndarray
    r: np.ndarray
    names: tuple[str, ...]
    regimes: RegimeLabels
    unit_id: str = "unit"

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", np.asarray(self.t, dtype=float))
        object.__setattr__(self, "r", _as_2d(self.r))
        object.__setattr__(self, "names", tuple(self.names))
        if len(self.t) != len(self.r):
            raise ValueError(f"t has {len(self.t)} samples but r has {len(self.r)}")
        if len(self.names) != self.r.shape[1]:
            raise ValueError(f"{len(self.names)} names for {self.r.shape[1]} channels")

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def d(self) -> int:
        return self.r.shape[1]

    def as_series(self) -> Series:
        """Present the residual as a plain ``Series``, so a detector cannot tell the two arms apart.

        This is what makes the head-to-head fair rather than merely well-intentioned: the detector
        receives the same type in both arms and has no way to behave differently.
        """
        return Series(self.t, self.r, self.names, self.unit_id)


@dataclass(frozen=True)
class Detection:
    """A detector's output: a continuous statistic, and nothing thresholded.

    ``statistic`` is oriented so that LARGER means MORE anomalous, for every method in this package.
    Some underlying statistics are naturally the other way round (a likelihood, a run-length mode); those
    are converted at the source rather than at the call site, because a single flipped sign in a
    benchmark harness produces a method that appears to perform exactly as badly as it should perform
    well, and that failure is very hard to see in a results table.
    """

    t: np.ndarray
    statistic: np.ndarray
    method: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", np.asarray(self.t, dtype=float))
        object.__setattr__(self, "statistic", np.asarray(self.statistic, dtype=float))
        if len(self.t) != len(self.statistic):
            raise ValueError(f"t has {len(self.t)} samples but statistic has {len(self.statistic)}")

    def alarms(self, threshold: float) -> np.ndarray:
        """Boolean mask of samples at or above ``threshold``.

        NaN is treated as "not alarming" rather than propagated. A detector needs a warm-up before its
        statistic is defined, and warm-up samples must not be able to raise an alarm.
        """
        s = self.statistic
        return np.where(np.isnan(s), False, s >= threshold)


@dataclass(frozen=True)
class Attribution:
    """Which channels drove an excursion, and by how much.

    ``scores`` are non-negative and comparable within one attribution; they are not probabilities and do
    not sum to one.

    A caveat travels with this type because it is the one most likely to be over-read. Contribution-style
    attribution SMEARS: a single faulty channel spreads score onto channels correlated with it, so a
    high score means "this channel is implicated", never "this channel is the cause". See
    Westerhuis, Gurden and Smilde (2000), doi:10.1016/S0169-7439(00)00062-9. The honest use is to narrow
    a candidate set, and the honest cross-check is to compare against an attribution obtained by a
    completely different route (this package provides one, in ``mstamp``).
    """

    scores: np.ndarray
    names: tuple[str, ...]
    method: str
    index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", np.asarray(self.scores, dtype=float))
        object.__setattr__(self, "names", tuple(self.names))
        if len(self.scores) != len(self.names):
            raise ValueError(f"{len(self.scores)} scores for {len(self.names)} names")

    def ranked(self) -> list[tuple[str, float]]:
        """Channels ordered by score, most implicated first."""
        order = np.argsort(-self.scores)
        return [(self.names[i], float(self.scores[i])) for i in order]

    def top(self, k: int = 3) -> tuple[str, ...]:
        """The ``k`` most implicated channel names."""
        return tuple(nm for nm, _ in self.ranked()[:k])
