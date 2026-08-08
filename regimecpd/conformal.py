"""Conformal calibration: turning any detector's score into a false-alarm rate you can budget.

This is the rung that makes the rest of the package comparable. Every detector here returns a continuous
statistic on its own arbitrary scale: a CUSUM in accumulated sigma, an SPE in squared residual units, a
matrix profile in normalised distance. Those numbers cannot be compared, and a threshold chosen on one
means nothing on another.

Split conformal fixes that by converting a score into a **p-value against a healthy calibration set**:

.. math:: p_t = \\frac{1 + \\left|\\{ i : s_i \\ge s_t \\}\\right|}{n_{\\mathrm{cal}} + 1}

where :math:`s_i` are the scores of :math:`n_{\\mathrm{cal}}` held-out healthy samples. Alarm when
:math:`p_t \\le \\alpha`.

.. rubric:: The guarantee, stated precisely enough to be useful

Under **exchangeability** of the calibration and test scores, this p-value is (super-)uniform, so

.. math:: \\Pr(p_t \\le \\alpha) \\le \\alpha

for any finite calibration set. Distribution-free: no normality, no asymptotics, no assumption about the
detector. The ``1 +`` in the numerator is what makes the bound hold at finite :math:`n` rather than only
asymptotically, and dropping it (the natural-looking simplification) breaks the guarantee in exactly the
direction that produces more alarms than promised.

.. rubric:: The assumption is FALSE on the data this package targets, and that is the point

Time series are not exchangeable. Consecutive samples from one truck are strongly dependent, the machine
drifts, and the operating context moves. So the marginal guarantee above is an idealisation, and treating
it as a delivered false-alarm rate would be exactly the kind of borrowed authority this package exists to
avoid.

Three things follow, and all three are implemented rather than discussed:

1. :class:`SplitConformal` reports the **realised** rate on held-out healthy data next to the nominal
   one, so the gap is visible rather than assumed away.
2. :class:`AdaptiveConformal` (Gibbs and Candes) updates :math:`\\alpha_t` online from its own recent
   error rate, which recovers the long-run rate **irrespective of the true data-generating process**.
   That is a genuinely different guarantee: not marginal coverage under exchangeability, but long-run
   empirical coverage under nothing at all.
3. Calibration is done **per unit** where possible, since exchangeability across trucks is more nearly
   true than exchangeability across time within one truck.

.. rubric:: Why this is what makes the central comparison honest

The package's claim is that a detector on regime-conditioned residuals beats the same detector on raw
channels. Compared at each arm's own favourite threshold, that claim is unfalsifiable. Compared at a
**fixed conformal false-alarm budget**, it is a measurement. Every benchmark in this package therefore
reads both arms off a common budget, and this module is what makes a common budget mean the same thing on
two different score scales.

Sources:

- Vovk, V., Gammerman, A., Shafer, G. *Algorithmic Learning in a Random World* (2005), for split
  conformal prediction. **UNVERIFIED:** not opened against a primary source during the research pass
  behind this package.
- Gibbs, I., Candes, E. J. "Adaptive Conformal Inference Under Distribution Shift."
  *Advances in Neural Information Processing Systems* **34**, 1660-1672 (2021).
- Xu, C., Xie, Y. "Conformal Prediction Interval for Dynamic Time-Series." *ICML 2021*,
  PMLR **139**, 11559-11569, for the ensemble route to conformal inference without exchangeability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import UnitOutcome, score_fleet
from .types import Detection

__all__ = ["SplitConformal", "AdaptiveConformal", "conformalise", "calibration_report"]


@dataclass
class SplitConformal:
    """Split conformal p-values from a healthy calibration set.

    Parameters
    ----------
    scores_
        Calibration scores, sorted ascending at fit time so each p-value is a binary search.
    """

    scores_: np.ndarray | None = None

    def fit(self, calibration: "np.ndarray | list[Detection]") -> "SplitConformal":
        """Fit on scores from HEALTHY data that the detector did not see during its own training.

        Passing the detector's training data here is the classic mistake: the scores would be optimistic,
        the p-values inflated, and the realised false-alarm rate far above nominal. The calibration set
        must be held out from the detector fit as well as from the test record.
        """
        if isinstance(calibration, (list, tuple)):
            scores = np.concatenate([np.asarray(d.statistic, dtype=float) for d in calibration])
        else:
            scores = np.asarray(calibration, dtype=float)
        scores = scores[np.isfinite(scores)]
        if scores.size < 20:
            raise ValueError(
                f"only {scores.size} finite calibration scores; the smallest p-value this can produce "
                f"is 1/{scores.size + 1}, so a budget below that is unreachable by construction"
            )
        self.scores_ = np.sort(scores)
        return self

    @property
    def resolution(self) -> float:
        """The smallest achievable p-value, ``1 / (n_cal + 1)``.

        Worth checking before setting a budget. Asking for a false-alarm rate below this is not strict,
        it is impossible: no observation, however extreme, can produce a smaller p-value.
        """
        if self.scores_ is None:
            raise RuntimeError("resolution requested before fit")
        return 1.0 / (len(self.scores_) + 1)

    def p_values(self, statistic: np.ndarray) -> np.ndarray:
        """Conformal p-values for an array of scores. NaN in, NaN out."""
        if self.scores_ is None:
            raise RuntimeError("p_values called before fit")
        s = np.asarray(statistic, dtype=float)
        n = len(self.scores_)
        # Count calibration scores >= s. searchsorted on the ascending array with side="left" gives the
        # number strictly below, so the complement is the number at or above.
        at_or_above = n - np.searchsorted(self.scores_, s, side="left")
        # The 1 + in numerator and denominator is what makes this valid at finite n. Dropping it looks
        # like a harmless simplification and breaks the bound in the direction of MORE alarms than
        # promised, which is the direction nobody checks.
        p = (1.0 + at_or_above) / (n + 1.0)
        return np.where(np.isnan(s), np.nan, p)

    def conform(self, detection: Detection) -> Detection:
        """Rewrite a detection so its statistic is ``-log10(p)``: higher is still more anomalous.

        The negative log rather than the p-value itself, for two reasons. The orientation convention in
        this package is that larger means more anomalous, and a p-value runs the other way. And a p-value
        floors at ``1/(n_cal+1)``, so a raw-p statistic saturates at exactly the sensitive end where an
        alarm-budget curve needs its resolution.
        """
        p = self.p_values(detection.statistic)
        with np.errstate(divide="ignore"):
            stat = -np.log10(np.where(np.isnan(p), np.nan, np.maximum(p, 1e-300)))
        return Detection(detection.t, stat, f"conformal({detection.method})", {
            **detection.meta, "p_value": p, "n_calibration": len(self.scores_),
            "resolution": self.resolution, "source_method": detection.method,
        })


@dataclass
class AdaptiveConformal:
    """Gibbs and Candes adaptive conformal inference: correct the level from realised errors.

    Split conformal assumes exchangeability. Time series violate it, so the realised rate drifts away
    from nominal. This wrapper tracks its own error rate online and adjusts the working level:

    .. math::
        \\alpha_{t+1} = \\alpha_t
        + \\gamma\\left(\\alpha_{\\mathrm{target}} - \\mathbb{1}[p_t \\le \\alpha_t]\\right)

    When alarms are coming too often the level tightens; when they dry up it loosens. The result
    converges to the target long-run rate **irrespective of the true data-generating process**, which is a
    different and stronger claim than marginal coverage under an assumption known to be false.

    Parameters
    ----------
    gamma
        Adaptation rate. Larger tracks a shifting distribution faster and wanders more around the target;
        smaller is steadier and slower to react. It is the usual bias-variance trade and there is no
        universally right value.

    Notes
    -----
    The guarantee is on the **long-run average** rate over a record, not on any particular window. A
    detector using this can still emit a burst of alarms and then stay silent for a stretch while the
    level recovers; the average comes out right and the experience is lumpy. That is worth knowing before
    it is presented to an operator as a steady budget.
    """

    target: float = 0.01
    gamma: float = 0.01
    base: SplitConformal = field(default_factory=SplitConformal)
    alpha_: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.target < 1.0:
            raise ValueError(f"target must be in (0, 1), got {self.target}")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        self.alpha_ = self.target

    def fit(self, calibration: "np.ndarray | list[Detection]") -> "AdaptiveConformal":
        self.base.fit(calibration)
        self.alpha_ = self.target
        return self

    def run(self, detection: Detection) -> "tuple[np.ndarray, np.ndarray]":
        """Stream a detection, returning ``(alarms, alpha_trace)``.

        ``alpha_trace`` is the working level at each step, which is the diagnostic worth plotting. A
        level that has gone deeply NEGATIVE means the detector fired far more often than the target and
        is working off the debt; that is the mechanism operating correctly, not a failure, and it is why
        the trace is returned rather than hidden.
        """
        p = self.base.p_values(detection.statistic)
        alarms = np.zeros(len(p), dtype=bool)
        trace = np.empty(len(p))
        alpha = self.target
        for i, pi in enumerate(p):
            trace[i] = alpha
            if np.isnan(pi):
                continue          # an undefined statistic is neither an error nor an alarm
            fired = bool(pi <= alpha)
            alarms[i] = fired
            # NOT clipped to [0, 1], and that is load-bearing rather than an oversight.
            #
            # A negative level simply means "do not alarm", since no p-value satisfies p <= alpha < 0.
            # The magnitude of the negative excursion is the DEBT the recursion has to work off before it
            # will fire again, and that debt is precisely what enforces the long-run rate.
            #
            # Clipping at 0 was the first implementation and it breaks the guarantee in the regime where
            # it matters most. With a saturated detector (every score above every calibration score, so
            # every p-value at the resolution floor), one alarm should cost gamma*(1 - target) of level
            # and take about 1/target quiet steps to repay. Clipped, the debt is thrown away at zero, the
            # level climbs back within two steps, and the realised rate comes out near 50% against a
            # target of 1%. Measured, and there is a regression test.
            alpha = alpha + self.gamma * (self.target - float(fired))
        self.alpha_ = alpha
        return alarms, trace


def conformalise(detections: "list[Detection]", calibration: "np.ndarray | list[Detection]"
                 ) -> "list[Detection]":
    """Put a whole fleet of detections on one conformal scale, from one calibration set."""
    model = SplitConformal().fit(calibration)
    return [model.conform(d) for d in detections]


def calibration_report(model: SplitConformal, held_out: "list[Detection]",
                       alphas: "tuple[float, ...]" = (0.05, 0.01, 0.005, 0.001)) -> dict:
    """Nominal against REALISED alarm rate on held-out healthy data.

    The single most useful output in this module. The conformal guarantee holds under exchangeability;
    fleet telemetry is not exchangeable, so the realised rate will not equal the nominal one and the size
    of the gap is an empirical question about the data rather than a theoretical one about the method.

    Reported both **per sample** and **per event** (contiguous excursions), because those differ by the
    dwell time of the statistic and the per-event number is the one an operator experiences.
    """
    p = np.concatenate([model.p_values(d.statistic) for d in held_out])
    p = p[np.isfinite(p)]
    outcomes = [UnitOutcome(model.conform(d), onset_t=None, unit_id=f"cal{i}")
                for i, d in enumerate(held_out)]

    rows = []
    for alpha in alphas:
        per_sample = float(np.mean(p <= alpha)) if p.size else float("nan")
        # The same alpha as a threshold on -log10(p), so the event count uses the package's own
        # rising-edge convention rather than a second, quietly different one.
        fleet = score_fleet(outcomes, -np.log10(alpha))
        rows.append({
            "alpha": alpha,
            "realised_per_sample": per_sample,
            "ratio": per_sample / alpha if alpha > 0 else float("nan"),
            "events_per_unit_time": fleet.false_alarms_per_unit_time,
            "n_events": fleet.n_false_alarms,
        })
    return {"resolution": model.resolution, "n_calibration": len(model.scores_), "rows": rows}
