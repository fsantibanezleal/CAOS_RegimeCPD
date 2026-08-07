# Changelog

All notable changes to this project are documented here, newest first, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions use the `X.XX.XXX` display form; the
`pyproject.toml` manifest carries the same version in PEP 440 form with the padding dropped.

## [0.02.000] - 2026-08-07

Regime conditioning: the stage every other rung sits downstream of, and the one the package's central
claim is about.

### Added

- `regime`: two segmentation routes. `DiscreteRegimes` for an observed context (the distinct combinations
  actually seen, rounded so float noise does not make every sample its own regime). `KMeansRegimes` for a
  discovered one (plain-numpy k-means++ seeding, Lloyd iterations, restarts, deterministic given a seed,
  context centred and scaled first). Running both and reporting the gap measures the price of not being
  told the regime, which matters because a benchmark that ships its operating conditions as columns would
  otherwise flatter the method.
- A **novelty radius** on the clustered route. Each cluster carries the `novelty_quantile` of baseline
  distances to its centre, and a monitored sample beyond `novelty_factor` times that is labelled `-1`.
  Without it, a machine running somewhere the baseline never went receives a confident regime label and a
  meaningless residual that looks exactly like every other residual.
- `residual`: `RegimeResidualizer` with two models. `zscore` subtracts the per-regime baseline mean and
  divides by its standard deviation. `linear` regresses each channel on the context within the regime
  (ridge-regularised, because inside one regime the context barely moves, which is what made it a regime)
  and additionally removes the within-regime context dependence a coarse partition leaves behind.
- `make_arms(baseline, monitored, ...)`: builds both arms of the central comparison from an explicit
  baseline and an explicit monitored record. The raw arm is left untouched rather than standardised,
  since global standardisation is itself a one-regime residual model and would blur the contrast being
  measured. Context channels are excluded from monitoring by default, their residual being zero by
  construction.
- `kmeans_inertia_sweep`: returns the within-cluster sum of squares per k rather than picking a k.
  Automatic elbow detection routinely returns k = 2 for a machine with six operating conditions, because
  the two loaded states dominate the variance.
- 32 further tests (71 total), ruff clean.

### Measured

On synthetic data where the regime effect is emergent (both channels respond to both context variables
because the generator puts the physics there, not because a step was pasted in), the share of a channel's
variance living between regimes:

| arm | between-regime share |
|---|---|
| raw channel | above 0.90 |
| within-regime residual | below 0.02 |

With a continuous grade and only two regimes, the `zscore` residual keeps a correlation above 0.5 with
grade while `linear` drops below 0.15.

This shows the residual removes the regime. It does **not** yet show that removing the regime reduces
false alarms at a fixed detection delay; that needs detectors and the controlled C-MAPSS contrast, and it
may come back negative.

### Notes

Samples that cannot be residualised become `NaN`, never a pooled or nearest-regime fill: a fill would
manufacture a plausible number for the one situation the method has nothing to say about, and downstream
it would be indistinguishable from a real residual. If no regime is usable at all the fit raises, because
returning an all-NaN residual successfully is how a pipeline reports zero detections and looks like it
ran.

The leakage trap is demonstrated by a test rather than only documented: fitting the residual model on the
record being monitored folds the fault into the definition of normal and shrinks the measured fault step
to under three quarters of its honest size, with nothing raising and every output looking healthy.

## [0.01.000] - 2026-08-06

The data contract and the measurement layer. Nothing detects anything yet, on purpose: the way a
detector is SCORED is what every later claim rests on, so it is built and tested first.

### Added

- `types`: the data contract shared by every rung of the ladder. `Series` carries explicit, possibly
  irregular sample times so windows can be expressed in time rather than in samples and false-alarm
  rates can be reported per unit of time. `RegimeLabels` keeps unassigned samples visible as `-1` rather
  than folding them into the nearest regime. `Residual.as_series()` presents a residual as a plain
  `Series`, so a detector cannot tell which arm of the comparison it is in. `Detection` returns a
  continuous statistic and never a thresholded flag, oriented so larger always means more anomalous.
  `Attribution` carries the smearing caveat with it.
- `metrics`: detection delay, false alarms per unit time, the alarm-budget curve, threshold calibration
  against a budget, and bootstrap intervals resampled over UNITS rather than over samples.
- 39 tests, ruff clean, CI on Python 3.10 and 3.13.

### Notes

Two conventions in `metrics` are deliberate and are the reason the module exists in this shape.

**Alarms are counted as events, not as samples above a line.** A statistic that sits above the threshold
for two hundred samples is one alarm a human acknowledges once. Counting samples instead inflates every
rate by the dwell time of the statistic, and inflates a smooth method more than a spiky one, which
quietly rigs any comparison between them.

**An excursion that starts before the onset is a false alarm even if it is still running when the onset
arrives.** Detection is the first rising edge at or after the onset, so a detector alarming continuously
from the start of the record scores one false alarm and a miss, which is what it deserves.

### Fixed during development, and worth recording

The event-counting convention makes the false-alarm rate **non-monotone** in the threshold: it is zero at
a high bar, peaks in the middle where the statistic crosses in and out, and collapses again at a very low
bar where the statistic never comes back down and the whole record is one excursion. The first
implementation of `threshold_for_budget` scanned upward and therefore selected that low degenerate point,
which reports a superb false-alarm rate for a detector that is permanently on and detects nothing. The
search now descends from the never-fires end and stops at the first threshold that breaks the budget, so
it can only return a threshold in the region where lowering the bar actually buys sensitivity. Both the
non-monotonicity and the degenerate point have regression tests.
