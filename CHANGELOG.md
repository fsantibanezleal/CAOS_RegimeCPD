# Changelog

All notable changes to this project are documented here, newest first, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions use the `X.XX.XXX` display form; the
`pyproject.toml` manifest carries the same version in PEP 440 form with the padding dropped.

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
