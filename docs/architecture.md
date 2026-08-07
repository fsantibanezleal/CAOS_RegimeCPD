# Architecture

How the package is put together, and specifically how it keeps the central comparison honest.

The architecture is not incidental to the science here. The claim this package is built to test is that
running a detector on within-regime residuals produces fewer false alarms than running the same detector
on raw channels. That comparison is worthless if the two arms differ in any way other than the one being
tested, and most of the design below exists to make an accidental difference impossible rather than
merely discouraged.

## Pages

| Page | Covers |
|---|---|
| [01 The data contract](architecture/01_data-contract.md) | `Series`, `RegimeLabels`, `Residual`, `Detection`, `Attribution`: what each guarantees and why |
| [02 Measuring a detector](architecture/02_measuring-a-detector.md) | Detection delay, false alarms per unit time, the alarm-budget curve, threshold calibration, unit bootstrap |

## The shape

```
Series  ---------------------------------> Detector -> Detection -> UnitOutcome -> FleetScore
   |                                          ^
   +-> RegimeSegmenter -> RegimeLabels        |
   |                          |               |
   +--------------------------+-> Residual ---+   (Residual.as_series() makes this edge identical)
```

The two paths into the detector are deliberately the same type. `Residual.as_series()` returns a plain
`Series`, so a detector receives an indistinguishable object in both arms and has no mechanism by which
to behave differently. This is stronger than a convention that both arms "should" be treated the same,
because it removes the possibility rather than discouraging it.

## Four invariants

**1. A detector never chooses its own threshold.** Every detector returns a continuous statistic in a
`Detection`. Thresholds are applied afterwards, by the calibration layer, against a stated false-alarm
budget. Two methods compared at their own preferred operating points say nothing about each other.

**2. Larger always means more anomalous.** Some underlying statistics are naturally inverted (a
likelihood, a run-length mode). They are flipped at the source, never at the call site. A single
sign error in a benchmark harness produces a method that appears to perform exactly as badly as it should
perform well, and that is very hard to see in a results table.

**3. Time is explicit and may be irregular.** `Series` carries `t`. Fleet telemetry is irregular,
multi-rate and gappy; a machine idle for a shift emits nothing. Indexing by sample would report a
different false-alarm rate for a machine that idles more, which is a property of the duty cycle rather
than of the detector.

**4. Unassignable is a state, not a rounding error.** `RegimeLabels` marks a sample that fits no known
regime as `-1` rather than snapping it to the nearest. A machine operating in a context the baseline
never saw is exactly the case where the residual is not trustworthy, and hiding it inside the closest
cluster turns a known unknown into a silent one.

## Dependencies

The core is **numpy and nothing else**. Every detector, the PCA, the change-point recursions and the
conformal calibration are written against numpy directly. `scikit-learn` appears only in the `learned`
extra, for the healthy-only novelty models that genuinely need it. A monitoring library that pulls in a
full ML stack to compute a CUSUM has misplaced its boundary, and CI asserts the core install really is
numpy-only rather than trusting the README to stay true.
