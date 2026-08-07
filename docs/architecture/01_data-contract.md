# 01 The data contract

Five types, shared by every rung of the ladder. They exist so that both arms of the central comparison
travel through identical structures, and so that a shape error surfaces at construction rather than as a
benchmark number.

## `Series`: a multivariate record from one unit

```python
Series(t, x, names, unit_id="unit")
```

| field | shape | meaning |
|---|---|---|
| `t` | `(n,)` | sample times, strictly increasing |
| `x` | `(n, d)` | channel values; a 1-D array is accepted as `d = 1` |
| `names` | `d` | channel names |
| `unit_id` | | which machine this came from |

### Why time is explicit

Fleet telemetry is irregular, multi-rate and gappy. A truck parked for a shift produces no readouts; two
trucks do not share a clock. The SCANIA Component X descriptor states outright that "vehicles do not
follow a uniform sampling frequency"
(doi:[10.1038/s41597-025-04802-6](https://doi.org/10.1038/s41597-025-04802-6)).

Carrying `t` rather than assuming a fixed interval buys two things. Windows can be expressed in **time**
rather than in samples, which is the only definition that means the same thing across two machines with
different duty cycles. And the false-alarm rate can be reported per unit of **time**, which is what a
planner budgets. A package that indexes by sample silently reports a different false-alarm rate for a
machine that idles more, and that difference is a property of the duty cycle, not of the detector.

Strictly-increasing `t` is enforced. An unsorted series produces negative durations and nonsense rates
without raising anything.

### Why names, not columns

Attribution output is only actionable if it can say "strut pressure". `channel("strut")` and
`select(["strut", "brake"])` index by meaning, so a reordered input cannot silently change which channel
a result is about.

### Why `unit_id` is carried

Intervals are bootstrapped over units (see
[02 Measuring a detector](02_measuring-a-detector.md#intervals-are-bootstrapped-over-units)). That is
impossible once unit identity has been dropped, and it is very easy to drop it while concatenating a
fleet into one array.

## `RegimeLabels`: which operating context each sample belongs to

```python
RegimeLabels(labels, n_regimes, method, meta={})
```

`labels[i] == -1` marks a sample no regime claimed.

### Why `-1` is not folded into the nearest regime

A machine operating in a context the baseline never saw is exactly the case where the within-regime
residual is **not trustworthy**: there is no fitted local model to subtract. Snapping such a sample to
its nearest cluster produces a residual that looks like every other residual and is meaningless, which
converts a known unknown into a silent one.

`coverage` reports the fraction of samples that were assigned. A low coverage invalidates the residual
for that unit, and it is meant to be checked rather than assumed.

## `Residual`: what is left after the regime is accounted for

```python
Residual(t, r, names, regimes, unit_id="unit")
```

This is the object the central comparison turns on. Arm A runs a detector on a `Series`; arm B runs the
identical detector on a `Residual`.

### `as_series()` is the mechanism that keeps the comparison fair

```python
detector(series)                 # arm A
detector(residual.as_series())   # arm B
```

`as_series()` returns a plain `Series`. The detector receives an indistinguishable object and has **no
mechanism** by which to behave differently between arms. This is stronger than a convention that both
arms should be treated the same: it removes the possibility rather than discouraging it.

The regime labels stay attached to the residual, so any result can be traced back to the segmentation
that produced it. A false-alarm reduction obtained from a segmentation with 40% coverage is a different
claim from one obtained at 99%, and without the labels attached the two are indistinguishable after the
fact.

## `Detection`: a statistic, never a flag

```python
Detection(t, statistic, method, meta={})
```

Two guarantees.

**Larger means more anomalous, for every method in this package.** Some underlying statistics are
naturally inverted: a likelihood, a run-length mode. Those are flipped at the source rather than at the
call site, because a single sign error in a benchmark harness produces a method that appears to perform
exactly as badly as it should perform well, and that failure is nearly invisible in a results table.

**No threshold is baked in.** `alarms(threshold)` exists, but the threshold comes from the calibration
layer. A detector that returns a boolean cannot be put on a common alarm budget, and comparing methods at
their own favourite thresholds is the easiest way to produce a meaningless benchmark.

`NaN` is treated as not-alarming rather than propagated: a detector needs a warm-up before its statistic
is defined, and warm-up must not be able to raise an alarm.

## `Attribution`: which channels are implicated

```python
Attribution(scores, names, method, index=None)
```

`ranked()` orders channels most-implicated first; `top(k)` gives the `k` names.

### Attribution, and why it is not a cause

Contribution-style attribution **smears**. A single faulty channel spreads score onto every channel
correlated with it, because the underlying statistic is computed in a rotated space where the fault is no
longer aligned with one axis. Westerhuis, Gurden and Smilde treat this properly
(doi:[10.1016/S0169-7439(00)00062-9](https://doi.org/10.1016/S0169-7439(00)00062-9)).

So a high score means "this channel is implicated", never "this channel is the cause". The honest use is
to **narrow a candidate set** from twelve channels to three. Any interface built on top of this should
say "variables that contributed to the excursion", not "root cause".

The honest cross-check is to obtain an attribution by a completely different route and see whether the
two agree. This package provides one: mSTAMP returns the anomalous window together with the subset of
dimensions involved, by a mechanism that shares nothing with PCA contributions. Agreement between two
unrelated methods is worth considerably more than either alone.

Scores are non-negative and comparable within one attribution. They are **not** probabilities and do not
sum to one; presenting them as percentages of blame would reintroduce exactly the causal reading the
smearing caveat rules out.
