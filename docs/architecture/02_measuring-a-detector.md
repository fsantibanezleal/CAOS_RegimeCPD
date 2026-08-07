# 02 Measuring a detector

Every claim this package makes is a number the measurement layer produced. A detector with a subtle bug
produces a wrong result; a metric with a subtle bug produces a wrong result that agrees with itself
everywhere and cannot be spotted from the outputs. So this page is the one to read first.

## Why not accuracy

Onset events are rare. On the SCANIA Component X fleet, 2,272 of 23,550 training vehicles carry a
failure label, roughly one in ten, and on a continuously monitored channel the imbalance per sample is
far more extreme than that. A detector that never fires scores superbly on accuracy and is worthless.

There is a sharper reason, from the published state of the art on that dataset. The best reported
**balanced accuracy** on its five-class time-window problem is $0.2428 \pm 0.01$
(<a href="https://arxiv.org/abs/2606.12486">Dimidov, Jafarnejad and Frank, 2026</a>), defined as the mean
per-class recall

$$\text{BA} = \frac{1}{5}\sum_{k=0}^{4} \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}.$$

Uniform random guessing over five classes scores $0.20$. The state of the art is about two points above
chance, and separates from its competitors on **cost**, not on any accuracy-shaped metric. That is the
strongest available argument that accuracy is the wrong instrument for this problem class.

What a maintenance planner actually trades is: how many nuisance call-outs per machine per month am I
willing to absorb, and at that budget, how early do I hear about a real problem. Those are the two
numbers this module computes.

## Alarms are events, not samples

A statistic that sits above the threshold for two hundred consecutive samples is **one** alarm. A human
acknowledges it once. So the package counts **rising edges**: indices where the alarm mask transitions
from false to true.

$$\mathcal{E}(\tau) = \{\, i : s_i \ge \tau \ \wedge\ (i = 0 \ \vee\ s_{i-1} < \tau) \,\}$$

A mask already true at $i = 0$ counts as an edge, because a record that opens mid-excursion is an alarm
the operator sees.

Counting samples instead would inflate every rate by the **dwell time** of the statistic, and inflate a
smooth method more than a spiky one. Since regime-conditioned residuals are typically smoother than raw
channels, sample counting would have systematically biased this package's own headline comparison in its
own favour. That is worth stating plainly: the convention was chosen before the result, and it is the
conservative choice for the claim being made.

Warm-up samples where the statistic is `NaN` never alarm. Undefined is not "quiet"; it is excluded.

## Detection, misses, and the excursion that spans the onset

Given a true onset time $t^\*$, with $T_{\mathcal{E}}$ the times of the rising edges:

- **False alarms**: $\lvert \{ t \in T_{\mathcal{E}} : t < t^\* \} \rvert$
- **Detection**: the first $t \in T_{\mathcal{E}}$ with $t \ge t^\*$
- **Detection delay**: $t_{\text{detect}} - t^\*$, undefined when there is no such edge

The consequential case is an excursion that **begins before the onset and is still running when the
onset arrives**. This package counts it as a false alarm and records a miss.

The alternative convention (any alarm covering the onset counts as a detection) rewards a detector for
being permanently on. That is the degenerate solution the whole package exists to avoid: a detector
alarming continuously from the start of the record would score a perfect detection rate. Under the
convention used here it scores one false alarm and a miss, which is what it deserves.

## Rates are pooled over exposure

**Healthy exposure** for a unit is the time during which any alarm would be false: from the start of the
record to the onset, or the whole record for a healthy unit.

$$\text{FA rate} = \frac{\sum_u \lvert \{ t \in T_{\mathcal{E}}^{(u)} : t < t^{*(u)} \} \rvert}{\sum_u E_u}$$

Pooled, not averaged over units, so a machine observed for a week cannot carry the same weight as one
observed for a year. `FleetScore.per(time_units)` converts into planner units: with `t` in hours,
`per(730.0)` gives false alarms per machine-month.

Healthy units are not filler. They are where the false-alarm rate is actually measured, and a benchmark
run only on faulty units cannot report one at all.

## The alarm-budget curve, and a trap inside it

The comparison instrument is the curve of (false-alarm rate, detection delay) traced as the threshold
sweeps. Two methods read off **at the same false-alarm rate** is the only comparison that a threshold
choice cannot rig.

### The event-counted rate is not monotone in the threshold

This is counter-intuitive and it silently breaks the obvious implementation, so it is documented rather
than left as a surprise. With event counting the rate is **unimodal**:

| threshold | behaviour of the statistic | events |
|---|---|---|
| very high | never crosses | 0 |
| middle | crosses in and out repeatedly | **peak** |
| very low | sits above the line for the entire record | 1 |

The low end is the trap. A detector pinned permanently above its threshold reports an excellent
false-alarm rate (one excursion per unit) while detecting nothing at all, because its single excursion
begins before every onset and is therefore false.

A search that scans upward from the bottom of the grid and returns the first threshold meeting the budget
returns exactly that degenerate point. The resulting benchmark row reads as a spectacularly cheap
detector. This was the first implementation in this package, and it was caught by a test asserting a
monotonicity that does not hold.

`threshold_for_budget` therefore **descends from the never-fires end** and stops at the first threshold
that breaks the budget, returning the lowest threshold in the region connected to never-firing. That is
the region where lowering the bar genuinely buys sensitivity. Both the non-monotonicity and the
degenerate point carry regression tests.

When even the top of the grid fails the budget, the function returns `None`. A detector can be unable to
operate at a given budget at all, and returning the grid maximum would disguise "cannot operate" as
"operates and detects nothing".

## Intervals are bootstrapped over units

A point number with no interval is not a result.

Samples within one machine are strongly dependent: consecutive readings of a strut pressure are nearly
the same reading. Resampling samples produces intervals that can be an order of magnitude too narrow and
a benchmark that looks far more certain than it is. `bootstrap_ci` resamples **whole units** with
replacement and recomputes the fleet statistic, taking percentile bounds over the replicates.

Replicates that come back non-finite are dropped rather than propagated: a resample can legitimately
contain no faulty unit and so have no defined detection rate. That loss is not hidden, it shows up as a
wider interval.

## References

- Dimidov, V., Jafarnejad, S., Frank, R. "An Empirical Study on Predictive Maintenance for Component X
  in Heavy-Duty Scania Trucks." arXiv:2606.12486 (2026).
  <https://arxiv.org/abs/2606.12486>. Source of the balanced-accuracy figure and of the cost-based
  scoreboard that motivates cost and delay over accuracy.
- Kharazian, Z., Lindgren, T., Magnusson, S., Steinert, O., Andersson Reyna, O. "SCANIA Component X
  dataset: a real-world multivariate time series dataset for predictive maintenance." *Scientific Data*
  **12**, 493 (2025). doi:[10.1038/s41597-025-04802-6](https://doi.org/10.1038/s41597-025-04802-6).
  Source of the class balance quoted above.
