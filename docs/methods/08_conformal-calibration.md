# 08 Conformal calibration

The rung that makes every other rung comparable, and the one the package's central claim depends on
being honest.

## The problem it solves

Every detector here returns a continuous statistic on its own arbitrary scale: a CUSUM in accumulated
sigma, an SPE in squared residual units, a matrix profile in normalised distance, an autoencoder in mean
squared error. Those numbers cannot be compared, and a threshold chosen on one means nothing on another.

Split conformal converts a score into a **p-value against a healthy calibration set**:

$$p_t = \frac{1 + \left|\{i : s_i \ge s_t\}\right|}{n_{\mathrm{cal}} + 1}$$

Two detectors on scales differing by six orders of magnitude land at the same p-value for the same
percentile of their own healthy behaviour. `test_it_puts_two_different_score_scales_onto_one_axis` checks
exactly that, with one detector scaled by 0.001 and the other by 1000.

## The guarantee

Under **exchangeability** of calibration and test scores, the p-value is super-uniform:

$$\Pr(p_t \le \alpha) \le \alpha$$

for any finite calibration set. Distribution-free: no normality, no asymptotics, no assumption about the
detector.

`test_the_finite_sample_bound_holds_on_exchangeable_data` asserts it directly, over 400 independent
replications at four values of $\alpha$, rather than arguing it from the formula.

### The `1 +` is load-bearing

The `1 +` in numerator and denominator is what makes the bound hold at finite $n$ rather than only
asymptotically. Dropping it is a natural-looking simplification and it breaks the guarantee **in the
direction of more alarms than promised**, which is the direction nobody checks.
`test_dropping_the_plus_one_would_break_the_bound` computes both forms on a small calibration set and
requires the naive one to fire more often.

### The resolution floor

The smallest achievable p-value is $1/(n_{\mathrm{cal}}+1)$. A budget below that is not strict, it is
**impossible**: no observation, however extreme, can produce a smaller p-value. `resolution` exposes it,
and a calibration set under 20 samples is rejected with that arithmetic in the error message.

## The assumption is FALSE here, and that is the point

Time series are not exchangeable. Consecutive samples from one truck are strongly dependent, the machine
drifts, and the operating context moves.

`test_the_bound_BREAKS_on_dependent_time_series_data` demonstrates it on an AR(1) process with
$\phi = 0.98$: across 60 records the realised rate at a nominal 0.05 has a standard deviation above 0.02,
and some records exceed twice nominal.

Quoting the guarantee without checking it against the data it runs on would be borrowed authority, which
is precisely what this package exists to avoid. Three things follow, and all three are implemented:

**`calibration_report` shows realised against nominal** on held-out healthy data, per sample and per
event, so the gap is visible rather than assumed away. It is the most useful output in the module.

**`AdaptiveConformal` recovers the long-run rate under nothing at all.** Following Gibbs and Candes:

$$\alpha_{t+1} = \alpha_t + \gamma\left(\alpha_{\mathrm{target}} - \mathbb{1}[p_t \le \alpha_t]\right)$$

When alarms come too often the level tightens; when they dry up it loosens. This is a genuinely different
guarantee: not marginal coverage under an assumption known to be false, but **long-run empirical
coverage irrespective of the data-generating process**. Measured on an AR(1) with $\phi = 0.99$ over
35,000 samples: realised within 0.01 of a 0.02 target.

**Calibration is per unit where possible**, since exchangeability across trucks is more nearly true than
exchangeability across time within one truck.

## A bug worth recording: the adaptive level must be allowed to go negative

The first implementation clipped $\alpha_t$ to $[0, 1]$. That looks obviously harmless. A negative level
already means "never alarm", since no p-value satisfies $p \le \alpha < 0$, so why carry it?

Because **the magnitude of the negative excursion is the debt** the recursion works off before firing
again, and that debt is what enforces the long-run rate.

With a saturated detector (every score above every calibration score, so every p-value sits at the
resolution floor), one alarm should cost $\gamma(1 - \alpha_{\mathrm{target}})$ of level and take roughly
$1/\alpha_{\mathrm{target}}$ quiet steps to repay. Clipped at zero the debt is discarded, the level climbs
back within two steps, and:

| variant | target | realised |
|---|---|---|
| clipped at 0 | 1% | **above 10%** |
| unclipped | 1% | within 0.5% of target |

`test_the_level_MUST_be_allowed_to_go_negative_or_the_guarantee_breaks` asserts both rows, computing the
clipped variant explicitly so the test fails if clipping ever stops breaking it.

The clipped version passed every other test in this file. It only failed in the saturated regime, which
is exactly the regime a badly calibrated detector lands in.

## Why this makes the central comparison honest

The package's claim is that a detector on regime-conditioned residuals beats the same detector on raw
channels.

Compared at each arm's own favourite threshold, that claim is **unfalsifiable**: any result can be
produced by choosing thresholds. Compared at a **fixed conformal false-alarm budget**, it is a
measurement.

Every benchmark in this package reads both arms off a common budget, and this module is what makes a
common budget mean the same thing on two different score scales.

## Practical notes

`conform` rewrites a detection so its statistic is $-\log_{10}(p)$ rather than $p$. Two reasons: the
package's orientation convention is that larger means more anomalous and a p-value runs the other way;
and a p-value floors at the resolution, so a raw-p statistic saturates at exactly the sensitive end where
an alarm-budget curve needs its resolution.

The calibration set must be **held out from the detector's own training** as well as from the test
record. Passing the detector's training data makes the scores optimistic, the p-values inflated, and the
realised false-alarm rate far above nominal.

`AdaptiveConformal`'s guarantee is on the **long-run average** over a record, not on any particular
window. A detector using it can emit a burst of alarms then stay quiet while the level recovers: the
average comes out right and the experience is lumpy. Worth knowing before it is presented to an operator
as a steady budget.

## References

- Vovk, V., Gammerman, A., Shafer, G. *Algorithmic Learning in a Random World* (2005).
  **UNVERIFIED** against a primary source in the research pass behind this package.
- Gibbs, I., Candes, E. J. "Adaptive Conformal Inference Under Distribution Shift." *NeurIPS* **34**,
  1660-1672 (2021). Achieves the desired coverage frequency over long time intervals irrespective of the
  true data-generating process.
- Xu, C., Xie, Y. "Conformal Prediction Interval for Dynamic Time-Series." *ICML 2021*, PMLR **139**,
  11559-11569. The ensemble route (EnbPI) to conformal inference without exchangeability; not implemented
  here, and its absence is stated rather than glossed.
- Martinez Gil, N., O'Donncha, F., Gifford, W. M., Zhou, N., Patel, D. C., Vaculin, R. "Adaptive Conformal
  Anomaly Detection with Time Series Foundation Models for Signal Monitoring."
  [arXiv:2604.20122](https://arxiv.org/abs/2604.20122) (2026). The p-value-as-false-alarm-rate framing
  used throughout this module.
