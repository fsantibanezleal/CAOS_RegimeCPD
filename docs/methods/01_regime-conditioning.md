# 01 Regime conditioning

The stage every other rung sits downstream of, and the one this package's central claim is about.

## The problem

On a machine whose operating context varies, a measured channel $x$ is a function of both the health
state and the context $c$:

$$x_t = f(c_t) + g(\text{health}_t) + \varepsilon_t$$

and on real machinery $\lVert f \rVert \gg \lVert g \rVert$. A haul truck climbing a ramp loaded shows
its strut pressure rise by tens of bar; the strut leak you want to catch moves it by a fraction of that,
over weeks. The regime term dominates, so any detector applied to $x_t$ directly is a detector of $c_t$.

This is not a subtlety that shows up in an edge case. It is the first-order behaviour, and it is the
reason condition monitoring on constant-load benchmarks (XJTU-SY, FEMTO, IMS) does not transfer to fleet
telemetry: those benchmarks fix $c$ by construction, so $f(c)$ is a constant and a rising RMS really does
mean damage.

## The construction

Estimate $\hat{f}$ from a **healthy baseline**, then monitor the residual:

$$r_t = \frac{x_t - \hat{f}(c_t)}{\hat{\sigma}(c_t)}$$

Two estimators are provided.

**Per-regime standardisation** (`method="zscore"`). Partition the context space into $K$ regimes, and
within regime $k$ take

$$\hat{f}(c_t) = \mu_k, \qquad \hat{\sigma}(c_t) = \sigma_k$$

the baseline mean and standard deviation of that channel among baseline samples assigned to $k$. Plain,
robust, and assumes nothing about the functional form of $f$ beyond it being roughly constant inside a
regime.

**Per-regime regression** (`method="linear"`). Within regime $k$, regress each monitored channel on the
context with a ridge term:

$$\hat{B}_k = \left( C_k^{\top} C_k + \lambda I \right)^{-1} C_k^{\top} X_k, \qquad
\hat{f}(c_t) = [c_t, 1]\,\hat{B}_k$$

then standardise the regression residual. This additionally removes variation *within* a regime that is
still explained by context, which matters whenever the partition is coarse: six clusters standing in for
a continuum of road grades leave a great deal of grade dependence inside each cluster. The ridge term is
not cosmetic. Inside one regime the context variables barely move, which is what made it a regime, so
$C_k^{\top} C_k$ is routinely near-singular.

The test `test_linear_removes_within_regime_context_dependence_that_zscore_leaves` measures exactly this:
with a continuous grade and only two regimes, the zscore residual retains a correlation above 0.5 with
grade, and the linear residual drops below 0.15.

## Getting the regimes

### Observed regime

The context is handed to you. NASA C-MAPSS carries three operational setting columns; a truck fleet might
carry a loaded flag and a gear indicator. `DiscreteRegimes` takes the distinct observed combinations.

Context values are rounded before combinations are formed, because real discrete settings arrive as
floats with noise in the last digits. Without rounding, every sample becomes its own regime, which looks
like it worked and is identical to having no regimes at all.

### Discovered regime

Nobody tells you. `KMeansRegimes` clusters the context: k-means++ seeding, Lloyd iterations, restarts
keeping the lowest inertia, deterministic given a seed. Context is centred and scaled first, or a feature
measured in Pascals dominates one measured in radians purely through its units.

k-means++ rather than random seeding is a real choice here. Random seeding regularly puts two centres
inside one dense cluster and none in a sparse one, and Lloyd's algorithm cannot recover from it. On
regime data the sparse clusters are the unusual operating conditions, which are exactly the regimes worth
having.

### Why both routes exist

Running both and reporting the gap **measures the price of not being told the regime**. A package that
only implemented the observed route would flatter itself on any benchmark that ships its operating
conditions as columns, and C-MAPSS is precisely such a benchmark. Reporting only the observed-regime
number there would be a result about the benchmark's convenience rather than about the method.

## The novelty radius

k-means assigns every point to its nearest centre regardless of distance. A machine running in a context
the baseline never contained would therefore receive a confident regime label and a meaningless residual,
one that looks exactly like every other residual.

So each cluster carries a radius, the `novelty_quantile` of baseline distances to that centre, and a
monitored sample farther than `novelty_factor` times that radius is labelled $-1$. Unassigned samples
become `NaN` in the residual and are reported through `RegimeLabels.coverage`.

This converts "I have never seen this" from a silent wrong answer into a state the caller can check. The
tests pin both directions: context far outside the baseline must be rejected, and ordinary held-out
baseline variation must stay inside (coverage above 0.95), because a radius tight enough to reject normal
operation would quietly disable the method.

## Samples that cannot be residualised

Two cases produce `NaN`:

1. The sample is in no regime ($-1$).
2. The sample is in a regime with fewer than `min_samples` baseline members, so its statistics are not
   trustworthy. A standard deviation from four points is noise.

`NaN` rather than a pooled or nearest-regime fill. The fill would manufacture a plausible number for the
one situation where the method has nothing to say, and downstream it would be indistinguishable from a
real residual. `NaN` keeps the time axis intact, so time-based windows and healthy exposure stay correct,
and every detector in this package treats it as "statistic undefined here" and refuses to alarm on it.

If **no** regime is usable the fit raises rather than returning an all-NaN residual. Returning nothing
successfully is how a pipeline reports zero detections and looks like it ran.

## Leakage: how this stage actually fails

The residual model is fitted on a healthy baseline and applied to the monitored record. Fit it on the
record being monitored and the fault is folded into the definition of normal, so the residual has the
fault partly subtracted out of itself.

**Nothing raises.** The residual looks well behaved, coverage is high, and detection performance is worse
for a reason that appears nowhere in any output. `test_fitting_on_the_faulty_record_leaks_and_shrinks_the_fault`
demonstrates it rather than asserting it: fitting on the faulty record shrinks the measured fault step to
under three quarters of its honest size.

`make_arms(baseline, monitored, ...)` exists so the split is two arguments rather than a habit.
`fit_transform` remains available for inspecting the residual on a baseline known to be healthy, and its
docstring says so.

## Keeping the comparison fair

`make_arms` returns both arms over the same channels, the same time axis and the same length. Three
things it does deliberately:

- **The raw arm is left untouched**, not standardised. Global standardisation is a one-regime residual
  model, so standardising the raw arm would be a partial regime correction and would blur the very
  contrast being measured.
- **Context channels are excluded from monitoring** by default. Their residual is zero by construction,
  so including them adds a channel that can never carry a fault and dilutes every multivariate statistic.
- **Both arms come back as plain `Series`**, so the detector receives an indistinguishable object and has
  no mechanism by which to behave differently.

## What is measured, and what is not

`test_residual_is_regime_flat_while_the_raw_channel_is_not` measures the property this page claims. On
synthetic data where the regime effect is emergent (both channels respond to both context variables
because the generator puts the physics there, not because a step was pasted in):

| arm | share of variance living BETWEEN regimes |
|---|---|
| raw channel | above 0.90 |
| within-regime residual | below 0.02 |

A raw channel whose variance is over 90% between-regime is largely reporting which regime the machine is
in. That is what a change detector latches onto.

**This is not yet the product's central claim.** It shows the residual removes the regime; it does not
show that removing the regime helps a detector. That requires detectors (E3 onward) and the controlled
C-MAPSS contrast, which has since been run downstream by TruckVitals. The measured axis is DETECTION at
a matched false-alarm budget, and there it came back positive (0.17 recovering to 0.95 under six
regimes); the false-alarm-reduction framing this paragraph originally used was WITHDRAWN downstream,
because neither lane demonstrates it. What is established here is that the input to that experiment is
what it says it is.

`test_a_fault_survives_residualisation` pins the other side: conditioning must remove the regime and not
the signal. A residual that also flattened the fault would be useless in the most flattering possible
way, since every stability metric would improve.

## References

- Hendrickx, K., Meert, W., Mollet, Y., Gyselinck, J., Cornelis, B., Gryllias, K., Davis, J.
  "A general anomaly detection framework for fleet-based condition monitoring of machines."
  *Mechanical Systems and Signal Processing* (2020).
  doi:[10.1016/j.ymssp.2019.106585](https://doi.org/10.1016/j.ymssp.2019.106585);
  preprint [arXiv:1912.12941](https://arxiv.org/abs/1912.12941).
  Detects a faulty machine by online comparison **against the rest of the fleet**, assuming most of the
  fleet is healthy. That is the FLEET axis, orthogonal to the REGIME axis implemented here (comparing a
  machine against itself within a matched context). Both belong in a complete system, and the two-way
  combination is the interesting cell.
- Dimidov, V., Jafarnejad, S., Frank, R. "An Empirical Study on Predictive Maintenance for Component X in
  Heavy-Duty Scania Trucks." [arXiv:2606.12486](https://arxiv.org/abs/2606.12486) (2026). The current
  best-performing published method on that dataset lists among its own limitations that it "doesn't
  account for the dynamic nature of the fleets and their operating conditions", requiring periodic
  retraining. That is the state of the art conceding this gap in its own limitations section.
- Saxena, A., Goebel, K., Simon, D., Eklund, N. "Damage propagation modeling for aircraft engine
  run-to-failure simulation." *2008 International Conference on Prognostics and Health Management*,
  pp. 1-9. C-MAPSS: FD001 has one operating condition and FD002 has six, with fault mode held fixed, so
  the pair varies exactly one factor and is the controlled public test of regime conditioning.
