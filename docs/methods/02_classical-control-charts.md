# 02 Classical control charts

Shewhart, CUSUM, EWMA and Page-Hinkley. **The baseline this package claims to beat, so implemented
properly.**

A strawman baseline invalidates every comparison built on top of it, and the temptation to leave the
incumbent slightly broken is strongest precisely because a result that comes out the way everyone
expected is the one nobody checks. There is a second reason to take these seriously: on a real fleet, a
fixed threshold is what is actually running, so a planner comparing this package against their current
system is comparing it against these.

## Shared design

All four follow the same contract, with the steps as separate calls on purpose:

1. `fit(baseline)` establishes per-channel location and scale from a **healthy** record.
2. `detect(series)` returns a `Detection` carrying a continuous statistic.
3. The threshold is applied later, by the calibration layer, against a stated false-alarm budget.

### Multivariate reduction is the maximum, not the mean

Each detector runs per channel and reports

$$S_t = \max_j S_{t,j}$$

the machine being out of control if any channel is. Averaging would dilute a single-channel fault by the
channel count, so a twelve-channel machine would need a twelve-times-larger fault to reach the same
statistic. Onset evidence is usually one channel moving, so averaging is the wrong behaviour, and the
test `test_across_channels_the_reduction_is_max_not_mean` pins it: adding seven healthy channels must not
shrink the response to a fault in the eighth.

Per-channel statistics are kept in `Detection.meta["per_channel"]`, so
`Detection.channel_attribution(index)` can name the channel that carried the largest standardised
excursion. Read the [smearing caveat](../architecture/01_data-contract.md#attribution-and-why-it-is-not-a-cause)
before presenting that as a cause.

### NaN

A residual carries `NaN` wherever a sample could not be residualised. Every recursive detector here
leaves its state untouched on a NaN input and emits `NaN` for that sample. Treating NaN as zero would
feed a fabricated in-control observation into the accumulator, which is exactly the quiet fill this
package refuses elsewhere.

## Shewhart

$$S_t = \max_j \frac{|x_{t,j} - \mu_j|}{\sigma_j}$$

A threshold of 3 reproduces the familiar 3-sigma chart exactly; any other threshold is a point on the
same curve. `test_three_sigma_reproduces_the_published_false_alarm_rate` checks the implementation
against the **published** property (about 0.27% of in-control samples flagged under normality) rather
than against its own output, which is what makes it capable of catching a wrong standardisation.

Two weaknesses, both asserted by tests rather than merely described:

**No memory.** Degradation onset is a small persistent shift, and a chart looking at one sample at a time
needs that shift to be large before any single sample crosses. On a sustained half-sigma shift Shewhart
stays under 6 while CUSUM exceeds 50. CUSUM and EWMA exist for this reason.

**The in-control distribution is a mixture.** On a load-varying machine, a single $(\mu, \sigma)$ pair is
fitted to a mixture over operating regimes and fits none of them. This is the reason the package exists;
whether it actually costs false alarms is measured, not assumed.

Source: Shewhart, W. A. *Economic Control of Quality of Manufactured Product* (1931). **Not verified
against a primary source** in the research pass behind this package; recorded as such.

## CUSUM

Per channel, on standardised observations $z_t$:

$$S^+_t = \max(0,\; S^+_{t-1} + z_t - k), \qquad S^-_t = \max(0,\; S^-_{t-1} - z_t - k)$$

with the channel statistic $\max(S^+_t, S^-_t)$.

**The reset at zero is what makes it a detector.** Without it the sum is a random walk that wanders from
the origin and eventually crosses any threshold on noise alone. `test_resets_at_zero_so_it_is_not_a_random_walk`
runs 20,000 in-control samples and requires the statistic to stay bounded.

The reference value $k$ is conventionally half the shift you want to detect quickly; the default 0.5
targets a one-sigma shift. It is not a free knob: too small and the accumulator drifts upward on noise,
too large and it never accumulates. For a sustained shift $\delta > k$ the statistic grows at
approximately $\delta - k$ per sample, and `test_accumulates_a_persistent_shift_roughly_linearly` checks
that **rate** to 5%, not merely that it went up. Checking the rate is what catches a wrong reference
value.

Source: Page, E. S. "Continuous Inspection Schemes." *Biometrika* **41**(1-2), 100-115 (1954).
doi:[10.1093/biomet/41.1-2.100](https://doi.org/10.1093/biomet/41.1-2.100)

## EWMA

$$w_t = \lambda z_t + (1-\lambda) w_{t-1}, \qquad
\sigma_{w_t}^2 = \frac{\lambda}{2-\lambda}\left[1 - (1-\lambda)^{2t}\right], \qquad
S_t = \frac{|w_t|}{\sigma_{w_t}}$$

### The time-varying limit is not a refinement to skip

The asymptotic form $\lambda/(2-\lambda)$ **overstates** the spread of the early samples, so a chart
using it from the start is too insensitive at the beginning of every record. The exact limit is stronger
there by a factor of

$$\frac{1}{\sqrt{\lambda(2-\lambda)}}$$

which is 1.667 at $\lambda = 0.2$. On a fleet where records are short and restarts are common, that dead
zone at the start of every record is a real loss of detection, and it is invisible unless you test for
it. `test_the_early_control_limit_is_the_time_varying_one_not_the_asymptote` asserts that ratio exactly.

A clean consequence pins the whole variance factor: at $t=1$,

$$S_1 = \frac{\lambda|z|}{\sqrt{\frac{\lambda}{2-\lambda}\left[1-(1-\lambda)^2\right]}} = |z|$$

for **every** $\lambda$, since $\frac{\lambda}{2-\lambda}\lambda(2-\lambda) = \lambda^2$. If the variance
factor were wrong in any $\lambda$-dependent way these would disagree, so
`test_the_first_sample_is_lambda_independent_which_pins_the_variance_factor` is a strong check on a
single line of arithmetic.

At $\lambda = 1$ the chart reduces exactly to Shewhart, and that identity is tested to floating-point
tolerance.

### What $\lambda$ actually trades, stated carefully

The common phrasing "small $\lambda$ reacts slowly" is about the **raw** smoothed value and does not
survive standardisation. For a sustained shift $z$, the statistic approaches

$$S_\infty = \frac{|z|}{\sqrt{\lambda/(2-\lambda)}}$$

which **grows as $\lambda$ shrinks**. Small $\lambda$ is therefore the right choice for the small
persistent shift degradation onset actually is, and the test checks both maxima against that closed form
to 1%.

The genuine cost of small $\lambda$ is memory: after a transient spike, a long-memory chart holds the
excursion far longer. Since this package counts alarms as **events** rather than samples, that turns one
spike into one long excursion rather than into many alarms, which is a further reason the counting
convention matters. Both halves are tested.

Source: Roberts, S. W. "Control Chart Tests Based on Geometric Moving Averages." *Technometrics*
**1**(3), 239-250 (1959).
doi:[10.1080/00401706.1959.10489860](https://doi.org/10.1080/00401706.1959.10489860). Originally the
geometric moving average chart; the EWMA name came later.

## Page-Hinkley

$$m^+_t = \sum_{i \le t} (z_i - \delta), \quad M^+_t = \min_{i \le t} m^+_i, \quad
\mathrm{PH}^+_t = m^+_t - M^+_t$$

with the mirrored form for a downward change and the channel statistic the larger of the two.

**This is a CUSUM variant, not an independent rung.** It is labelled that way here rather than presented
as a fourth opinion, and `test_agrees_closely_with_cusum_because_it_is_the_same_construction` requires a
correlation above 0.98 between the two on the same data. Counting them as two independent classical
methods would inflate any claim of the form "we beat N baselines", and this package would rather have the
test than the larger number.

Source: Hinkley, D. V. (1971), building directly on Page (1954). The exact Hinkley reference was
**not verified against a primary source**, so it is recorded here as UNVERIFIED and the implementation
follows the standard formulation above rather than a specific paper's notation.

## What is not claimed here

None of these have been run against the regime-conditioned residual in a controlled comparison yet. This
release establishes that the baselines behave the way their sources say they behave. Whether conditioning
on regime reduces their false-alarm rate at a fixed detection delay is the question the C-MAPSS contrast
answers, and it may answer it negatively.
