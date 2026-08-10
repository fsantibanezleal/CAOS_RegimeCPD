# Changelog

All notable changes to this project are documented here, newest first, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions use the `X.XX.XXX` display form; the
`pyproject.toml` manifest carries the same version in PEP 440 form with the padding dropped.

## [0.09.002] - 2026-08-10

### Fixed

**ADWIN's `epsilon_cut` used a harmonic term matching neither the paper nor MOA**, while this package's
own docstring and method page asserted it was the paper's. Found by opening the primary source, not by a
test.

Bifet and Gavalda (2007), Section 3.2, verbatim: `m = 1 / (1/n0 + 1/n1) (harmonic mean of n0 and n1)`.
This package used `1/m = 1/(n0-1) + 1/(n1-1)`. MOA's reference implementation uses a third form again,
`1/(n0-4) + 1/(n1-4)`, from its `mintMinWinLength = 5`. Corrected to the paper.

The old term is strictly larger, so `m` was smaller and `epsilon_cut` **larger**: the detector was more
conservative than the guarantee it advertised. **The measured cost was one sample.** Head to head over 20
seeds of a three-sigma shift and 5 seeds of 5000 stationary samples at `delta = 0.002`, median detection
delay went from 8 to 7 with zero stationary false alarms on both. The correction matters because a reader
reproducing the method from the docs would have got the paper's answer and disagreed with the code, not
because the numbers moved.

Two lessons recorded rather than fixed away. The previously documented behaviour, "at most 2 detections
in 5000 stationary samples", was *consistent* with the wrong formula and was therefore no evidence it was
right; a quiet detector is also what an over-large threshold looks like. And the whole existing drift
suite passed both before and after, which is exactly what a behavioural test does when the defect is
worth `O(1/n^2)`. The new `test_cut_threshold_is_the_papers_equation_3_1` recomputes the closed form from
the paper instead.

`delta' = delta/n` is **kept** over MOA's `delta/ln n` and now documented as deliberate: the paper
justifies the looser value by ADWIN2 checking only `O(log n)` subwindows, and this class is the direct
ADWIN, which checks all `O(n)` splits.

### Changed

- ADWIN's guarantee is now quoted verbatim from the primary source rather than paraphrased, with the
  note that **Theorem 3.1 is stated for the direct algorithm implemented here**, not for ADWIN2. The
  canonical `cs.upc.edu` PDF URL is dead; the working route is recorded in the method page.
- The Hotelling (1947) citation is corrected and promoted from UNVERIFIED. The book is *Selected
  Techniques of Statistical Analysis*, the chapter title is longer than the usual short form, and the
  chapter is **not** the origin of `T^2`, which Hotelling himself attributes on p. 114.

## [0.09.001] - 2026-08-10

### Fixed

**A channel constant to within floating-point noise was being standardised by that noise**, producing
z-scores around 1e14. Every scaler in the package guarded with `spread > 0`, which a strictly positive
7e-15 passes.

Found on real data, not in review. NASA C-MAPSS `sensor_06` holds 21.61 across a whole baseline window
and reports a standard deviation of 7.1e-15, on 43 of 100 FD001 units. Later in the record the sensor
reports the adjacent quantisation level, 21.60. That utterly ordinary 0.01 step becomes a z-score of
1.4e12, and a CUSUM accumulated it to **8.4e12** over a healthy stretch.

The consequence was not a crash. Because the multivariate statistic is a maximum across channels, one
unit doing this set the fleet-wide threshold for an entire benchmark, and detection rate at a fixed
false-alarm budget fell from 0.79 to **0.07**. Every individual unit still looked fine. The benchmark
simply reported that the method did not work.

- New `regimecpd.scaling` with one shared rule: a channel is standardisable only when
  `sigma > max(atol, rtol * |mu|)`, with `rtol = 1e-8`, roughly the square root of double precision.
  That sits far below any real quantisation step (a sensor reporting two decimals has a relative spread
  near 1e-4) and far above representation noise, so it separates the two without judgement calls.
- Applied in `classical`, `spc`, `regime`, `residual` and `novelty`, which were the five places the weak
  guard had been copied to.
- 12 regression tests reproducing the real mechanism, including one that asserts the OLD guard would
  still explode, so the suite fails if the fixture ever stops reproducing the defect.

## [0.09.000] - 2026-08-08

Conformal calibration. **Every rung on the ladder is now implemented.**

### Added

- `conformal.SplitConformal`: p-values against a healthy calibration set, with the finite-sample
  guarantee `Pr(p <= alpha) <= alpha` under exchangeability. Exposes `resolution`, the smallest reachable
  p-value, because a budget below `1/(n_cal+1)` is not strict but impossible.
- `conformal.AdaptiveConformal`: the Gibbs and Candes online level correction, which recovers the long-run
  rate irrespective of the true data-generating process. A genuinely different guarantee from marginal
  coverage under an assumption known to be false.
- `conformal.conformalise` to put a whole fleet on one scale, and `conformal.calibration_report` to show
  realised against nominal, per sample and per event.
- 23 further tests (290 total), and
  [docs/methods/08_conformal-calibration.md](docs/methods/08_conformal-calibration.md).

### Why this rung matters more than its size

Every detector returns a statistic on its own arbitrary scale, so a threshold chosen on one means nothing
on another. Conformal p-values put two detectors differing by six orders of magnitude onto the same axis,
which is what lets the package's central claim be a measurement rather than an unfalsifiable assertion:
compared at each arm's own favourite threshold anything can be produced; compared at a fixed conformal
budget, the result is a number.

### Verified, and then verified to fail

The bound is asserted **directly**, over 400 independent replications at four values of alpha, rather
than argued from the formula. The `1 +` in numerator and denominator is shown to be load-bearing: the
naive form fires more often, breaking the guarantee in the direction of more alarms than promised.

Then the honest half. The bound is shown to **BREAK on dependent time-series data**, which is the case
this package actually operates in: on an AR(1) with phi = 0.98, the realised rate at a nominal 0.05 has a
standard deviation above 0.02 across records and some records exceed twice nominal. Quoting a guarantee
without checking it against the data it runs on is borrowed authority.

The adaptive variant answers that: on an AR(1) with phi = 0.99 over 35,000 samples it lands within 0.01
of a 0.02 target.

### Fixed: the adaptive level must be allowed to go negative

The first implementation clipped alpha to [0, 1], which looks obviously harmless, since a negative level
already means "never alarm".

It is not harmless. The magnitude of the negative excursion is the **debt** the recursion works off
before firing again, and that debt is what enforces the long-run rate. With a saturated detector (every
p-value at the resolution floor) the clipped version discards the debt, recovers within two steps, and
realises above 10% against a 1% target; unclipped it lands within 0.5%.

The clipped version passed every other test in the file. It failed only in the saturated regime, which is
exactly where a badly calibrated detector ends up. There is a regression test that computes the clipped
variant explicitly, so it fails if clipping ever stops breaking things.

### Notes

`conform` reports `-log10(p)` rather than `p`: the orientation convention is that larger means more
anomalous, and a raw p-value saturates at the resolution floor, which is exactly the sensitive end where
an alarm-budget curve needs resolution.

The adaptive guarantee is on the long-run average over a record, not on any window. A detector using it
can emit a burst then stay quiet while the level recovers: the average is right and the experience is
lumpy.

EnbPI (Xu and Xie, 2021) is not implemented, and its absence is stated rather than glossed. Vovk et al.
(2005) remains UNVERIFIED against a primary source.

## [0.08.000] - 2026-08-08

Streaming drift detectors and healthy-only novelty models. The SOTA tier is now complete.

### Added

- `drift.ADWIN`: adaptive windowing with the Hoeffding-style cut bound. The one detector here that
  arrives with bounds on both its false positive and false negative rate rather than a threshold to
  sweep.
- `drift.KSWIN`: two-sample Kolmogorov-Smirnov windowing, comparing whole distributions rather than
  means.
- `drift.ks_two_sample_pvalue`: the KS statistic and its asymptotic p-value, implemented by hand to keep
  the core numpy-only.
- `novelty.IsolationForestDetector`, `novelty.OneClassSVMDetector` (extra `learned`) and
  `novelty.AutoencoderDetector` (extra `deep`, CUDA when available), all trained on healthy data only,
  all supporting a stacked window so they can see temporal structure.
- A new `deep` extra, kept separate from `learned` because PyTorch is a two-gigabyte install with a CUDA
  variant and a user who wants an isolation forest should not pay for it.
- 43 further tests (267 total), and
  [docs/methods/07_drift-and-novelty.md](docs/methods/07_drift-and-novelty.md).

### The sign, handled once at the source

scikit-learn's novelty estimators return higher scores for more NORMAL samples; this package returns
higher for more anomalous. A missed flip produces a detector that scores exactly as badly as it should
score well, whose budget curve looks like a plausible weak method rather than an error, and which reads
on a benchmark table as "this approach does not work for this problem". There is an orientation test for
every model.

### Measured

- ADWIN at delta = 0.002: at most 2 detections in 5000 stationary samples; a three-sigma shift found
  within 60 samples; a stationary stretch at the NEW level goes quiet again (at most 2 in 1500), so it
  adapts rather than alarming forever.
- ADWIN is largely blind to a variance-only change (at most 3 detections in 1600 samples), which is the
  entire justification for carrying KSWIN. On the same change, KSWIN's statistic rises above three times
  its quiet median.
- The KS **D** statistic matches scipy exactly. The p-value uses the Numerical Recipes correction term
  where scipy does not, agreeing to within about 1% at p = 0.9 and 2% at p = 8e-4. In the deep tail the
  relative difference reaches about 50% at p = 2e-9 and means nothing, since both are approximations to
  something around a billionth. Asserted where a p-value is actually read, and by order of magnitude
  beyond.
- A windowed model's contrast on a burst of fast oscillation with the SAME amplitude as the baseline (so
  the marginal distribution is nearly untouched) is more than three times a per-sample model's.

### Two limitations recorded rather than smoothed over

**KSWIN's per-step alpha is not the realised rate**: about 1.2x nominal at alpha = 0.05 and 1.7x at
alpha = 0.005, with no multiple-testing correction. And **single-record variance is enormous**, because
overlapping windows make the tests strongly dependent so alarms arrive in clusters: one seed gave 0.0005
and another 0.008 at the same nominal alpha, a factor of sixteen. Both are inherent to the method, and
the clustering is a direct argument for counting alarm events rather than samples.

**ADWIN's statistic is a small integer count**, so its alarm-budget curve has only d+1 points where the
others have hundreds. `delta` should be swept instead whenever ADWIN is compared against anything.

### Notes

Only the direct ADWIN is implemented, not the exponential-histogram ADWIN2. Same cuts, so no loss of
detection quality, but the per-item cost grows with the window and `max_window` bounds it.

An autoencoder bottleneck at least as wide as the input is rejected: such a network learns the identity,
reconstructs a fault perfectly, and gives a detector that never fires while looking well trained.

The one-class SVM records the training count it actually used after subsampling, because a model quietly
trained on a fraction of the baseline is a different model.

Scholkopf et al. (2001) and the KSWIN reference remain UNVERIFIED against primary sources.

## [0.07.000] - 2026-08-08

mSTAMP: the second, independent answer to "which variables", plus a finding and a measured correction to
received wisdom.

### Added

- `mstamp.MatrixProfile`: the multidimensional matrix profile. Distances by MASS (FFT sliding dot
  product, so a distance profile is O(n log n) rather than O(nm)); sliding window statistics from prefix
  sums; the k-dimensional profile for every k in one pass.
- `matching_subset`: mSTAMP's own k-channel subset.
- `anomalous_channels`: the discord attribution, per-channel distance to the full-dimensional nearest
  neighbour.
- 24 further tests (224 total), and
  [docs/methods/06_mstamp-matrix-profile.md](docs/methods/06_mstamp-matrix-profile.md).

### The finding: the matching subset names the wrong channels at a discord

mSTAMP's k-dimensional subset is the k channels with the **smallest** distances, because the algorithm
was designed to find motifs. At a discord that names the channels which still look normal.

Measured on a four-channel record with the discord planted in `b` and `d`: `matching_subset(k=2)` returns
`{a, c}`, the two channels still repeating cleanly. Reading it as attribution reports the two **healthy**
channels as the culprits, with a perfectly sensible-looking discord peak beside it.

Hence two separate methods answering two different questions, and a test asserting both halves so the
distinction stays load-bearing.

The same asymmetry decides which k detects a partial fault: since P_k uses the k smallest distances, a
fault in two of four channels is invisible at k=2 and appears at k=d. **For fault detection, prefer a
high k**, which is the opposite of the natural intuition.

### Measured: the exclusion-zone folklore is wrong for this class of data

The standard telling is that without a trivial-match exclusion zone the profile collapses to zero and the
method reports an ordinary record no matter what is in it. A test was written to assert that and **it
failed**. Measured on a stochastic smooth signal at window 40, median profile:

| smoothing | lag-1 correlation | no zone | zone 0.5 | ratio |
|---|---|---|---|---|
| 30 | 0.959 | 0.291 | 0.322 | 0.91 |
| 100 | 0.987 | 0.323 | 0.374 | 0.87 |
| 300 | 0.998 | 0.296 | 0.334 | 0.89 |

About a tenth, not a collapse, and saturating by a zone of about four samples. z-normalisation is the
reason: it strips the mean and scale, leaving pure shape, and a window's shape changes measurably from
one sample to the next even when the raw signal barely moves.

Three framings were tried before this one and all would have passed for the wrong reason (a noise
discord, a high-frequency discord, and a single planted ramp). **For this class of data the exclusion
zone is not the critical parameter, the window length is.** The 0.5 default is kept because it costs
nothing and does matter for near-exactly-repeating motifs.

### Fixed

`exclusion=0` now means exactly "only the self-match is removed". It was floored at 1, which silently
applied a small exclusion zone even when none was asked for and made the parameter impossible to test at
its own boundary.

### Notes

Per-dimension distances are normalised before the sort, or a channel that happens to be noisier
contributes larger distances at every k and dominates subset selection through its variance rather than
through carrying the pattern.

The statistic sits at the window START. Placing it at the centre or the end shifts every reported onset
by a fixed amount, which is invisible in a plot and fatal in a delay metric.

A gapped series is rejected rather than filled: z-normalisation is undefined over a gap, and quietly
interpolating would invent the very shape the method is about to measure.

## [0.06.000] - 2026-08-07

PELT: exact retrospective segmentation, and a deliberate absence.

### Added

- `pelt.PELT`: optimal partitioning with the PELT pruning step, two cost functions (`meanvar` and
  `mean`), BIC penalty by default, `min_size` enforced, incomplete samples dropped with the mapping back
  to original indices preserved so a reported changepoint always refers to a real observation.
- `pelt.optimal_partition`: the unpruned quadratic dynamic programme. Not for production, since it
  returns the same answer more slowly. It exists so the exactness claim can be TESTED rather than
  trusted.
- `pelt.segmentation_error`: onset-time error **and** the changepoint count, as a pair, computed on the
  time axis rather than on sample indices.
- 25 further tests (200 total), and
  [docs/methods/05_pelt-retrospective-segmentation.md](docs/methods/05_pelt-retrospective-segmentation.md).

### Verified

The paper's contribution is that pruning costs nothing in accuracy, so the pruned and unpruned dynamic
programmes must return **identical** segmentations, not similar ones. Asserted over five random seeds,
both cost functions, and pure noise (the case where pruning is most aggressive and so most likely to
over-prune). A pruning bug would otherwise surface as slightly different changepoints, which reads as a
tuning difference rather than as an error.

### Measured, and left visible rather than tuned around

With BIC and a free variance, a record whose changes are purely in the mean comes back **over-segmented**:

| cost | penalty | changepoints on a 3-changepoint record |
|---|---|---|
| `meanvar` | BIC | 6 |
| `mean` | BIC | 3, all correct |
| `meanvar` | 60 | 3, all correct |

Nothing is broken. The cost function was asked the wrong question and the answer fits the data well while
meaning something other than what was wanted. Both configurations locate every true changepoint to within
3 samples, so an error-only report would rate them as equally good. A test pins all three rows.

### The deliberate absence

**PELT exposes no `detect` and returns no `Detection`.** It is retrospective: it sees the whole record,
including everything after the onset, before deciding where the onset was. Wiring it into an online
detection metric would score a method with access to the future, and it would score spectacularly:
detection delay near zero, false alarms near zero, because it is not detecting anything, it is describing
a record it has already read.

There is a test asserting the absence of `detect`, because a helpful future addition is exactly how the
leak would be reintroduced.

For the same reason `segmentation_error` returns the count with the error. Taking the nearest changepoint
is optimistic: a segmentation cutting the record every 5 samples scores an error of 0.0 with 60
changepoints. "Located the onset to within 3 samples" and "declared 47 other changepoints" are the same
measurement.

## [0.05.000] - 2026-08-07

Bayesian online changepoint detection: the run-length posterior, and two findings worth more than the
code.

### Added

- `bocpd.BOCPD`: the Adams and MacKay recursion with a swappable predictive model, returning the full
  run-length posterior in `meta` indexed by true run length, so a consumer can plot the distribution
  rather than a line.
- `bocpd.StudentTUPM`: Gaussian observations with unknown mean and variance under a Normal-Inverse-Gamma
  prior. Channels are conditionally independent given the run length, so every channel shares ONE
  run-length posterior, which is what a product needs: one onset time for the machine, not one per
  channel.
- 27 further tests (175 total), and
  [docs/methods/04_bayesian-online-changepoint-detection.md](docs/methods/04_bayesian-online-changepoint-detection.md).

### The finding: the paper's changepoint probability is a constant

The obvious reading of this algorithm is that it hands you `p(r_t = 0 | x_1:t)` and that you threshold
it. **Under a constant hazard that quantity is identically the hazard rate**, at every step, on every
dataset, because both branches of the recursion weight the same predictive and the hazard factors
straight out. The new observation is evaluated under the OLD run's parameters in both branches, so it
cannot yet distinguish them; the evidence arrives on the following steps.

This is not a defect in the algorithm and it was not a bug here. It is a trap in the reading, and it
produces a detector whose statistic is a flat line at 0.005 next to a perfectly sensible-looking
posterior plot. `statistic="changepoint"` is kept with a test asserting it equals the hazard at three
different rates, so the property is recorded rather than rediscovered.

The default statistic is now `short_run`, the posterior mass on runs of length at most `w`, which is the
informative version of the same idea: after a real change, mass collapses onto short runs and stays there
for several steps. Measured on a six-sigma step, above 0.9 in the twenty samples after the change against
a median below 0.2 before it. `surprise` (negative log evidence) is also available and is unbounded,
which often calibrates better because it does not saturate near 1.

### Fixed: run length is not array position

The exact posterior grows by one hypothesis per sample, so pruning is unavoidable. Once it happens the
survivors are no longer a contiguous 0, 1, 2 block, and array position stops equalling run length. The
first implementation read the shape parameters by position, so every hypothesis got the wrong prior
strength and the predictive stopped discriminating between hypotheses.

The symptom was a changepoint probability pinned at exactly the hazard rate, which is indistinguishable
from the genuine constant described above without checking the other statistics at the same time. Run
lengths are now carried explicitly, and the reported posterior is scattered to the true run length so a
pruned posterior still plots against a meaningful axis.

### Fixed: the warm-up

`p(r_t <= w)` is structurally 1 while `t <= w`, since the run cannot be longer than the record. Without a
warm-up the largest value of the statistic on ANY record is at index 0, so every detection lands on the
record beginning rather than on a fault. The first `warmup` samples (default 20) are now undefined.

### Notes

The derivation assumes the parameters before and after a changepoint are independent. That suits a step
change and suits a slow drift poorly: when the post-change state is nearly the pre-change state, evidence
for a changepoint at any instant stays weak no matter how far the drift eventually travels. A test pins
that behaviour. It matters for the product built on this package, because degradation onset is often
gradual and BOCPD is therefore not automatically the best rung for it.

The `lgamma` table is precomputed by run length rather than called per hypothesis per sample. That is a
performance decision with no accuracy cost, and it is checked against `math.lgamma` at a relative
tolerance of 1e-14 rather than assumed.

## [0.04.000] - 2026-08-07

Multivariate SPC: the rung that answers **which variables**, which is what turns an alarm into an action.

### Added

- `spc.PCAMonitor`: Hotelling T-squared on the retained principal subspace, SPE (the Q statistic) on the
  residual subspace, both classical control limits, and contribution plots for both statistics. PCA by
  SVD of the autoscaled baseline, so the component structure reports the machine rather than whichever
  channel is measured in the largest units.
- `normal_quantile` and `chi2_quantile`: Acklam's rational inverse normal CDF refined by one Halley step
  against `math.erfc`, and the Wilson-Hilferty chi-square transformation. Implemented rather than
  imported, because pulling in a full scientific stack so a monitoring library can look up one quantile
  would misplace the dependency boundary. Both are tested against published quantiles.
- 30 further tests (148 total), and
  [docs/methods/03_multivariate-spc-and-contributions.md](docs/methods/03_multivariate-spc-and-contributions.md).

### Measured

T-squared and SPE catch different faults, and neither subsumes the other. Constructed and asserted:

| fault | T-squared | SPE |
|---|---|---|
| driven far along a direction the machine already varies in | rises above 20x | barely moves |
| two channels that moved together stop doing so, each staying in its usual range | barely moves | rises above 20x |

The second row is the one usually lost: every individual channel stays inside its normal range, so only
the broken correlation gives it away. `detect` therefore always carries both arrays in `meta`, so a
result cannot be reported without the half that might have contradicted it.

The Jackson-Mudholkar SPE limit was checked against the property it claims (flagging about alpha of
held-out in-control samples) at two values of alpha, rather than against this implementation's output.

### Notes

**Smearing is demonstrated by a test, not only disclaimed.** With only channel `a` faulted, the healthy
channel `b`, which is merely correlated with `a`, is required to pick up more than 20% of `a`'s
contribution. If `b` ever stops smearing, that test fails and the documentation gets revisited rather
than quietly becoming untrue. A contribution narrows a candidate set; it does not identify a cause.

The SPE contribution decomposes SPE exactly, asserted to a relative tolerance of 1e-10. The T-squared
contribution takes an absolute value per variable, because a signed contribution can cancel across
components and report a variable as uninvolved while it carries the excursion in both directions.

A component whose eigenvalue is numerically zero is pushed into the residual subspace rather than
retained, since dividing by it in T-squared would turn rounding noise into an unbounded statistic.

The control limits exist because they are the published contribution of these papers and because they let
the implementation be checked against a stated property. **They are not what comparisons use.** Both
assume multivariate normality of the baseline, which real telemetry frequently violates, and two methods
at their own preferred limits say nothing about each other.

`statistic="combined"` is provided and is honestly the weakest of the three: it inherits the
approximations in both limits and hides which subspace moved, which is the one thing this rung exists to
tell you.

Hotelling (1947) was not verified against a primary source and is marked UNVERIFIED in the docs.

## [0.03.000] - 2026-08-07

The classical tier: the baseline this package claims to beat, implemented properly rather than left
convenient.

### Added

- `classical`: `Shewhart`, `CUSUM`, `EWMA` and `PageHinkley`. Each fits location and scale on a healthy
  baseline, returns a continuous statistic oriented so larger is more anomalous, and applies no threshold
  of its own.
- Multivariate reduction is the **maximum** across channels, not the mean. Averaging dilutes a
  single-channel fault by the channel count, so a twelve-channel machine would need a twelve-times-larger
  fault to reach the same statistic, and onset evidence is usually one channel moving.
- `Detection.channel_attribution(index)`: names the channels that drove the statistic, for any detector
  that records per-channel detail. A detector without that detail raises rather than inventing an even
  split.
- 47 further tests (118 total), and
  [docs/methods/02_classical-control-charts.md](docs/methods/02_classical-control-charts.md).

### Verified against published properties, not against this implementation's own output

- Shewhart at 3 sigma flags 0.20% to 0.35% of in-control samples over 200,000 draws, bracketing the
  published 0.27% under normality. A self-consistency check would not have caught a wrong
  standardisation; this does.
- CUSUM accumulates a sustained shift at a rate matching the closed form (delta minus k) to within 5%.
  Checking the rate rather than the direction is what catches a wrong reference value.
- EWMA at lambda = 1 reduces to Shewhart to floating-point tolerance, and at t = 1 the standardised
  statistic equals the standardised observation for **every** lambda, which pins the variance factor.
- Shewhart's known weakness is asserted rather than described: on a sustained half-sigma shift it stays
  under 6 while CUSUM exceeds 50.

### Notes

**The EWMA control limit is the exact time-varying one**, not the asymptote. The asymptotic form
overstates the spread of the early samples, so a chart using it from the start is weaker at the beginning
of every record by a factor of 1/sqrt(lambda(2-lambda)), which is 1.667 at lambda = 0.2. On a fleet where
records are short and restarts are common, that dead zone is a real loss of detection and is invisible
unless tested. The observation count is tracked **per channel**, so a channel that was NaN for a stretch
follows its own control limit rather than the wall-clock index.

**A correction to a claim made while writing this.** "Small lambda reacts more slowly" is true of the raw
smoothed value and false of the standardised statistic. For a sustained shift the statistic approaches
|z|/sqrt(lambda/(2-lambda)), which grows as lambda shrinks, so small lambda is uniformly stronger on the
small persistent shift degradation onset actually is. The genuine cost of small lambda is memory: after a
transient it holds the excursion far longer. Both halves are now tested against closed forms.

**Page-Hinkley is recorded as a CUSUM variant, not an independent rung**, with a test requiring
correlation above 0.98 between the two. Counting them as two independent classical baselines would
inflate any "we beat N baselines" claim.

Shewhart (1931) and Hinkley (1971) were not verified against primary sources in the research pass behind
this package and are marked UNVERIFIED in the docs.

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
