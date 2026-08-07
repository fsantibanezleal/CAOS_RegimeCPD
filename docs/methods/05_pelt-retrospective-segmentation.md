# 05 PELT: retrospective segmentation

Killick, Fearnhead and Eckley (2012). Exact multiple-changepoint detection in linear expected time.

This rung answers a **different question** from every other rung here. BOCPD and the control charts ask
"is something changing now". PELT asks "looking back over the whole record, when did the changes actually
happen". Both are worth having and they are not substitutes.

## Why this module returns segments and NOT a `Detection`

PELT is retrospective. It sees the entire record, including everything after the onset, before deciding
where the onset was.

Feeding its output into an online detection metric would score a method that had access to the future,
and it would score **spectacularly**: detection delay near zero, false alarms near zero, because it is not
detecting anything, it is describing a record it has already read.

That is leakage, and it is easy to commit by accident because the output shape is so close to a
detection. So this module deliberately provides no `detect` and returns no `Detection`. There is a test,
`test_the_module_exposes_no_detect_method`, asserting the absence, because a helpful future addition is
exactly how the leak would get reintroduced.

## What it is for here

**Scoring onset-time error against synthetic ground truth.** When the true onset is known by
construction, PELT gives the best retrospective estimate of where it was, which is the right yardstick for
"how close did we get" as opposed to "how fast did we notice".

**Regime segmentation.** A record can be cut into operating regimes retrospectively, as an alternative to
clustering the context, and the two compared.

## The algorithm

Optimal partitioning minimises

$$\sum_{i=1}^{m+1} \mathcal{C}\left(y_{\tau_{i-1}+1:\tau_i}\right) + \beta m$$

over all segmentations, which is $O(n^2)$ by dynamic programming. PELT adds a pruning step: a candidate
$\tau$ can be discarded **permanently** once

$$F(\tau) + \mathcal{C}(y_{\tau+1:t}) + K > F(t)$$

where $K$ satisfies $\mathcal{C}(y_{u+1:v}) + \mathcal{C}(y_{v+1:w}) + K \le \mathcal{C}(y_{u+1:w})$. For
the costs used here $K = 0$.

### The result is exact, and that is the thing tested

The paper's contribution is that pruning costs **nothing in accuracy**. So `optimal_partition` implements
the unpruned quadratic version and the suite asserts the two return **identical** segmentations, over
five random seeds, both cost functions, and pure noise (the case where pruning is most aggressive and
therefore most likely to over-prune).

A pruning bug would otherwise surface as slightly different changepoints, which reads as a tuning
difference rather than as an error.

## Cost functions, and a documented over-segmentation

**`meanvar`** is the Gaussian negative log-likelihood with both mean and variance free per segment, up to
constants:

$$\mathcal{C} = n \log \hat{\sigma}^2$$

**`mean`** holds the variance at its whole-record estimate and lets only the mean move, giving the
residual sum of squares scaled by that variance.

### BIC with a free variance over-segments a pure level shift

On a record whose changes are purely in the mean, `meanvar` leaves the variance free, so a stretch of
ordinary noise can be explained as a short segment with a different variance. With the general-purpose
BIC penalty that is cheap enough to be worth doing.

Measured on a four-level record with three true changepoints:

| cost | penalty | changepoints returned |
|---|---|---|
| `meanvar` | BIC | **6** (the 3 true ones plus 3 spurious) |
| `mean` | BIC | **3**, all correct |
| `meanvar` | 60 | **3**, all correct |

**Nothing is broken here.** The cost function was asked the wrong question, and the answer fits the data
well while meaning something other than what was wanted. Both configurations *locate* every true
changepoint to within 3 samples, so an error-only report would rate them as equally good.

The test `test_bic_with_a_free_variance_OVER_segments_a_pure_level_shift` pins all three rows, so the
property stays visible rather than being quietly tuned around.

### The penalty is not a nuisance parameter

`penalty` sets how many changepoints come back, and BIC is a general-purpose default that knows nothing
about the machine. Sweep it and look at the curve before trusting a count. The suite checks that the
count is monotone non-increasing in the penalty and that the penalty actually bites.

## `segmentation_error` returns the count with the error, always

```python
error, n_changepoints = segmentation_error(changepoints, t, true_onset)
```

`error` is the absolute time difference to the **nearest** changepoint, which is optimistic on purpose
and therefore dangerous alone. A segmentation cutting the record into fifty pieces will almost always
have one landing near the true onset:

| segmentation | error | count |
|---|---|---|
| three sensible changepoints | 3.0 | 3 |
| a changepoint every 5 samples | **0.0** | **60** |

"Located the onset to within 3 samples" and "declared 47 other changepoints" are the same measurement.
Quoting the first without the second is how this metric gets gamed, including accidentally, so the
function returns both and there is a test named for the shotgun case.

The error is computed on the **time axis**, not on sample indices. Irregular sampling is the norm on
fleet telemetry, so an error in samples is not an error in time.

## Incomplete samples

Rows that are not complete are **dropped** rather than filled, and the mapping back to original indices
is kept, so a reported changepoint always refers to a real observation. `test_incomplete_samples_are_dropped_and_indices_still_refer_to_real_samples`
checks that a returned index never lands on a dropped sample.

## Reference

Killick, R., Fearnhead, P., Eckley, I. A. "Optimal Detection of Changepoints With a Linear Computational
Cost." *Journal of the American Statistical Association* **107**(500), 1590-1598 (2012).
doi:[10.1080/01621459.2012.737745](https://doi.org/10.1080/01621459.2012.737745)
