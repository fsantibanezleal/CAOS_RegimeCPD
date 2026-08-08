# 04 Bayesian online changepoint detection

Adams and MacKay (2007). The rung that returns a **run-length posterior** rather than a flag, and the one
whose modularity makes this package's central comparison fair.

## The recursion

Let $r_t$ be the run length, the time since the last changepoint, $H(r)$ the hazard function, and
$\pi_t^{(r)} = p(x_t \mid \text{run } r)$ the predictive probability of the new observation under each
run hypothesis. Then

$$p(r_t = r_{t-1}+1,\, x_{1:t}) = p(r_{t-1},\, x_{1:t-1})\,\pi_t^{(r)}\,(1 - H(r_{t-1}+1))$$

$$p(r_t = 0,\, x_{1:t}) = \sum_{r_{t-1}} p(r_{t-1},\, x_{1:t-1})\,\pi_t^{(r)}\,H(r_{t-1}+1)$$

normalised by the evidence. Inference is **exact for the most recent changepoint**, online, by message
passing. It is not a retrospective segmentation, which is why PELT is a separate rung rather than a
substitute.

## THE trap: the changepoint probability is a constant

The obvious reading is that this algorithm hands you $p(r_t = 0 \mid x_{1:t})$, the probability that a
changepoint just happened, and that you threshold it.

**Under a constant hazard that quantity is identically the hazard rate.** At every step. On every
dataset:

$$p(r_t = 0 \mid x_{1:t}) = \frac{\sum_r p(r_{t-1}=r)\,\pi_t^{(r)}\,H}{\sum_r p(r_{t-1}=r)\,\pi_t^{(r)}} = H$$

The hazard factors straight out, because **both branches of the recursion weight the same predictive**.
The new observation $x_t$ is evaluated under the OLD run's parameters in both branches, so it cannot yet
distinguish them. Evidence that the run is new accumulates on the *following* steps, as the long-run
hypotheses get crushed by the predictive.

This is not a defect in the algorithm and it was not a bug in this implementation. It is a trap in the
reading, and it produces a detector whose statistic is a flat line at 0.005 sitting next to a
perfectly sensible-looking posterior plot. `test_the_changepoint_probability_is_IDENTICALLY_the_hazard_rate`
asserts it at three hazard rates, so the property is recorded rather than rediscovered.

`statistic="changepoint"` is kept for exactly that reason.

## The statistics that do work

**`short_run` (the default)** reports $p(r_t \le w \mid x_{1:t})$, the posterior mass on short runs. It
is the informative version of the same idea: after a real change, mass collapses onto short runs and
stays there for several steps, which is where the evidence actually arrives. Measured on a step of six
sigma: above 0.9 in the twenty samples after the change, median below 0.2 in the stationary stretch
before it.

**`surprise`** reports $-\log p(x_t \mid x_{1:t-1})$. Unbounded, which often calibrates better precisely
because it does not saturate near 1 and therefore leaves an alarm-budget curve some resolution at the
sensitive end.

## The warm-up

$p(r_t \le w)$ is **structurally 1** while $t \le w$: the run cannot be longer than the record. Without a
warm-up, the largest value of the statistic on any record is at index 0, so every detection lands on the
record beginning rather than on a fault. This package hit that and fixed it.

BOCPD has a genuine warm-up that the control charts do not, because it learns its parameters online
rather than receiving them from a `fit` on a baseline. The first `warmup` samples (default 20) are
reported as undefined, and every detector in this package treats undefined as "cannot alarm".

## Why the modularity matters here specifically

Adams and MacKay factor the algorithm so the predictive model is swappable while the recursion stays
fixed. In this package that is not a convenience, it is what makes the central comparison fair.

Arm A runs this detector on raw channels; arm B runs it on within-regime residuals. Because the
observation model is the only thing that differs, the comparison is **the same recursion, the same
hazard function, a different observation model**. Nobody can object that two different pipelines were
compared and the better-engineered one won.

## The predictive model

Gaussian observations with unknown mean and variance under a Normal-Inverse-Gamma prior, whose conjugate
predictive is a Student-t:

$$p(x \mid \text{run } r) = \mathrm{St}\left(x;\ \mu_r,\ \frac{\beta_r(\kappa_r+1)}{\alpha_r\kappa_r},\ 2\alpha_r\right)$$

with the standard updates $\mu \leftarrow \frac{\kappa\mu + x}{\kappa+1}$, $\kappa \leftarrow \kappa+1$,
$\alpha \leftarrow \alpha + \tfrac12$, $\beta \leftarrow \beta + \frac{\kappa(x-\mu)^2}{2(\kappa+1)}$.

The heavy tails matter. A run with two observations behind it must not declare a changepoint every time a
slightly unusual sample arrives, and a Gaussian predictive with a plugged-in variance estimate does
exactly that while looking like sensitivity rather than like a modelling error.

Channels are **conditionally independent given the run length**, so the joint predictive is the product
across channels and every channel shares one run-length posterior. That is what the product needs: one
onset time for the machine, not one per channel. Cross-channel correlation is not modelled here, which is
what [the PCA rung](03_multivariate-spc-and-contributions.md) is for.

### A performance note that is not an approximation

The Student-t log density needs `lgamma`, which numpy does not vectorise. Naively that is one call per
run hypothesis per sample, tens of millions on a long record in pure Python.

It is avoidable exactly: after $r$ observations the shape is $\alpha_0 + r/2$ and the mean strength is
$\kappa_0 + r$, both determined by the run length alone. Every gamma term is precomputed once on a grid
of $r$ and looked up. `test_the_precomputed_gamma_terms_are_exact_not_an_approximation` checks the table
against `math.lgamma` at a relative tolerance of $10^{-14}$.

## A bug worth recording: run length is not array position

The exact posterior grows by one hypothesis per sample, so pruning is unavoidable on a long record. Once
it happens, **the survivors are no longer a contiguous $0, 1, 2, \dots$ block**, and array position stops
equalling run length.

The first implementation read the shape parameters by position. Every hypothesis then got the wrong prior
strength, the predictive stopped discriminating between hypotheses, and the symptom was a changepoint
probability pinned at exactly the hazard rate. That looked like a plausible constant rather than like a
bug, and it is only distinguishable from the genuine constant described above by checking the *other*
statistics at the same time.

Run lengths are now carried explicitly alongside the parameters, and the reported posterior is scattered
to the **true** run length rather than to array position, so a pruned posterior still plots against a
meaningful axis.

## The assumption, stated where it can be seen

The derivation assumes the parameters **before and after a changepoint are independent**. That is a real
modelling assumption, not a formality.

It suits a step change in a machine's behaviour. It suits a slow drift poorly: when the post-change state
is nearly the pre-change state, evidence for a changepoint at any particular instant stays weak no matter
how far the drift eventually travels. Expect BOCPD to be strong on abrupt onsets and unexceptional on
gradual ones, and to say so through a diffuse posterior rather than by being quietly wrong.
`test_the_posterior_is_wider_for_a_gradual_change_than_an_abrupt_one` pins that behaviour.

**This matters for the product built on this package.** Degradation onset is often gradual, so BOCPD is
not automatically the best rung for it. That is what the benchmark is for, and it is a live possibility
that CUSUM on a regime-conditioned residual beats BOCPD on raw channels for reasons that have nothing to
do with regimes.

## Reference

Adams, R. P., MacKay, D. J. C. "Bayesian Online Changepoint Detection."
[arXiv:0710.3742](https://arxiv.org/abs/0710.3742) [stat.ML], 19 October 2007. A preprint with no journal
version; cite it as such. Verified by fetching the record rather than from memory.
