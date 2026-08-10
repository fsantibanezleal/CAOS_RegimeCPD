# 03 Multivariate SPC and contributions

Hotelling $T^2$, SPE (the $Q$ statistic) and contribution plots on a PCA fitted to a healthy baseline.

This is the rung that answers **which variables**, which is what turns an alarm into an action. A
maintenance planner cannot dispatch against "the machine is anomalous". They can dispatch against "strut
pressure and brake temperature are the channels carrying the excursion".

## Two subspaces, two different faults

Autoscale the baseline, take its principal components, retain $a$ of them. That splits the measurement
space in two, and the split is the whole reason this rung exists rather than being a fancier threshold.

**$T^2$ lives in the retained subspace.** Is the machine in an unusual place along the directions it
normally varies in? In-model, but extreme.

$$T^2_t = \sum_{i=1}^{a} \frac{t_{t,i}^2}{\lambda_i}$$

**SPE lives in the residual subspace.** Has the machine left the model? A correlation that held
throughout the baseline has broken.

$$\mathrm{SPE}_t = \lVert x_t - \hat{x}_t \rVert^2 = \lVert (I - PP^{\top})x_t \rVert^2$$

### Neither subsumes the other, and that is tested

| fault | $T^2$ | SPE |
|---|---|---|
| A sensor driven far along a direction the machine already varies in | **rises above 20x** | barely moves |
| Two channels that used to move together stop doing so, each staying inside its usual range | barely moves | **rises above 20x** |

`test_an_in_model_excursion_raises_t2_and_leaves_spe_flat` and
`test_a_broken_correlation_raises_spe_and_leaves_t2_flat` construct exactly those two faults and require
each statistic to catch its own and miss the other. Reporting only one of the two is the common mistake,
and the second row is the one usually lost: every individual channel stays inside its normal range, so
nothing about the marginal distributions is unusual, and only the broken correlation gives it away.

`detect` therefore always carries **both** arrays in `Detection.meta`, so a result can never be reported
without the half that might have contradicted it.

## Autoscaling, not covariance PCA

The baseline is centred and divided by its per-channel standard deviation before decomposition. Without
that, the component structure is decided by whichever channel happens to be measured in the largest
units, which is a property of the instrumentation rather than of the machine.
`test_autoscaling_stops_the_largest_unit_from_deciding_the_components` puts one channel in units a
thousand times larger than its correlated partner and requires the first component to load on both.

The decomposition uses an SVD of the centred data rather than an eigendecomposition of the covariance
matrix. Same answer, better conditioned, and it avoids forming a matrix whose condition number is the
square of the data's.

### Directions with no variance go into the residual subspace

A component whose eigenvalue is numerically zero is a direction the baseline never moved in. Dividing by
it in $T^2$ would turn rounding noise into an unbounded statistic. Those directions are pushed into the
residual subspace instead, where SPE handles them correctly, and the retained count drops accordingly.
A baseline with no varying direction at all raises rather than returning a monitor that reports
infinities.

## Contributions

`kind="spe"` gives the squared residual per variable, which decomposes SPE **exactly**:

$$c_j = (z_{t,j} - \hat{z}_{t,j})^2, \qquad \sum_j c_j = \mathrm{SPE}_t$$

That identity is asserted to a relative tolerance of $10^{-10}$, so it either holds or the implementation
is wrong.

`kind="t2"` gives the Kourti and MacGregor form, summed over the retained components:

$$c_j = \left| \sum_{i=1}^{a} \frac{t_{t,i}}{\lambda_i} p_{ji} z_{t,j} \right|$$

The absolute value is taken deliberately. A signed contribution can cancel across components and report a
variable as uninvolved when it is carrying the excursion in both directions.

### Smearing, demonstrated rather than only disclaimed

Contribution plots **smear**: a single faulty variable spreads contribution onto every variable
correlated with it, because the statistic is computed in a rotated space where the fault is no longer
aligned with one axis.

`test_contributions_smear_onto_a_correlated_healthy_channel` faults only channel `a` and then requires
channel `b`, which is healthy and merely correlated with `a`, to pick up more than 20% of `a`'s
contribution. The test exists so that the caveat cannot quietly stop being true: if `b` ever stops
smearing, the test fails and the documentation gets revisited.

So a high contribution means "this variable is implicated", never "this variable is the cause". The
honest use is to narrow a candidate set from twelve channels to three, and the honest cross-check is an
attribution obtained by a completely unrelated route.

## The control limits, and why they are not what comparisons use

Both classical limits are implemented, because they are the published contribution of these papers and
because they let the implementation be checked against a **stated property** rather than against its own
output.

**SPE**, by Jackson and Mudholkar (1979):

$$Q_\alpha = \theta_1 \left[ \frac{c_\alpha\sqrt{2\theta_2 h_0^2}}{\theta_1} + 1
+ \frac{\theta_2 h_0(h_0-1)}{\theta_1^2} \right]^{1/h_0}$$

with $\theta_i = \sum_{j>a}\lambda_j^i$ and $h_0 = 1 - \frac{2\theta_1\theta_3}{3\theta_2^2}$.
`test_the_spe_limit_flags_about_alpha_of_in_control_samples` checks it against held-out in-control data
at two values of $\alpha$.

**$T^2$**, by the chi-square approximation $T^2 \sim \chi^2_a$. Exact for a known covariance; here the
covariance is estimated, so the $F$-distribution form is the stricter statement. With a baseline of a few
thousand samples the difference is small, and the approximation is named rather than hidden.

Both quantiles are computed without scipy: an inverse normal CDF by Acklam's rational approximation
refined with one Halley step against `math.erfc`, and a chi-square quantile by the Wilson-Hilferty
transformation. Pulling in a full scientific stack so that a monitoring library can look up one quantile
would misplace the dependency boundary. Both are tested against **published** quantiles, and the normal
quantile is additionally checked against `erfc` across six orders of magnitude.

**These limits are not what this package uses to compare methods.** Comparisons run through the
calibration layer at a common false-alarm budget, because two methods at their own preferred limits tell
you nothing about each other. Both limits also assume multivariate normality of the baseline, which on
real telemetry is wrong often enough that the empirical calibration is the more trustworthy route even
setting the comparison argument aside.

## The `combined` statistic, and why it is the weakest option

`statistic="combined"` reports $\max(T^2/T^2_\alpha,\; \mathrm{SPE}/Q_\alpha)$. It is a convenience for a
single-number workflow and it is honestly the worst of the three: it inherits the approximations in both
limits and it hides which subspace moved, which is the one piece of information this rung exists to
provide. Prefer running the two separately and reporting both.

## References

- Jackson, J. E., Mudholkar, G. S. "Control Procedures for Residuals Associated With Principal Component
  Analysis." *Technometrics* **21**(3), 341-349 (1979).
  doi:[10.1080/00401706.1979.10489779](https://doi.org/10.1080/00401706.1979.10489779). Introduces the
  control chart for the sum of squared residuals $Q$ alongside $T^2$ on the retained components.
- Kourti, T., MacGregor, J. F. "Multivariate SPC Methods for Process and Product Monitoring."
  *Journal of Quality Technology* **28**(4), 409-428 (1996).
  doi:[10.1080/00224065.1996.11979699](https://doi.org/10.1080/00224065.1996.11979699). The contribution
  formula used for $T^2$.
- Westerhuis, J. A., Gurden, S. P., Smilde, A. K. "Generalized contribution plots in multivariate
  statistical process monitoring." *Chemometrics and Intelligent Laboratory Systems* **51**(1), 95-114
  (2000). doi:[10.1016/S0169-7439(00)00062-9](https://doi.org/10.1016/S0169-7439(00)00062-9). Covers
  contributions for both statistics and treats the smearing properly rather than presenting contributions
  as diagnosis.
- Hotelling, Harold. "Multivariate Quality Control, Illustrated by the Air Testing of Sample
  Bombsights." Chapter 3 in *Selected Techniques of Statistical Analysis for Scientific and Industrial
  Research and Production and Management Engineering*, by the Statistical Research Group, Columbia
  University; edited by Churchill Eisenhart, Millard W. Hastay and W. Allen Wallis. New York and London:
  McGraw-Hill, 1947, pp. 111-184. **Verified** from the scanned title page and table of contents
  (Internet Archive `in.ernet.dli.2015.264585`) plus K10plus MARC records.

  Three things this package had wrong, or would have got wrong by copying the usual short citation.
  The chapter title is longer than the running head everyone quotes. The book title is
  *Selected Techniques of Statistical Analysis*, not the *Techniques of Statistical Analysis* that the
  copyright page verso gives, which is why both forms circulate. And the chapter is **not the origin of
  $T^2$**: Hotelling attributes the statistic to the generalized Student ratio on p. 114 of the chapter
  itself. What the chapter contributes is the quality-control application, which is what this package
  uses it for.
