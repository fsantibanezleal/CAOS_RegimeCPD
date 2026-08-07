# Methods

One deep page per rung of the ladder: theory, equations, the primary source with a real DOI, what the
method uniquely contributes, and its honest limitation.

**A rung appears in this table only when it is implemented, tested and documented.** Listing a method the
package does not implement is a defect, and so is implementing one whose theory was written from memory
rather than transcribed from its source.

## The ladder

### Classical, the baseline

The baseline this package is built to be measured against, so it is implemented properly. A strawman
baseline invalidates every comparison made on top of it.

| method | primary source | status |
|---|---|---|
| Shewhart / fixed threshold | Shewhart (1931) | **implemented** (v0.03.000) |
| CUSUM | Page (1954), *Biometrika* **41**(1-2):100-115, doi:[10.1093/biomet/41.1-2.100](https://doi.org/10.1093/biomet/41.1-2.100) | **implemented** (v0.03.000) |
| EWMA | Roberts (1959), *Technometrics* **1**(3):239-250, doi:[10.1080/00401706.1959.10489860](https://doi.org/10.1080/00401706.1959.10489860) | **implemented** (v0.03.000) |
| Page-Hinkley | Hinkley (1971) | **implemented** (v0.03.000) |
| Hotelling $T^2$ and SPE / $Q$ on PCA | Jackson and Mudholkar (1979), *Technometrics* **21**(3):341-349, doi:[10.1080/00401706.1979.10489779](https://doi.org/10.1080/00401706.1979.10489779) | **implemented** (v0.04.000) |
| Contribution plots | Kourti and MacGregor (1996), *JQT* **28**(4):409-428, doi:[10.1080/00224065.1996.11979699](https://doi.org/10.1080/00224065.1996.11979699); Westerhuis, Gurden and Smilde (2000), doi:[10.1016/S0169-7439(00)00062-9](https://doi.org/10.1016/S0169-7439(00)00062-9) | **implemented** (v0.04.000) |

### State of the art

| method | primary source | status |
|---|---|---|
| BOCPD (run-length posterior) | Adams and MacKay (2007), [arXiv:0710.3742](https://arxiv.org/abs/0710.3742) | **implemented** (v0.05.000) |
| PELT | Killick, Fearnhead and Eckley (2012), *JASA* **107**(500):1590-1598, doi:[10.1080/01621459.2012.737745](https://doi.org/10.1080/01621459.2012.737745) | planned |
| mSTAMP (window and dimension subset) | Yeh, Kavantzas and Keogh (2017), *ICDM 2017*, pp. 565-574 | planned |
| ADWIN | Bifet and Gavalda (2007), *SDM 2007*, pp. 443-448 | planned |
| Isolation Forest (healthy-only) | Liu, Ting and Zhou (2008), *ICDM 2008*, pp. 413-422, doi:[10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17) | planned |

### Beyond

| method | primary source | status |
|---|---|---|
| Regime conditioning (segmentation + residual) | [01 Regime conditioning](methods/01_regime-conditioning.md) | **implemented** (v0.02.000) |
| Regime-conditional detection, measured against raw | this package | planned |
| Conformal calibration of the alarm rate | Xu and Xie (2021), *ICML*, PMLR **139**:11559-11569; Gibbs and Candes (2021), *NeurIPS* **34**:1660-1672 | planned |

## Currently implemented

| release | what landed |
|---|---|
| v0.01.000 | The data contract and the measurement layer. Built before any detector on purpose: how a detector is SCORED is what every later claim rests on, and building it first already caught a defect that would have corrupted every benchmark run through it (see [the alarm-budget trap](architecture/02_measuring-a-detector.md#the-event-counted-rate-is-not-monotone-in-the-threshold)) |
| v0.02.000 | [Regime conditioning](methods/01_regime-conditioning.md): observed and discovered segmentation, the novelty radius, both residual models, and `make_arms` |
| v0.03.000 | [Classical control charts](methods/02_classical-control-charts.md): Shewhart, CUSUM, EWMA with the exact time-varying limit, Page-Hinkley, plus per-channel attribution |
| v0.04.000 | [Multivariate SPC and contributions](methods/03_multivariate-spc-and-contributions.md): Hotelling T-squared, SPE/Q, both classical control limits, and contribution plots for both statistics |
| v0.05.000 | [BOCPD](methods/04_bayesian-online-changepoint-detection.md): the run-length posterior, a swappable Student-t predictive, and the recorded finding that the paper's changepoint probability is identically the hazard rate |

The classical tier is complete and the first SOTA rung has landed. **No controlled
raw-against-residual comparison has been run**: the releases so far establish only that each method
behaves the way its source says it behaves. Each remaining rung lands in its own release with its own
tests and its own page here.

## The comparison design, stated once

Every rung is evaluated the same way, and the design is fixed in advance so it cannot drift toward a
flattering configuration:

1. **One detector at a time**, held identical across arms. Not "our pipeline against theirs".
2. **Arm A** on raw channels, **arm B** on within-regime residuals. Same detector, same
   hyperparameters, same calibration split, same alarm budget.
3. **The budget is set by conformal calibration**, not by sweeping thresholds until something looks good.
4. **Both directions reported**: false alarms per unit-month at fixed detection delay, and detection
   delay at fixed false alarms per unit-month.
5. **Every number carries an interval**, bootstrapped over units.
6. **A negative result ships as the finding**, with the same intervals and the same prominence.
