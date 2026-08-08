# 07 Streaming drift and healthy-only novelty

Two families that both differ from the control charts, in opposite directions.

**Drift detectors** (ADWIN, KSWIN) carry no baseline at all. They compare the recent past against the less
recent past and adapt as the machine changes. A truck that is rebuilt, re-shod or reassigned to a
different pit has legitimately changed; a fixed baseline alarms on it forever, while a drift detector
reports the change once and accepts the new normal. That is either exactly right or exactly wrong
depending on whether the change was a repair or a fault, which is why running both families is
informative: they disagree about permanent changes, and the disagreement is the signal.

**Healthy-only novelty models** (isolation forest, one-class SVM, autoencoder) learn a rich model of
normal from a healthy baseline. No fault labels, which matters because a fleet has thousands of healthy
machine-years and a handful of labelled failures.

## ADWIN

Keep a window; wherever a split divides it into two halves whose means differ by more than a
Hoeffding-style bound, drop the older part.

$$\epsilon_{\mathrm{cut}} = \sqrt{\frac{2}{m}\sigma_W^2 \ln\frac{2}{\delta'}} + \frac{2}{3m}\ln\frac{2}{\delta'},
\qquad \frac{1}{m} = \frac{1}{n_0-1} + \frac{1}{n_1-1}$$

The harmonic $m$ matters: using $n_0$ and $n_1$ directly would understate the bound for a lopsided split,
which is exactly where a spurious cut is easiest to make.

ADWIN earns its place because it arrives with **bounds on both the false positive and the false negative
rate**, governed by one confidence parameter, rather than a threshold to sweep. That is worth having
beside conformal calibration as an independent route to the same property. Measured here: at
$\delta = 0.002$, at most 2 detections in 5000 stationary samples, and a three-sigma shift detected
within 60 samples.

### What is implemented, and what is not

This is the **direct** ADWIN, which examines every split of an explicit window. The paper's ADWIN2 keeps
an exponential-histogram summary so memory is logarithmic and the amortised cost per item is far lower.

**ADWIN2 is not implemented.** The direct version returns the same cuts, so nothing about detection
quality is lost, but the cost per item grows with the window and `max_window` bounds it. On records of a
few tens of thousands of samples this is fine; on a genuinely unbounded stream it is not.

The theorem statement and the ADWIN2 complexity bounds were **not verified against the primary source**
(the PDF fetch returned compressed binary), so the guarantee is described as the paper is generally
reported rather than quoted.

### ADWIN adapts, and it is blind to variance

Two tested properties that together define its role:

- After a shift is absorbed, a stationary stretch at the **new** level goes quiet again (at most 2
  detections in 1500 samples). It does not alarm forever.
- A **variance-only** change with the mean held fixed produces at most 3 detections in 1600 samples.
  ADWIN watches the mean; a change in spread alone is not its job.

That second point is the entire justification for KSWIN.

## KSWIN

Compare the most recent $r$ samples against an older reservoir with a two-sample Kolmogorov-Smirnov test.
Where ADWIN watches the mean, KSWIN compares whole **distributions**, so a fault that changes the spread
or the shape while leaving the mean alone is visible to it and invisible to ADWIN and to every
mean-centred control chart. Measured on a spread change from $\sigma = 1$ to $\sigma = 5$: KSWIN's
statistic rises above three times its quiet median while ADWIN fires at most 3 times.

The KS p-value is implemented by hand to keep the core numpy-only, using the Numerical Recipes form

$$Q(\lambda) = 2\sum_{j\ge1}(-1)^{j-1}e^{-2j^2\lambda^2}, \qquad
\lambda = \left(\sqrt{n_e} + 0.12 + \frac{0.11}{\sqrt{n_e}}\right)D$$

`scipy` evaluates $Q(\sqrt{n_e}D)$ without the correction term. The correction improves small-sample
accuracy, and with `recent = 50` that is exactly where KSWIN operates, so it is kept. The tests assert
$D$ matches scipy **exactly** (it is a definition, not an approximation) and that the p-value agrees to
within 5% wherever a p-value is actually read. In the deep tail the relative difference grows to about
50% at $p \approx 2\times10^{-9}$ and means nothing, because both numbers are approximations to something
around a billionth.

### Two measured limitations

**The per-step alpha is not the realised rate.** There is no multiple-testing correction, so on
stationary data the fraction of steps with $p < \alpha$ runs above nominal: about 1.2x at
$\alpha = 0.05$ and 1.7x at $\alpha = 0.005$, averaged over four records of 4000 samples.

**Single-record variance is enormous.** Consecutive windows overlap by all but one sample, so the tests
are strongly dependent and excursions arrive in **clusters**. One seed of that experiment gave 0.0005 and
another 0.008, a factor of sixteen at the same nominal $\alpha$.

Both are inherent to the method rather than defects here, and both are why the reported statistic is the
KS **distance** (bounded, already oriented, no flip needed) with the operating point coming from the
calibration layer. The clustering is also a direct argument for counting alarm events rather than
samples.

### ADWIN on an alarm-budget curve

ADWIN's statistic is a small integer count of channels flagging, so there are only $d+1$ distinct
thresholds and its budget curve has $d+1$ points where the others have hundreds. That is a real
limitation of putting a detector with a built-in decision rule onto a common budget, and it means
`delta` should be swept instead whenever ADWIN is being compared against anything.

## Healthy-only novelty models

### THE sign, which is where these go wrong

scikit-learn's novelty estimators return **higher scores for more normal** samples. Every detector in
this package returns higher for more anomalous. The flip happens once, at the source, in `detect`.

A missed flip produces a detector that scores exactly as badly as it should score well. Its alarm-budget
curve looks like a plausible weak method rather than like an error, and on a benchmark table it reads as
"this approach does not work for this problem". That is the worst way for a bug to fail, so there is an
orientation test for every model.

### The three models

| model | what it adds | source |
|---|---|---|
| Isolation forest | no distributional assumption, linear time | Liu, Ting and Zhou, *ICDM 2008*, pp. 413-422, doi:[10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17) |
| One-class SVM | a sharper boundary when normal operation is a compact curved region; superlinear training, hence `max_train` | Scholkopf et al. (2001), *Neural Computation* **13**(7):1443-1471. **UNVERIFIED** |
| Autoencoder | a nonlinear manifold in place of the linear subspace SPE uses; CUDA when available | this package |

The autoencoder is the learned analogue of the SPE statistic in
[03 Multivariate SPC](03_multivariate-spc-and-contributions.md), and comparing the two directly is worth
doing for exactly that reason. No sign flip: reconstruction error is already larger for more anomalous.

**A bottleneck at least as wide as the input is rejected.** Such a network learns the identity,
reconstructs a fault perfectly, and produces a detector that never fires, which looks like a
well-trained model. There is a test.

The one-class SVM subsamples a baseline larger than `max_train`, deterministically, and **records the
count it actually used** in `meta`. A model quietly trained on a fraction of the baseline is a different
model.

### Windowing, and where the statistic sits

`window > 1` stacks consecutive samples into one feature vector. Without it these models judge each
sample in isolation and cannot distinguish a channel sitting at an unusual value from a channel that got
there in an unusual way.

Measured: on a burst of fast oscillation with the **same amplitude** as the slow baseline (so the marginal
distribution is nearly untouched and only the trajectory is anomalous), the windowed model's contrast is
more than three times the per-sample model's.

The statistic sits at the window's **last** sample. These are online detectors: when the window closes,
its last sample is the newest one and the earliest point at which the evidence exists.
[mSTAMP](06_mstamp-matrix-profile.md) places its statistic at the window **start** for the opposite
reason, being retrospective over the whole record. The two conventions are deliberate rather than
inconsistent.

## Optional dependencies

The core is numpy and nothing else, and CI asserts it. A monitoring library that pulls in a full ML stack
to compute a CUSUM has misplaced its boundary.

```bash
pip install "regimecpd[learned]"   # scikit-learn: isolation forest, one-class SVM
pip install "regimecpd[deep]"      # PyTorch: the autoencoder
```

`deep` is separate from `learned` because PyTorch is a two-gigabyte install with a CUDA variant, and a
user who wants an isolation forest should not pay for it. Each import happens inside the method that
needs it and raises an error naming the extra, rather than a `ModuleNotFoundError` from three frames
down.
