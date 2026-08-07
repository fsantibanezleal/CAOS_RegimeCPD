# regimecpd

[![CI](https://github.com/fsantibanezleal/CAOS_RegimeCPD/actions/workflows/ci.yml/badge.svg)](https://github.com/fsantibanezleal/CAOS_RegimeCPD/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/regimecpd.svg)](https://pypi.org/project/regimecpd/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Regime-conditional change-point detection for machines whose operating context moves.**

```bash
pip install regimecpd
```

## The problem, in one paragraph

On a machine whose load varies, every measured channel varies with it. A haul truck climbing a ramp
loaded shows rising strut pressure, rising fuel rate and rising temperatures all at once, and shows the
reverse coming back empty. Run a change detector on those raw channels and it fires on the ramp. It has
detected the hill, not a fault. So the pipeline is not "detect":

```
raw channels -> regime segmentation -> within-regime residual -> change point -> onset
```

## What makes this package different from a bag of detectors

It is built to answer whether the regime stage is worth anything, rather than to assume it is.

- **Both arms, one detector.** The identical detector runs on raw channels and on within-regime
  residuals. `Residual.as_series()` hands the detector the same type either way, so it has no way to
  behave differently between arms.
- **Thresholds are never chosen by the detector.** Every method returns a continuous statistic. The
  calibration layer picks the threshold against a stated false-alarm budget, so two methods are always
  compared at the same operating point rather than at each one's favourite.
- **False alarms are counted the way an operator experiences them.** Excursions, not samples above a
  line, and per unit of TIME rather than per sample.
- **Intervals are bootstrapped over units, not samples.** Samples inside one machine are dependent;
  resampling them produces intervals that are far too narrow.
- **A negative result is a supported outcome.** If conditioning on regime does not reduce the false-alarm
  rate on your data, the package reports that with the same intervals it would report a win with.

Nothing here is truck-specific. The reference validation runs on turbofan data, which is the evidence for
that claim rather than an assertion of it.

## Status

**v0.01.000.** The data contract and the measurement layer are complete and tested. The detector ladder
is being built on top of them, one rung per release, each with its own tests and its own documentation.
See the [CHANGELOG](CHANGELOG.md) for what exists today and the
[wiki](docs/README.md) for the theory.

## Quick look

```python
import numpy as np
import regimecpd as rc

# A detector's output is a continuous statistic, never a flag.
t = np.arange(1000.0)
statistic = np.abs(np.random.default_rng(0).normal(size=1000))
statistic[700:] += 3.0                      # something starts at t = 700

det = rc.Detection(t, statistic, method="demo")
faulty = rc.UnitOutcome(det, onset_t=700.0, unit_id="truck-01")
healthy = [rc.UnitOutcome(rc.Detection(t, np.abs(np.random.default_rng(i).normal(size=1000)), "demo"),
                          onset_t=None, unit_id=f"truck-{i:02d}") for i in range(2, 12)]
fleet = [faulty, *healthy]

# Pick the threshold from a stated budget rather than by eye.
th = rc.threshold_for_budget(fleet, target_rate=1e-3)
score = rc.score_fleet(fleet, th)
print(f"{score.per(730.0):.2f} false alarms per unit-month, "
      f"detected {score.n_detected}/{score.n_faulty} with median delay {score.median_delay}")

# Every reported number carries an interval, resampled over UNITS.
point, lo, hi = rc.bootstrap_ci(fleet, lambda o: rc.score_fleet(o, th).false_alarms_per_unit_time)
print(f"rate {point:.5f} (95% CI {lo:.5f} to {hi:.5f})")
```

## Documentation

The [`docs/`](docs/README.md) wiki carries the theory, the equations, the primary references with real
DOIs, and the honest limits of each method:

- [`docs/methods/`](docs/methods.md) one deep page per rung of the ladder
- [`docs/architecture/`](docs/architecture.md) the data contract, and how the two arms stay comparable
- [`docs/guides/`](docs/guides.md) install, use it on your own data, and how to read the results

## Licence

MIT. See [LICENSE](LICENSE).
