# 01 Install and quickstart

## Install

```bash
pip install regimecpd
```

The core needs **numpy only**. The healthy-only novelty models (isolation forest, one-class SVM) need
scikit-learn, which is an extra rather than a core dependency:

```bash
pip install "regimecpd[learned]"
```

CI asserts on every run that the core install really does declare numpy and nothing else, and that the
package imports and runs under that install. The claim is checked, not merely written down.

## The minimal end-to-end path

The package separates three things that are usually tangled together: producing a statistic, choosing a
threshold, and scoring the result. This is what that looks like end to end.

```python
import numpy as np
import regimecpd as rc

rng = np.random.default_rng(0)
t = np.arange(3000.0)                       # hours, in this example

# Five machines that develop a fault at t = 2200, and fifteen that never do.
faulty = []
for k in range(5):
    stat = np.abs(rng.normal(size=3000))
    stat[2200:] += 4.0
    faulty.append(rc.UnitOutcome(rc.Detection(t, stat, "demo"), onset_t=2200.0, unit_id=f"f{k}"))

healthy = [rc.UnitOutcome(rc.Detection(t, np.abs(rng.normal(size=3000)), "demo"), None, f"h{k}")
           for k in range(15)]

fleet = faulty + healthy
```

### Choose the threshold from a budget, not by eye

```python
th = rc.threshold_for_budget(fleet, target_rate=1e-3)   # per hour, matching the units of t
if th is None:
    raise SystemExit("this detector cannot operate at that budget on this data")
```

`None` is a real answer. A detector can be unable to meet a budget at all, and that is a finding about
the detector rather than an error to work around.

### Score the fleet

```python
score = rc.score_fleet(fleet, th)
print(f"{score.per(730.0):.3f} false alarms per machine-month")   # 730 hours in a month
print(f"detected {score.n_detected}/{score.n_faulty}, median delay {score.median_delay} h")
```

### Never report a point number alone

```python
point, lo, hi = rc.bootstrap_ci(
    fleet,
    lambda o: rc.score_fleet(o, th).false_alarms_per_unit_time,
    n_boot=1000,
)
print(f"rate {point:.5f} (95% CI {lo:.5f} to {hi:.5f})")
```

The bootstrap resamples **machines**, not samples. Consecutive readings from one machine are nearly the
same reading, so a sample-level interval would be far too narrow.

## Units are yours to choose

Everything is expressed per unit of `t`. If `t` is in hours, rates are per hour and `per(730.0)` gives
per month. If `t` is in operating cycles, rates are per cycle. The package never assumes a sampling
interval, which is what lets it handle irregular and gappy telemetry without a resampling step that would
invent data.

## Comparing two arms

This is what the package is for. Note that the detector call is identical in both lines:

```python no-run
raw_detections = [detector(series) for series in fleet_series]
res_detections = [detector(residual.as_series()) for residual in fleet_residuals]
```

`Residual.as_series()` hands the detector a plain `Series`, so it has no way to know which arm it is in.
Score both at the **same** false-alarm budget and compare the detection delays; or fix the delay and
compare the rates. Reporting only one direction is how this kind of result gets quietly gamed, so report
both.

## Development setup

```bash
git clone https://github.com/fsantibanezleal/CAOS_RegimeCPD.git
cd CAOS_RegimeCPD
python -m venv .venv
.venv/bin/pip install -e ".[dev]"      # .venv\Scripts\pip on Windows
.venv/bin/pytest
.venv/bin/ruff check regimecpd tests
```

Dependencies go in the project-local virtualenv, never globally.

Branch flow is `task/<slug>` into `develop`, then `develop` into `main` by pull request. `main` is the
released branch and is never committed to directly. See [CONTRIBUTING.md](../../CONTRIBUTING.md).
