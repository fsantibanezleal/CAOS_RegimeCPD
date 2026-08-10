"""Tests for the degenerate-scale guard.

This is a regression suite for a real defect found on real data, so the numbers used here are the ones
that actually occurred rather than invented ones. A channel constant to within floating-point noise had a
strictly positive standard deviation, passed the old `spread > 0` guard, and produced z-scores around
1e14.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimecpd import Series
from regimecpd.classical import CUSUM, Shewhart
from regimecpd.scaling import ATOL, RTOL, degenerate_mask, robust_scale
from regimecpd.spc import PCAMonitor


def series(values, names=None):
    v = np.asarray(values, dtype=float)
    v = v.reshape(-1, 1) if v.ndim == 1 else v
    names = names or tuple("abcdefgh"[: v.shape[1]])
    return Series(np.arange(len(v), dtype=float), v, names)


def cmapss_like_channel(n=200, value=21.61, step_every=0):
    """A channel exactly as C-MAPSS sensor_06 appears.

    Over a baseline window it holds one repeated value and its standard deviation is pure representation
    noise, around 7e-15 against a mean of 21.61, a relative spread near 3e-16.

    ``step_every`` reproduces the other half of the real mechanism: later in the record the sensor does
    occasionally report the adjacent quantisation level (21.60 rather than 21.61). That step is utterly
    ordinary in raw units, and it is what the tiny baseline spread turns into a z-score of 1e12. Both
    halves are needed: a constant baseline alone does not explode, and a normal step alone does not
    either.
    """
    x = np.full(n, value)
    x[::7] = np.nextafter(value, value + 1)     # ULP-level differences, as float storage produces
    if step_every:
        x[::step_every] = value - 0.01          # a real quantisation step, entirely unremarkable
    return x


class TestTheRule:
    def test_a_channel_constant_to_floating_point_noise_is_flagged(self):
        x = cmapss_like_channel()
        sd, mean = float(x.std()), float(x.mean())
        assert 0 < sd < 1e-12, f"setup: expected a tiny but positive sd, got {sd}"
        assert degenerate_mask(np.array([sd]), np.array([mean]))[0]
        assert robust_scale(np.array([sd]), np.array([mean]))[0] == 1.0

    def test_a_genuinely_quantised_channel_is_NOT_flagged(self):
        # A sensor reporting to two decimals has a relative spread around 1e-4, far above the 1e-8 floor.
        # The guard must not throw away real, coarsely quantised measurements.
        rng = np.random.default_rng(0)
        x = np.round(21.61 + rng.normal(0, 0.02, 500), 2)
        sd, mean = float(x.std()), float(x.mean())
        assert not degenerate_mask(np.array([sd]), np.array([mean]))[0]
        assert robust_scale(np.array([sd]), np.array([mean]))[0] == pytest.approx(sd)

    def test_an_exactly_constant_channel_is_flagged(self):
        assert degenerate_mask(np.array([0.0]), np.array([5.0]))[0]

    def test_the_floor_scales_with_the_channel_magnitude(self):
        # A spread of 1e-6 is real on a channel centred at 0.001 and is noise on one centred at 1e6.
        assert not degenerate_mask(np.array([1e-6]), np.array([1e-3]))[0]
        assert degenerate_mask(np.array([1e-6]), np.array([1e6]))[0]

    def test_a_channel_centred_at_zero_falls_back_to_the_absolute_floor(self):
        assert degenerate_mask(np.array([1e-15]), np.array([0.0]))[0]
        assert not degenerate_mask(np.array([1e-6]), np.array([0.0]))[0]

    def test_non_finite_spread_is_flagged(self):
        assert degenerate_mask(np.array([np.nan]), np.array([1.0]))[0]
        assert degenerate_mask(np.array([np.inf]), np.array([1.0]))[0]

    def test_the_tolerances_sit_between_representation_noise_and_real_quantisation(self):
        assert ATOL < RTOL
        assert 1e-12 <= RTOL <= 1e-6, "if this moves, re-read the reasoning in regimecpd.scaling"


class TestTheDefectItPrevents:
    def test_a_near_constant_channel_no_longer_produces_an_astronomical_statistic(self):
        # THE regression test, with the real failure reproduced. Before the guard, a CUSUM on this
        # channel accumulated past 1e12 over a healthy stretch.
        base = series(cmapss_like_channel(120))
        rest = series(cmapss_like_channel(400, step_every=5))

        out = CUSUM(k=0.5).fit(base).detect(rest)
        peak = float(np.nanmax(out.statistic))
        assert peak < 10.0, f"a channel that does not move must not accumulate; got {peak:.3g}"

    def test_the_same_channel_would_have_exploded_under_the_old_guard(self):
        # Computed explicitly, so this test fails if the guard ever stops being the thing that saves it.
        x = cmapss_like_channel(120)
        sd_old = x.std() if x.std() > 0 else 1.0          # the old `spread > 0` rule
        later = cmapss_like_channel(400, step_every=5)
        z_old = abs(later[0] - x.mean()) / sd_old
        assert z_old > 1e10, (
            f"if this no longer explodes, the test data stopped reproducing the defect: {z_old:.3g}"
        )

    def test_one_degenerate_channel_does_not_hijack_the_across_channel_maximum(self):
        # The mechanism by which this destroyed a fleet benchmark: the statistic is a max across
        # channels, so a single exploding channel becomes the whole statistic for every unit.
        rng = np.random.default_rng(1)
        good = rng.normal(size=600)
        flat = cmapss_like_channel(600, step_every=5)
        base = series(np.column_stack([good[:200], flat[:200]]), ("good", "flat"))
        test = np.column_stack([good[200:], flat[200:]])
        test[300:, 0] += 4.0                              # a real fault, in the good channel

        out = CUSUM(k=0.5).fit(base).detect(series(test, ("good", "flat")))
        att = out.channel_attribution(int(np.nanargmax(out.statistic)))
        assert att.top(1) == ("good",), f"the degenerate channel hijacked attribution: {att.ranked()}"

    def test_shewhart_is_protected_too(self):
        base = series(cmapss_like_channel(120))
        out = Shewhart().fit(base).detect(series(cmapss_like_channel(200, step_every=5)))
        assert float(np.nanmax(out.statistic)) < 10.0

    def test_pca_autoscaling_is_protected_too(self):
        # A degenerate channel in a PCA would otherwise dominate every component after autoscaling.
        rng = np.random.default_rng(2)
        f = rng.normal(size=800)
        x = np.column_stack([f + 0.1 * rng.normal(size=800),
                             f + 0.1 * rng.normal(size=800),
                             cmapss_like_channel(800, step_every=5)])
        base = series(x, ("a", "b", "flat"))
        monitor = PCAMonitor(n_components=1).fit(base)
        loading = np.abs(monitor.loadings_[:, 0])
        assert loading[2] < 0.3 * max(loading[0], loading[1]), (
            f"the degenerate channel dominated the first component: {loading}"
        )
        assert np.all(np.isfinite(monitor.t2(base)))
        assert np.all(np.isfinite(monitor.spe(base)))
