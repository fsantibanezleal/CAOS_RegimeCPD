"""Tests for the streaming drift detectors.

The KS p-value is checked against `scipy.stats.ks_2samp` where scipy is installed, because an asymptotic
series is the kind of code that returns plausible numbers while being subtly wrong, and this package
implements it by hand to keep the core numpy-only.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimecpd import Series
from regimecpd.drift import ADWIN, KSWIN, ks_two_sample_pvalue


def series(values, names=None):
    v = np.asarray(values, dtype=float)
    v = v.reshape(-1, 1) if v.ndim == 1 else v
    names = names or tuple("abcdefgh"[: v.shape[1]])
    return Series(np.arange(len(v), dtype=float), v, names)


class TestADWIN:
    def test_detects_a_mean_shift(self):
        rng = np.random.default_rng(0)
        det = ADWIN(delta=0.002)
        fired = [i for i, v in enumerate(np.concatenate([rng.normal(0, 1, 500),
                                                         rng.normal(3, 1, 500)]))
                 if det.update(v)]
        assert fired, "a three-sigma shift must be detected"
        assert 500 <= fired[0] <= 560, f"detected at {fired[0]}, shift at 500"

    def test_does_not_fire_on_a_stationary_stream(self):
        # The property ADWIN is chosen for: a bound on the false positive rate, not a tuned heuristic.
        rng = np.random.default_rng(1)
        det = ADWIN(delta=0.002)
        fired = sum(det.update(v) for v in rng.normal(0, 1, 5000))
        assert fired <= 2, f"{fired} false detections in 5000 stationary samples"

    def test_a_smaller_delta_fires_less(self):
        rng = np.random.default_rng(2)
        data = rng.normal(0, 1, 4000)
        loose = sum(ADWIN(delta=0.2).update(v) for v in data)
        tight_det = ADWIN(delta=1e-6)
        tight = sum(tight_det.update(v) for v in data)
        assert tight <= loose, f"tight {tight} vs loose {loose}"

    def test_the_window_shrinks_when_a_cut_is_made(self):
        # The mechanism: the older part is DROPPED, so the detector adapts to the new normal rather than
        # alarming forever, which is the whole difference from a fixed-baseline control chart.
        rng = np.random.default_rng(3)
        det = ADWIN(delta=0.002)
        for v in rng.normal(0, 1, 400):
            det.update(v)
        before = len(det._window)
        cut_at = None
        for i, v in enumerate(rng.normal(5, 1, 200)):
            if det.update(v):
                cut_at = i
                break
        assert cut_at is not None
        assert len(det._window) < before, "a cut must discard the stale part of the window"

    def test_it_adapts_rather_than_alarming_forever(self):
        # After the shift is absorbed, a stationary stretch at the NEW level must go quiet again.
        rng = np.random.default_rng(4)
        det = ADWIN(delta=0.002)
        for v in rng.normal(0, 1, 400):
            det.update(v)
        after_shift = [det.update(v) for v in rng.normal(4, 1, 1500)]
        assert any(after_shift[:200]), "the shift itself must be detected"
        assert sum(after_shift[400:]) <= 2, "the new level must become the accepted normal"

    def test_detects_a_variance_only_change_poorly_which_is_expected(self):
        # ADWIN watches the MEAN. A change in spread alone is not its job, and recording that is what
        # justifies carrying KSWIN alongside it rather than treating the two as interchangeable.
        rng = np.random.default_rng(5)
        det = ADWIN(delta=0.002)
        fired = sum(det.update(v) for v in np.concatenate([rng.normal(0, 1, 800),
                                                           rng.normal(0, 5, 800)]))
        assert fired <= 3, f"ADWIN should be largely blind to a variance-only change, fired {fired}"

    def test_nan_samples_are_skipped_per_channel(self):
        x = np.random.default_rng(6).normal(size=(600, 2))
        x[100:150, 0] = np.nan
        out = ADWIN().detect(series(x))
        assert np.all(np.isnan(out.meta["per_channel"][100:150, 0]))
        assert np.all(np.isfinite(out.meta["per_channel"][100:150, 1]))

    def test_the_statistic_counts_channels_flagging(self):
        rng = np.random.default_rng(7)
        x = np.vstack([rng.normal(0, 1, (400, 3)), rng.normal(4, 1, (400, 3))])
        out = ADWIN(delta=0.002).detect(series(x))
        assert np.nanmax(out.statistic) >= 1
        assert np.nanmax(out.statistic) <= 3

    def test_invalid_parameters_are_rejected(self):
        for bad in (0.0, 1.0, -1.0):
            with pytest.raises(ValueError, match="delta must be"):
                ADWIN(delta=bad)
        with pytest.raises(ValueError, match="min_sub must be"):
            ADWIN(min_sub=1)


class TestKSPValue:
    def test_the_D_statistic_matches_scipy_exactly(self):
        # D is a definition, not an approximation, so it must agree to floating point. This is the half
        # of the calculation that can be exactly right, and it is where an indexing error would show.
        scipy_stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(0)
        for a_args, b_args in [((0, 1, 60), (0, 1, 80)), ((0, 1, 100), (1, 1, 100)),
                               ((0, 1, 50), (0, 3, 70)), ((0, 1, 200), (0.3, 1, 150))]:
            a, b = rng.normal(*a_args), rng.normal(*b_args)
            d_mine, _ = ks_two_sample_pvalue(a, b)
            assert d_mine == pytest.approx(float(scipy_stats.ks_2samp(a, b).statistic), abs=1e-12)

    def test_the_p_value_agrees_with_scipy_in_the_range_that_matters(self):
        # The p-value is an ASYMPTOTIC approximation and this implementation and scipy use different
        # ones. scipy evaluates Q(sqrt(ne) D); this uses the Numerical Recipes form
        # Q((sqrt(ne) + 0.12 + 0.11/sqrt(ne)) D), whose correction term improves small-sample accuracy.
        # With recent=50 that regime is exactly where KSWIN operates, so the correction is kept.
        #
        # Measured difference against scipy: about 1% at p = 0.9, about 2% at p = 8e-4. In the deep tail
        # the RELATIVE difference grows (roughly 50% at p = 2e-9) and means nothing, because both numbers
        # are approximations to something around a billionth. Agreement is therefore asserted where a
        # p-value is actually read, and by order of magnitude beyond that.
        scipy_stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(0)
        for a_args, b_args in [((0, 1, 60), (0, 1, 80)), ((0, 1, 100), (1, 1, 100)),
                               ((0, 1, 50), (0, 3, 70)), ((0, 1, 200), (0.3, 1, 150))]:
            a, b = rng.normal(*a_args), rng.normal(*b_args)
            _, p_mine = ks_two_sample_pvalue(a, b)
            p_ref = float(scipy_stats.ks_2samp(a, b, method="asymp").pvalue)
            if p_ref > 1e-4:
                assert p_mine == pytest.approx(p_ref, rel=0.05)
            else:
                assert abs(np.log10(max(p_mine, 1e-300)) - np.log10(max(p_ref, 1e-300))) < 0.5

    def test_identical_samples_give_distance_zero_and_p_one(self):
        x = np.arange(50.0)
        d, p = ks_two_sample_pvalue(x, x)
        assert d == 0.0 and p == 1.0

    def test_disjoint_samples_give_distance_one_and_a_tiny_p(self):
        d, p = ks_two_sample_pvalue(np.arange(100.0), np.arange(100.0) + 1000.0)
        assert d == pytest.approx(1.0)
        assert p < 1e-10

    def test_an_empty_sample_does_not_raise(self):
        d, p = ks_two_sample_pvalue(np.empty(0), np.arange(10.0))
        assert d == 0.0 and p == 1.0


class TestKSWIN:
    def test_detects_a_mean_shift(self):
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.normal(0, 1, 600), rng.normal(2, 1, 400)])
        out = KSWIN(window=200, recent=50).detect(series(x))
        peak = int(np.nanargmax(out.statistic))
        assert 600 <= peak <= 700, f"peak at {peak}"

    def test_detects_a_VARIANCE_only_change_that_adwin_misses(self):
        # THE reason to carry KSWIN. A fault that changes the spread while leaving the mean alone is
        # invisible to ADWIN and to every control chart centred on the mean, and it is exactly the kind
        # of thing a developing mechanical fault does before it moves an average.
        rng = np.random.default_rng(1)
        x = np.concatenate([rng.normal(0, 1, 600), rng.normal(0, 5, 400)])

        ks = KSWIN(window=200, recent=50).detect(series(x))
        quiet = np.nanmedian(ks.statistic[200:590])
        assert np.nanmax(ks.statistic[600:750]) > 3 * quiet, "KSWIN must see the spread change"

        adwin_fired = sum(ADWIN(delta=0.002).update(v) for v in x)
        assert adwin_fired <= 3, "ADWIN should be largely blind to it, which is the point"

    def test_the_statistic_is_the_ks_distance_not_the_p_value(self):
        # Bounded in [0, 1] and already oriented so larger is more anomalous. A p-value would need
        # flipping and saturates at exactly the interesting end.
        out = KSWIN(window=200, recent=50).detect(
            series(np.random.default_rng(2).normal(size=800)))
        finite = out.statistic[np.isfinite(out.statistic)]
        assert finite.min() >= 0.0 and finite.max() <= 1.0
        assert "pvalues" in out.meta

    def test_the_warmup_is_undefined_until_the_window_is_full(self):
        out = KSWIN(window=200, recent=50).detect(
            series(np.random.default_rng(3).normal(size=600)))
        assert np.all(np.isnan(out.statistic[:199]))
        assert np.isfinite(out.statistic[199])

    def test_invalid_parameters_are_rejected(self):
        with pytest.raises(ValueError, match="recent must be"):
            KSWIN(recent=2)
        with pytest.raises(ValueError, match="must exceed"):
            KSWIN(window=50, recent=50)
        with pytest.raises(ValueError, match="alpha must be"):
            KSWIN(alpha=1.5)

    def test_the_per_step_alpha_is_not_the_realised_rate_and_alarms_CLUSTER(self):
        # Two measured properties, both limitations of the method rather than defects here, and both
        # reasons the reported statistic is the distance while the operating point comes from the
        # calibration layer.
        #
        # First: the realised rate of p < alpha on stationary data runs ABOVE nominal, by roughly
        # 1.2x at alpha = 0.05 and 1.7x at alpha = 0.005 (mean over four records of 4000 samples).
        #
        # Second, and the more important one: single-record variance is enormous, because consecutive
        # windows overlap by all but one sample so the tests are strongly dependent and excursions arrive
        # in CLUSTERS. One seed of this experiment gave 0.0005 and another 0.008, a factor of sixteen at
        # the same nominal alpha. Averaging over records is the only way to state the rate at all, and
        # the clustering is precisely why this package counts alarm EVENTS rather than samples.
        rates = []
        for seed in range(6):
            rng = np.random.default_rng(100 + seed)
            out = KSWIN(window=200, recent=50, alpha=0.005).detect(series(rng.normal(size=4000)))
            p = out.meta["pvalues"][:, 0]
            rates.append(float(np.mean(p[np.isfinite(p)] < 0.005)))

        assert np.mean(rates) > 0.005, f"expected inflation above nominal, got {np.mean(rates)}"
        assert np.mean(rates) < 0.05, f"but not wildly so: {np.mean(rates)}"
        assert max(rates) > 3 * min(max(min(rates), 1e-6), 1.0), (
            f"single-record variance should be large; if it is not, re-read this test: {rates}"
        )
