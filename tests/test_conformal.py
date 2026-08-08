"""Tests for conformal calibration.

Two things are checked that most tests of this kind skip.

The finite-sample validity bound is asserted directly, on exchangeable data, over many independent
replications: `Pr(p <= alpha) <= alpha` is the whole reason to use this method and it either holds or it
does not.

And the bound is then shown to BREAK on dependent time-series data, which is the case this package
actually operates in. A conformal layer whose guarantee is quoted but never checked against the data it
runs on is borrowed authority, and the adaptive variant exists precisely because of that gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimecpd import Detection
from regimecpd.conformal import (
    AdaptiveConformal,
    SplitConformal,
    calibration_report,
    conformalise,
)


def det(statistic, method="test"):
    s = np.asarray(statistic, dtype=float)
    return Detection(np.arange(len(s), dtype=float), s, method)


class TestValidity:
    def test_the_finite_sample_bound_holds_on_exchangeable_data(self):
        # THE guarantee: Pr(p <= alpha) <= alpha for any finite calibration set, distribution-free.
        # Checked by replication rather than argued from the formula.
        rng = np.random.default_rng(0)
        n_cal, n_test, reps = 200, 200, 400
        hits = {a: 0 for a in (0.2, 0.1, 0.05, 0.02)}
        total = 0
        for _ in range(reps):
            pool = rng.normal(size=n_cal + n_test)
            model = SplitConformal().fit(pool[:n_cal])
            p = model.p_values(pool[n_cal:])
            total += len(p)
            for a in hits:
                hits[a] += int(np.sum(p <= a))
        for a, count in hits.items():
            realised = count / total
            assert realised <= a * 1.10, f"alpha={a}: realised {realised:.4f} exceeds the bound"

    def test_the_p_value_is_super_uniform_not_merely_bounded(self):
        rng = np.random.default_rng(1)
        pool = rng.normal(size=20000)
        model = SplitConformal().fit(pool[:10000])
        p = model.p_values(pool[10000:])
        for a in (0.05, 0.2, 0.5, 0.8):
            assert abs(float(np.mean(p <= a)) - a) < 0.02, a

    def test_dropping_the_plus_one_would_break_the_bound(self):
        # The `1 +` in numerator and denominator is what makes this valid at finite n. Dropping it looks
        # like a harmless simplification and breaks the bound in the direction of MORE alarms than
        # promised, which is the direction nobody checks. Demonstrated with a small calibration set,
        # where the difference is largest.
        rng = np.random.default_rng(2)
        n_cal = 40
        naive_exceed = 0
        proper_exceed = 0
        reps = 3000
        for _ in range(reps):
            pool = rng.normal(size=n_cal + 1)
            cal = np.sort(pool[:n_cal])
            s = pool[-1]
            at_or_above = n_cal - np.searchsorted(cal, s, side="left")
            proper = (1.0 + at_or_above) / (n_cal + 1.0)
            naive = at_or_above / n_cal if n_cal else 1.0
            alpha = 0.05
            proper_exceed += int(proper <= alpha)
            naive_exceed += int(naive <= alpha)
        assert proper_exceed / reps <= 0.05 * 1.15, proper_exceed / reps
        assert naive_exceed / reps > proper_exceed / reps, (
            "the naive form must fire more often; if it does not, this test proves nothing"
        )

    def test_the_bound_BREAKS_on_dependent_time_series_data(self):
        # The honest half. Conformal validity assumes exchangeability, and a time series is not
        # exchangeable. Quoting the guarantee without checking it on the data it runs on is borrowed
        # authority, and this is why AdaptiveConformal exists.
        rng = np.random.default_rng(3)

        def ar1(n, phi=0.98, seed_rng=rng):
            e = seed_rng.normal(size=n)
            x = np.empty(n)
            x[0] = e[0]
            for i in range(1, n):
                x[i] = phi * x[i - 1] + e[i]
            return x

        realised = []
        for _ in range(60):
            series = np.abs(ar1(2000))
            model = SplitConformal().fit(series[:1000])
            p = model.p_values(series[1000:])
            realised.append(float(np.mean(p <= 0.05)))

        spread = float(np.std(realised))
        assert spread > 0.02, (
            f"with dependent data the realised rate should vary wildly between records, got sd {spread}"
        )
        assert max(realised) > 0.05 * 2, (
            f"some records must exceed nominal substantially; max was {max(realised)}"
        )


class TestSplitConformal:
    def test_a_tiny_calibration_set_is_rejected_with_its_resolution_explained(self):
        with pytest.raises(ValueError, match="smallest p-value"):
            SplitConformal().fit(np.arange(10.0))

    def test_the_resolution_is_the_smallest_reachable_p_value(self):
        model = SplitConformal().fit(np.random.default_rng(0).normal(size=99))
        assert model.resolution == pytest.approx(1 / 100)
        p = model.p_values(np.array([1e9]))
        assert p[0] == pytest.approx(model.resolution)

    def test_a_budget_below_the_resolution_is_unreachable_by_construction(self):
        # Not strictness, impossibility: no observation however extreme can produce a smaller p-value.
        model = SplitConformal().fit(np.random.default_rng(1).normal(size=50))
        p = model.p_values(np.array([1e12, 1e15]))
        assert np.all(p >= model.resolution - 1e-15)
        assert np.all(p > 1e-4)

    def test_nan_in_gives_nan_out(self):
        model = SplitConformal().fit(np.random.default_rng(2).normal(size=200))
        p = model.p_values(np.array([np.nan, 0.0, np.nan]))
        assert np.isnan(p[0]) and np.isnan(p[2]) and np.isfinite(p[1])

    def test_p_values_before_fit_raise(self):
        with pytest.raises(RuntimeError, match="before fit"):
            SplitConformal().p_values(np.zeros(3))

    def test_conform_keeps_the_orientation_and_avoids_saturation(self):
        # -log10(p) rather than p: larger stays more anomalous, and it does not floor at 1/(n+1) where
        # an alarm-budget curve most needs its resolution.
        rng = np.random.default_rng(3)
        model = SplitConformal().fit(rng.normal(size=1000))
        out = model.conform(det(np.array([-2.0, 0.0, 2.0, 10.0])))
        assert np.all(np.diff(out.statistic) > 0), "must stay monotone in the original score"
        assert out.statistic[-1] > out.statistic[0]
        assert "p_value" in out.meta and out.meta["source_method"] == "test"

    def test_it_puts_two_different_score_scales_onto_one_axis(self):
        # THE reason this module exists. Two detectors on wildly different scales must become comparable.
        rng = np.random.default_rng(4)
        small = rng.normal(size=2000) * 0.001
        huge = rng.normal(size=2000) * 1000.0

        m_small = SplitConformal().fit(small[:1000])
        m_huge = SplitConformal().fit(huge[:1000])

        # The 99th percentile of each held-out set must land at about the same conformal p-value.
        p_small = m_small.p_values(np.quantile(small[1000:], 0.99))
        p_huge = m_huge.p_values(np.quantile(huge[1000:], 0.99))
        assert abs(p_small - p_huge) < 0.02, (p_small, p_huge)

    def test_conformalise_shares_one_calibration_across_a_fleet(self):
        rng = np.random.default_rng(5)
        cal = rng.normal(size=2000)
        fleet = [det(rng.normal(size=300)) for _ in range(5)]
        out = conformalise(fleet, cal)
        assert len(out) == 5
        assert all(o.meta["n_calibration"] == 2000 for o in out)


class TestAdaptiveConformal:
    def test_it_recovers_the_target_rate_on_dependent_data_where_split_does_not(self):
        # The point of the adaptive variant: long-run empirical coverage under NOTHING, rather than
        # marginal coverage under an assumption known to be false.
        rng = np.random.default_rng(0)
        phi = 0.99
        e = rng.normal(size=40000)
        x = np.empty(40000)
        x[0] = e[0]
        for i in range(1, 40000):
            x[i] = phi * x[i - 1] + e[i]
        x = np.abs(x)

        target = 0.02
        adaptive = AdaptiveConformal(target=target, gamma=0.02).fit(x[:5000])
        alarms, trace = adaptive.run(det(x[5000:]))
        realised = float(np.mean(alarms))
        assert abs(realised - target) < 0.01, f"adaptive realised {realised} against target {target}"
        # The trace is deliberately unclipped, so it is only required to stay in a sane band rather than
        # inside [0, 1]. See test_the_level_MUST_be_allowed_to_go_negative_or_the_guarantee_breaks.
        assert np.all(np.isfinite(trace))
        assert trace.max() < 1.0, "on well-behaved data the level should not need to exceed 1"

    def test_the_level_tightens_when_alarms_come_too_often(self):
        # A detector whose test scores are all far above calibration should drive the level down.
        rng = np.random.default_rng(1)
        adaptive = AdaptiveConformal(target=0.01, gamma=0.05).fit(rng.normal(size=1000))
        _, trace = adaptive.run(det(np.full(500, 100.0)))
        assert trace[-1] < trace[0], "the level must tighten under a flood of alarms"
        assert trace.min() < 0.0, "the level must be allowed to go negative; see the next test"

    def test_the_level_MUST_be_allowed_to_go_negative_or_the_guarantee_breaks(self):
        # A bug found while writing these tests, kept as a regression test because the broken version
        # looked entirely reasonable.
        #
        # Clipping the level at 0 seems harmless: a negative level already means "never alarm", so why
        # carry it? Because the magnitude of the negative excursion is the DEBT the recursion works off
        # before firing again, and that debt is what enforces the long-run rate.
        #
        # With a saturated detector (every score above every calibration score, so every p-value sits at
        # the resolution floor), one alarm should cost gamma*(1 - target) of level and take about
        # 1/target quiet steps to repay. Clipped at zero the debt is discarded, the level climbs back
        # within two steps, and the realised rate lands near 50% against a target of 1%.
        rng = np.random.default_rng(11)
        target, gamma = 0.01, 0.05
        adaptive = AdaptiveConformal(target=target, gamma=gamma).fit(rng.normal(size=1000))
        alarms, _ = adaptive.run(det(np.full(20000, 100.0)))
        realised = float(np.mean(alarms))
        assert abs(realised - target) < 0.005, (
            f"a saturated detector must still be held to its budget, got {realised}"
        )

        # And the clipped variant, computed here explicitly, must fail that.
        alpha, fired_count = target, 0
        p_floor = 1.0 / 1001
        for _ in range(20000):
            fired = p_floor <= alpha
            fired_count += int(fired)
            alpha = min(max(alpha + gamma * (target - float(fired)), 0.0), 1.0)
        clipped_rate = fired_count / 20000
        assert clipped_rate > 10 * target, (
            f"if clipping no longer breaks it, this test no longer proves anything: {clipped_rate}"
        )

    def test_the_level_loosens_when_alarms_dry_up(self):
        rng = np.random.default_rng(2)
        adaptive = AdaptiveConformal(target=0.05, gamma=0.02).fit(rng.normal(size=1000))
        _, trace = adaptive.run(det(np.full(500, -100.0)))
        assert trace[-1] > trace[0], "the level must loosen when nothing ever fires"

    def test_an_undefined_statistic_is_neither_an_alarm_nor_an_error(self):
        rng = np.random.default_rng(3)
        adaptive = AdaptiveConformal(target=0.05, gamma=0.1).fit(rng.normal(size=1000))
        alarms, trace = adaptive.run(det(np.full(200, np.nan)))
        assert not alarms.any()
        assert np.allclose(trace, trace[0]), "the level must not drift on undefined samples"

    def test_invalid_parameters_are_rejected(self):
        for bad in (0.0, 1.0, -0.5):
            with pytest.raises(ValueError, match="target must be"):
                AdaptiveConformal(target=bad)
        for bad in (0.0, 1.5):
            with pytest.raises(ValueError, match="gamma must be"):
                AdaptiveConformal(gamma=bad)


class TestCalibrationReport:
    def test_it_reports_realised_against_nominal_and_they_differ_on_real_data(self):
        # The most useful output in the module: the gap between the promise and the measurement, shown
        # rather than assumed away.
        rng = np.random.default_rng(0)
        cal = np.abs(rng.normal(size=4000))
        held = [det(np.abs(rng.normal(size=1000))) for _ in range(6)]
        report = calibration_report(SplitConformal().fit(cal), held)

        assert report["n_calibration"] == 4000
        assert report["resolution"] == pytest.approx(1 / 4001)
        for row in report["rows"]:
            if row["alpha"] >= report["resolution"] * 10:
                assert 0.5 < row["ratio"] < 2.0, row

    def test_it_reports_events_as_well_as_samples(self):
        # Those differ by the dwell time of the statistic, and the per-event number is the one an
        # operator experiences.
        rng = np.random.default_rng(1)
        cal = np.abs(rng.normal(size=3000))
        held = [det(np.abs(rng.normal(size=800))) for _ in range(4)]
        report = calibration_report(SplitConformal().fit(cal), held)
        for row in report["rows"]:
            assert "events_per_unit_time" in row and "n_events" in row
            assert row["n_events"] >= 0

    def test_a_budget_under_the_resolution_shows_up_as_zero_events(self):
        rng = np.random.default_rng(2)
        model = SplitConformal().fit(np.abs(rng.normal(size=200)))
        held = [det(np.abs(rng.normal(size=500))) for _ in range(3)]
        report = calibration_report(model, held, alphas=(0.0001,))
        assert report["rows"][0]["realised_per_sample"] == 0.0, (
            "a budget below the resolution cannot be met by any observation, so nothing fires"
        )
