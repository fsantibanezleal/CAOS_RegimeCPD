"""Regressions for the 2026-08-10 engine review.

Every defect below was reproduced against a GREEN 301-test suite. None of them raises, none produces an
obviously wrong shape, and every one produces a number a reader would accept. That is the whole reason
they need tests written from the defect rather than from the behaviour: a behavioural test written after
the fact would have been written against the broken output.

The IDs are the review's own, so a reader can go from a failure here to the analysis that found it.
"""
from __future__ import annotations

import numpy as np
import pytest

import regimecpd as rc
from regimecpd.metrics import UnitOutcome, rising_edges, score_fleet, score_unit
from regimecpd.regime import KMeansRegimes
from regimecpd.types import Detection


def _det(stat, t=None):
    s = np.asarray(stat, dtype=float)
    return Detection(np.arange(len(s), dtype=float) if t is None else np.asarray(t, float), s, "fixture")


class TestD6GapInsideAnExcursion:
    """A NaN inside a sustained excursion must not split one alarm into two events.

    The residual arm carries NaN BY DESIGN: a sample whose context falls outside every regime seen in the
    baseline is deliberately unassigned. So this bias fell on precisely the arm the product argues for,
    and inflated its false-alarm count by roughly 18x at 10% unassigned.
    """

    def test_one_excursion_with_a_hole_is_one_alarm(self):
        stat = np.concatenate([np.zeros(10), np.full(30, 5.0), np.zeros(10)])
        stat[20:24] = np.nan                      # the truck left every known regime for four samples
        out = UnitOutcome(_det(stat), onset_t=None, unit_id="h")
        score = score_unit(out, threshold=1.0)
        assert score.n_false_alarms == 1, (
            f"one excursion with a hole in it counted as {score.n_false_alarms} alarms")

    def test_two_genuinely_separate_excursions_are_still_two(self):
        stat = np.concatenate([np.zeros(5), np.full(10, 5.0), np.zeros(10), np.full(10, 5.0), np.zeros(5)])
        out = UnitOutcome(_det(stat), onset_t=None, unit_id="h")
        assert score_unit(out, threshold=1.0).n_false_alarms == 2, "the fix must not merge real events"

    def test_an_all_nan_gap_does_not_invent_an_alarm(self):
        stat = np.full(40, np.nan)
        stat[:10] = 0.0
        out = UnitOutcome(_det(stat), onset_t=None, unit_id="h")
        assert score_unit(out, threshold=1.0).n_false_alarms == 0

    def test_rising_edges_without_observed_still_behaves_as_before(self):
        m = np.array([False, True, True, False, True])
        assert list(rising_edges(m)) == [1, 4]


class TestD7NanOnset:
    """A NaN onset used to score as a DETECTION and erase the whole fleet's delay."""

    def test_a_nan_onset_is_refused_rather_than_scored(self):
        with pytest.raises(ValueError, match="non-finite onset"):
            score_unit(UnitOutcome(_det(np.zeros(10)), onset_t=float("nan"), unit_id="bad"), 1.0)

    def test_one_bad_unit_can_no_longer_poison_the_fleet_median(self):
        good = [UnitOutcome(_det(np.concatenate([np.zeros(50 + d), np.full(50, 9.0)])),
                            onset_t=50.0, unit_id=f"u{d}") for d in (10, 30, 50)]
        fleet = score_fleet(good, threshold=1.0)
        assert np.isfinite(fleet.median_delay), "a clean fleet must report a finite median delay"
        with pytest.raises(ValueError):
            score_fleet(good + [UnitOutcome(_det(np.zeros(100)), onset_t=float("nan"))], threshold=1.0)


class TestD9DegenerateClusterRadius:
    """A cluster whose baseline members sit at one point must not reject every sample it ever sees."""

    def test_a_point_cluster_still_accepts_its_own_centre(self):
        # Two well-separated groups, one of which is EXACTLY constant: a held setpoint or a discrete
        # operating condition. Its quantile radius is roundoff, and `distance > factor * ~0` was then
        # true for everything, so the regime existed and was never assignable.
        rng = np.random.default_rng(0)
        ctx = np.vstack([
            np.column_stack([rng.normal(0, 0.4, 200), rng.normal(0, 0.4, 200)]),
            np.tile([10.0, 10.0], (200, 1)),
        ])
        seg = KMeansRegimes(n_regimes=2, seed=0).fit(ctx)
        labels = seg.label(ctx).labels
        # The constant group is the second half; every one of its samples sits on its own centre.
        assert np.all(labels[200:] >= 0), (
            f"{np.count_nonzero(labels[200:] < 0)} of 200 samples sitting exactly on their own centre "
            "were rejected as novel")
        assert len(set(labels[labels >= 0])) == 2, "both regimes must be usable"

    def test_a_genuinely_novel_context_is_still_rejected(self):
        rng = np.random.default_rng(1)
        ctx = np.column_stack([rng.normal(0, 1, 300), rng.normal(0, 1, 300)])
        seg = KMeansRegimes(n_regimes=3, seed=0).fit(ctx)
        assert seg.label(np.array([[500.0, 500.0]])).labels[0] == -1


class TestD3PeltConstantChannel:
    """PELT `meanvar` must not segment on floating-point roundoff in a channel that never moves."""

    def test_a_constant_channel_contributes_no_changepoints(self):
        rng = np.random.default_rng(2)
        n = 600
        live = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(4, 1, n // 2)])
        dead = np.full(n, 21610.0)
        s = rc.Series(np.arange(n, dtype=float), np.column_stack([live, dead]), ("live", "dead"))
        cps_both = rc.PELT(min_size=40, cost="meanvar").segment(s)

        only_dead = rc.Series(np.arange(n, dtype=float), dead.reshape(-1, 1), ("dead",))
        cps_dead = rc.PELT(min_size=40, cost="meanvar").segment(only_dead)
        assert len(cps_dead) == 0, (
            f"a channel that never moves produced {len(cps_dead)} changepoints on its own")
        # And the live channel's real change is still found.
        assert any(abs(c - n // 2) < 60 for c in cps_both), f"the real change was lost: {cps_both}"


class TestD4PcaResidualSubspace:
    """SPE must not calibrate itself to roundoff when a channel is a linear combination of others."""

    def test_a_degenerate_residual_subspace_reports_an_UNDEFINED_limit(self):
        rng = np.random.default_rng(3)
        n = 400
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        # c is EXACTLY a + b: the residual subspace direction along it carries no variance at all.
        x = np.column_stack([a, b, a + b])
        s = rc.Series(np.arange(n, dtype=float), x, ("a", "b", "c"))
        mon = rc.PCAMonitor(statistic="spe", variance_target=0.95).fit(s)
        # The residual variance here is roundoff, so a LIMIT computed from it would be roundoff too, and
        # comparable to an SPE that is also roundoff: a chart that looks calibrated and measures nothing.
        # NaN says "undefined", which a caller can see.
        assert not np.isfinite(mon.spe_limit()), (
            f"spe_limit returned {mon.spe_limit():.3g} from a residual subspace that carries no variance")

    def test_a_real_break_in_the_correlation_structure_is_still_caught(self):
        rng = np.random.default_rng(4)
        n = 400
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        base = rc.Series(np.arange(n, dtype=float), np.column_stack([a, b, a + b]), ("a", "b", "c"))
        mon = rc.PCAMonitor(statistic="spe", variance_target=0.95).fit(base)
        a2 = rng.normal(size=n)
        b2 = rng.normal(size=n)
        broken = np.column_stack([a2, b2, a2 - b2])       # the relationship c = a + b is gone
        spe_ok = mon.detect(base).statistic
        spe_broken = mon.detect(rc.Series(np.arange(n, dtype=float), broken, ("a", "b", "c"))).statistic
        # Against the FITTING data, not an arbitrary constant: what has to hold is that breaking the
        # relationship raises the statistic by orders of magnitude, whatever its units happen to be.
        assert np.nanmedian(spe_broken) > 1e3 * max(np.nanmedian(spe_ok), 1e-12), (
            f"broken median {np.nanmedian(spe_broken):.3g} against fitted {np.nanmedian(spe_ok):.3g}: "
            "SPE no longer sees a break in the correlation structure it was fitted to")


class TestD2MstampFlatness:
    """mSTAMP must decide "flat" once, with one estimator, on a scale-relative floor."""

    def test_the_flat_decision_does_not_depend_on_the_unit_of_the_channel(self):
        rng = np.random.default_rng(5)
        n = 500
        small = rng.normal(0, 1e-6, n)                    # millivolts
        large = small * 1e9                               # the same signal in different units
        mp = rc.MatrixProfile(window=40)
        d_small = mp.detect(rc.Series(np.arange(n, dtype=float), small.reshape(-1, 1), ("x",)))
        d_large = mp.detect(rc.Series(np.arange(n, dtype=float), large.reshape(-1, 1), ("x",)))
        a = d_small.statistic[np.isfinite(d_small.statistic)]
        b = d_large.statistic[np.isfinite(d_large.statistic)]
        assert a.size == b.size and a.size > 0
        # z-normalised distance is scale free, so the SAME signal in different units must profile the same.
        assert np.allclose(a, b, rtol=1e-6, atol=1e-9), (
            "the flatness decision is unit dependent: rescaling the channel changed the profile")
