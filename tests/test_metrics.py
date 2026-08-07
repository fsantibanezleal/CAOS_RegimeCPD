"""Tests for the measurement layer.

These are the most important tests in the package, because every claim it makes is a number this module
produced. A detector with a subtle bug produces a wrong result; a metric with a subtle bug produces a
wrong result that agrees with itself everywhere and cannot be spotted from the outputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimecpd import (
    Detection,
    UnitOutcome,
    alarm_budget_curve,
    bootstrap_ci,
    candidate_thresholds,
    rising_edges,
    score_fleet,
    score_unit,
    threshold_for_budget,
)


def _det(statistic, t=None, method="test"):
    s = np.asarray(statistic, dtype=float)
    return Detection(np.arange(len(s), dtype=float) if t is None else t, s, method)


class TestRisingEdges:
    def test_counts_excursions_not_samples(self):
        # One excursion lasting five samples is ONE alarm, not five. This is the convention that makes
        # "false alarms per truck-month" mean what an operator experiences.
        mask = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
        assert rising_edges(mask).tolist() == [2]

    def test_separate_excursions_count_separately(self):
        mask = np.array([1, 0, 1, 1, 0, 1], dtype=bool)
        assert rising_edges(mask).tolist() == [0, 2, 5]

    def test_alarming_at_index_zero_is_an_edge(self):
        # A record that opens mid-excursion is an alarm the operator sees, so it counts.
        assert rising_edges(np.array([1, 1, 0], dtype=bool)).tolist() == [0]

    def test_empty_and_all_quiet(self):
        assert rising_edges(np.empty(0, dtype=bool)).size == 0
        assert rising_edges(np.zeros(5, dtype=bool)).size == 0


class TestScoreUnit:
    def test_detection_delay_is_measured_from_the_onset(self):
        # Statistic crosses at t = 12, onset at t = 10, so the delay is 2.
        stat = np.zeros(20)
        stat[12:] = 5.0
        out = UnitOutcome(_det(stat), onset_t=10.0, unit_id="u1")
        s = score_unit(out, threshold=1.0)
        assert s.detected
        assert s.delay == pytest.approx(2.0)
        assert s.n_false_alarms == 0

    def test_alarms_before_onset_are_false_alarms(self):
        stat = np.zeros(20)
        stat[2:4] = 5.0     # false alarm
        stat[7:9] = 5.0     # false alarm
        stat[15:] = 5.0     # the detection
        out = UnitOutcome(_det(stat), onset_t=12.0)
        s = score_unit(out, threshold=1.0)
        assert s.n_false_alarms == 2
        assert s.detected and s.delay == pytest.approx(3.0)

    def test_an_excursion_spanning_the_onset_is_a_false_alarm_and_a_miss(self):
        # THE convention that stops a permanently-alarming detector from scoring as a perfect detector.
        # The edge is at t = 0, before the onset, so it is false; no edge occurs at or after the onset,
        # so nothing was detected.
        out = UnitOutcome(_det(np.full(20, 5.0)), onset_t=10.0)
        s = score_unit(out, threshold=1.0)
        assert s.n_false_alarms == 1
        assert not s.detected
        assert s.delay is None

    def test_healthy_unit_has_no_delay_and_all_alarms_are_false(self):
        stat = np.zeros(20)
        stat[5:7] = 5.0
        out = UnitOutcome(_det(stat), onset_t=None)
        s = score_unit(out, threshold=1.0)
        assert not s.is_faulty
        assert s.n_false_alarms == 1
        assert s.delay is None

    def test_healthy_exposure_stops_at_the_onset(self):
        # t runs 0..19. A faulty unit is only "at risk of a false alarm" before its onset.
        faulty = UnitOutcome(_det(np.zeros(20)), onset_t=10.0)
        healthy = UnitOutcome(_det(np.zeros(20)), onset_t=None)
        assert faulty.healthy_exposure == pytest.approx(10.0)
        assert healthy.healthy_exposure == pytest.approx(19.0)

    def test_onset_before_the_record_gives_zero_exposure(self):
        out = UnitOutcome(_det(np.zeros(20), t=np.arange(20.0) + 100.0), onset_t=50.0)
        assert out.healthy_exposure == 0.0

    def test_nan_warmup_never_alarms(self):
        # A detector's statistic is undefined during warm-up. Undefined must not be able to fire.
        stat = np.concatenate([np.full(5, np.nan), np.zeros(15)])
        out = UnitOutcome(_det(stat), onset_t=None)
        assert score_unit(out, threshold=-np.inf).n_false_alarms == 1  # the first defined sample only


class TestScoreFleet:
    def test_rate_is_pooled_over_exposure_not_averaged_over_units(self):
        # A short record must not carry the same weight as a long one. Unit A: 1 false alarm in 10
        # units of time. Unit B: 1 false alarm in 90. Pooled rate is 2/100, not the mean of 0.1 and 0.011.
        a = np.zeros(11)
        a[3] = 5.0
        b = np.zeros(91)
        b[3] = 5.0
        outs = [UnitOutcome(_det(a), None, "a"), UnitOutcome(_det(b), None, "b")]
        fleet = score_fleet(outs, threshold=1.0)
        assert fleet.n_false_alarms == 2
        assert fleet.healthy_exposure == pytest.approx(100.0)
        assert fleet.false_alarms_per_unit_time == pytest.approx(0.02)

    def test_per_converts_the_rate_into_planner_units(self):
        a = np.zeros(101)
        a[3] = 5.0
        fleet = score_fleet([UnitOutcome(_det(a), None)], threshold=1.0)
        assert fleet.false_alarms_per_unit_time == pytest.approx(0.01)
        assert fleet.per(100.0) == pytest.approx(1.0)

    def test_detection_rate_counts_only_faulty_units(self):
        detect = np.zeros(20)
        detect[12:] = 5.0
        miss = np.zeros(20)
        outs = [
            UnitOutcome(_det(detect), 10.0, "f1"),
            UnitOutcome(_det(miss), 10.0, "f2"),
            UnitOutcome(_det(miss), None, "h1"),
        ]
        fleet = score_fleet(outs, threshold=1.0)
        assert fleet.n_faulty == 2
        assert fleet.n_detected == 1
        assert fleet.detection_rate == pytest.approx(0.5)

    def test_no_faulty_units_gives_nan_detection_rate_not_zero(self):
        # Absence of a measurement is not a measurement of zero.
        fleet = score_fleet([UnitOutcome(_det(np.zeros(10)), None)], threshold=1.0)
        assert np.isnan(fleet.detection_rate)


class TestAlarmBudget:
    def test_event_counting_makes_the_rate_NON_monotone_and_that_is_the_trap(self):
        # Documented here because it is counter-intuitive and it silently breaks the obvious search.
        # Counting excursions rather than samples makes the rate unimodal: zero at a high bar (nothing
        # crosses), peaking in the middle (the statistic crosses in and out), and collapsing again at a
        # very low bar, where the statistic sits above the line for the whole record and that is ONE
        # excursion. The low end is a detector that is permanently on and detects nothing.
        rng = np.random.default_rng(0)
        outs = [UnitOutcome(_det(rng.normal(size=500)), None, f"u{i}") for i in range(8)]
        curve = alarm_budget_curve(outs, n=120)
        rates = np.array([s.false_alarms_per_unit_time for s in curve])
        assert np.isfinite(rates).all()

        peak = int(np.argmax(rates))
        assert 0 < peak < len(rates) - 1, "the rate must peak in the interior, not at either end"
        assert rates[0] < rates[peak], "the lowest threshold must be CHEAPER than the peak"
        assert rates[-1] < rates[peak], "the highest threshold must be cheaper than the peak"
        assert not all(b <= a + 1e-12 for a, b in zip(rates, rates[1:])), (
            "if this ever becomes monotone the counting convention changed; re-read threshold_for_budget"
        )

    def test_the_rate_is_monotone_in_the_region_connected_to_never_firing(self):
        # The upper region IS well behaved, which is why the budget search works there.
        rng = np.random.default_rng(7)
        outs = [UnitOutcome(_det(rng.normal(size=500)), None, f"u{i}") for i in range(8)]
        rates = np.array([s.false_alarms_per_unit_time for s in alarm_budget_curve(outs, n=120)])
        upper = rates[int(np.argmax(rates)):]
        assert all(b <= a + 1e-12 for a, b in zip(upper, upper[1:]))

    def test_threshold_for_budget_picks_the_most_sensitive_that_fits(self):
        rng = np.random.default_rng(1)
        outs = [UnitOutcome(_det(rng.normal(size=2000)), None, f"u{i}") for i in range(5)]
        target = 0.01
        th = threshold_for_budget(outs, target)
        assert th is not None
        assert score_fleet(outs, th).false_alarms_per_unit_time <= target
        # A step down from the chosen point must break the budget, or it was not the most sensitive
        # choice in its region. Stepping to the NEXT grid point down, not an arbitrary offset, because a
        # large offset lands in the degenerate always-on region where the rate is low again.
        grid = candidate_thresholds(outs, n=400)
        below = grid[grid < th]
        assert below.size, "the chosen threshold should not be the bottom of the grid"
        assert score_fleet(outs, float(below[-1])).false_alarms_per_unit_time > target

    def test_the_degenerate_always_on_threshold_is_never_selected(self):
        # THE regression test for the trap. A permanently-alarming detector reports one excursion per
        # unit and therefore a tiny rate, while detecting nothing. Selecting it would produce a
        # benchmark row reading as a spectacularly cheap detector.
        rng = np.random.default_rng(11)
        stat = rng.normal(size=3000)
        detect = stat.copy()
        detect[2000:] += 8.0  # a real onset, so "detects nothing" is a meaningful failure
        outs = [UnitOutcome(_det(detect), onset_t=2000.0, unit_id="f0")]
        outs += [UnitOutcome(_det(rng.normal(size=3000)), None, f"h{i}") for i in range(4)]

        floor = float(np.min([o.detection.statistic.min() for o in outs])) - 1.0
        assert score_fleet(outs, floor).false_alarms_per_unit_time < 1e-3, "setup: the trap is cheap"
        assert score_fleet(outs, floor).n_detected == 0, "setup: the trap detects nothing"

        th = threshold_for_budget(outs, target_rate=1e-3)
        assert th is not None and th > floor + 1.0, "the search must not fall into the always-on region"
        assert score_fleet(outs, th).n_detected == 1, "the chosen threshold must actually detect"

    def test_unreachable_budget_returns_none_rather_than_a_silent_maximum(self):
        # A detector pinned above any threshold cannot operate at a negative budget. Returning the top
        # of the grid would disguise "cannot operate" as "operates and detects nothing".
        outs = [UnitOutcome(_det(np.full(50, 1.0)), None)]
        assert threshold_for_budget(outs, target_rate=-1.0) is None


class TestBootstrap:
    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(2)
        outs = [UnitOutcome(_det(rng.normal(size=300)), None, f"u{i}") for i in range(30)]

        def rate(o):
            return score_fleet(o, 1.5).false_alarms_per_unit_time

        point, lo, hi = bootstrap_ci(outs, rate, n_boot=200, seed=3)
        assert lo <= point <= hi
        assert hi > lo, "a fleet of 30 units must not produce a degenerate interval"

    def test_resampling_units_gives_a_wider_interval_than_pretending_samples_are_independent(self):
        # The reason the unit bootstrap exists. Units differ from each other; a sample-level interval
        # would ignore that between-unit spread entirely.
        rng = np.random.default_rng(4)
        # Each unit has its own offset, so between-unit variance dominates within-unit variance.
        outs = [UnitOutcome(_det(rng.normal(loc=rng.normal(0, 2.0), size=200)), None, f"u{i}")
                for i in range(20)]

        def rate(o):
            return score_fleet(o, 1.0).false_alarms_per_unit_time

        _, lo, hi = bootstrap_ci(outs, rate, n_boot=300, seed=5)
        pooled = np.concatenate([o.detection.statistic for o in outs])
        naive_se = np.sqrt(np.mean(pooled > 1.0) * (1 - np.mean(pooled > 1.0)) / pooled.size)
        assert (hi - lo) > 4 * naive_se, "the unit bootstrap must not collapse to the sample-level width"

    def test_empty_fleet_does_not_raise(self):
        point, lo, hi = bootstrap_ci([], lambda o: float("nan"), n_boot=10)
        assert np.isnan(lo) and np.isnan(hi)
