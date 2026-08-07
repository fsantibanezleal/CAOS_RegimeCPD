"""Tests for the data contract.

Most of these assert that a malformed object is REJECTED. A silently-accepted shape mismatch here would
surface later as a benchmark number, which is the worst place to find it.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimecpd import Attribution, Detection, RegimeLabels, Residual, Series


class TestSeries:
    def test_univariate_input_becomes_one_channel(self):
        s = Series(np.arange(5.0), np.arange(5.0), ("fuel",))
        assert s.d == 1 and s.n == 5 and s.x.shape == (5, 1)

    def test_channel_lookup_is_by_name(self):
        s = Series(np.arange(3.0), np.array([[1, 2], [3, 4], [5, 6]]), ("strut", "brake"))
        assert s.channel("brake").tolist() == [2.0, 4.0, 6.0]
        with pytest.raises(KeyError, match="tyre"):
            s.channel("tyre")

    def test_select_preserves_the_requested_order(self):
        s = Series(np.arange(3.0), np.array([[1, 2, 3]] * 3), ("a", "b", "c"))
        assert s.select(["c", "a"]).names == ("c", "a")
        assert s.select(["c", "a"]).x[0].tolist() == [3.0, 1.0]

    def test_non_increasing_time_is_rejected(self):
        # An unsorted series produces negative "durations" and nonsense rates, silently.
        with pytest.raises(ValueError, match="strictly increasing"):
            Series(np.array([0.0, 2.0, 1.0]), np.zeros(3), ("a",))

    def test_length_and_name_mismatches_are_rejected(self):
        with pytest.raises(ValueError, match="samples"):
            Series(np.arange(4.0), np.zeros(5), ("a",))
        with pytest.raises(ValueError, match="names"):
            Series(np.arange(3.0), np.zeros((3, 2)), ("a",))

    def test_duration_of_a_single_sample_is_zero_not_an_error(self):
        assert Series(np.array([7.0]), np.array([1.0]), ("a",)).duration == 0.0


class TestRegimeLabels:
    def test_unassigned_samples_are_visible_rather_than_folded_away(self):
        # -1 must stay -1. Snapping an out-of-baseline sample to its nearest regime hides the one case
        # where the residual is not trustworthy.
        lab = RegimeLabels(np.array([0, 1, -1, 0]), n_regimes=2, method="test")
        assert lab.unassigned.tolist() == [False, False, True, False]
        assert lab.coverage == pytest.approx(0.75)

    def test_coverage_of_an_empty_labelling_is_zero(self):
        assert RegimeLabels(np.empty(0, dtype=int), 1, "test").coverage == 0.0

    def test_zero_regimes_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            RegimeLabels(np.array([0]), n_regimes=0, method="test")


class TestResidual:
    def test_as_series_makes_the_two_arms_indistinguishable_to_a_detector(self):
        # This is what keeps the head-to-head fair: the detector receives the same type either way and
        # has no way to know which arm it is in.
        lab = RegimeLabels(np.zeros(4, dtype=int), 1, "test")
        res = Residual(np.arange(4.0), np.arange(4.0), ("fuel",), lab, "u7")
        s = res.as_series()
        assert isinstance(s, Series)
        assert s.names == ("fuel",) and s.unit_id == "u7" and s.n == 4

    def test_shape_mismatch_is_rejected(self):
        lab = RegimeLabels(np.zeros(4, dtype=int), 1, "test")
        with pytest.raises(ValueError, match="samples"):
            Residual(np.arange(4.0), np.zeros(3), ("a",), lab)


class TestDetection:
    def test_alarms_are_inclusive_at_the_threshold(self):
        det = Detection(np.arange(3.0), np.array([0.9, 1.0, 1.1]), "test")
        assert det.alarms(1.0).tolist() == [False, True, True]

    def test_nan_is_not_alarming(self):
        det = Detection(np.arange(3.0), np.array([np.nan, 5.0, np.nan]), "test")
        assert det.alarms(-np.inf).tolist() == [False, True, False]

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="samples"):
            Detection(np.arange(3.0), np.zeros(4), "test")


class TestAttribution:
    def test_ranking_is_most_implicated_first(self):
        att = Attribution(np.array([0.2, 0.9, 0.5]), ("fuel", "strut", "brake"), "test")
        assert att.ranked()[0] == ("strut", pytest.approx(0.9))
        assert att.top(2) == ("strut", "brake")

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="scores"):
            Attribution(np.array([1.0, 2.0]), ("only-one",), "test")
