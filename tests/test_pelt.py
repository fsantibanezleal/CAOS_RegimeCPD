"""Tests for PELT.

The load-bearing test is `test_pruning_is_exact_and_matches_the_quadratic_optimum`. PELT's contribution is
that pruning costs nothing in accuracy, so verifying it against the unpruned optimum is verifying the
paper's actual claim. A pruning bug otherwise surfaces as slightly different changepoints, which reads as
a tuning difference rather than as an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimecpd import Series
from regimecpd.pelt import PELT, optimal_partition, segmentation_error


def series(values, names=None):
    v = np.asarray(values, dtype=float)
    v = v.reshape(-1, 1) if v.ndim == 1 else v
    names = names or tuple("abcdefgh"[: v.shape[1]])
    return Series(np.arange(len(v), dtype=float), v, names)


def piecewise(breaks, levels, n_each=60, sd=0.4, seed=0, d=1):
    rng = np.random.default_rng(seed)
    parts = [rng.normal(loc=lv, scale=sd, size=(n_each, d)) for lv in levels]
    return series(np.vstack(parts)), [i * n_each for i in range(1, len(levels))]


class TestExactness:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_pruning_is_exact_and_matches_the_quadratic_optimum(self, seed):
        # THE test for this rung. PELT's claim is that pruning costs nothing in accuracy, so the pruned
        # and unpruned dynamic programmes must return the SAME segmentation, not similar ones.
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(150, 1))
        x[50:100] += 2.5
        s = series(x)

        pruned = PELT(min_size=3).segment(s)
        exact = optimal_partition(s, min_size=3)
        assert pruned.tolist() == exact.tolist(), f"pruned {pruned} vs exact {exact}"

    @pytest.mark.parametrize("cost", ["meanvar", "mean"])
    def test_exactness_holds_for_both_cost_functions(self, cost):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(120, 2))
        x[40:80] += 2.0
        s = series(x)
        assert PELT(min_size=3, cost=cost).segment(s).tolist() == \
            optimal_partition(s, min_size=3, cost=cost).tolist()

    def test_exactness_holds_on_pure_noise_with_no_changepoints(self):
        # The case where pruning is most aggressive, so most likely to over-prune.
        s = series(np.random.default_rng(11).normal(size=(200, 1)))
        assert PELT(min_size=3).segment(s).tolist() == optimal_partition(s, min_size=3).tolist()


class TestSegmentation:
    def test_recovers_known_changepoints(self):
        s, truth = piecewise(None, [0.0, 4.0, 0.0, -4.0], n_each=80, seed=0)
        found = PELT(min_size=5, cost="mean").segment(s)
        assert len(found) == len(truth), f"found {found}, expected near {truth}"
        for f, t in zip(found, truth):
            assert abs(f - t) <= 3, (found, truth)

    def test_bic_with_a_free_variance_OVER_segments_a_pure_level_shift(self):
        # A property worth pinning rather than tuning around, and the concrete reason both costs exist.
        #
        # On a record whose changes are purely in the mean, "meanvar" leaves the variance free, so a
        # stretch of ordinary noise can be explained as a short segment with a different variance. With
        # the general-purpose BIC penalty that is cheap enough to be worth doing, and the segmentation
        # comes back with the three true changepoints PLUS spurious ones.
        #
        # Nothing here is broken. The cost function was asked the wrong question, and the answer fits the
        # data well while meaning something other than what was wanted. Reporting only the onset error
        # would have hidden it entirely, which is why segmentation_error returns the count too.
        s, truth = piecewise(None, [0.0, 4.0, 0.0, -4.0], n_each=80, seed=0)

        meanvar = PELT(min_size=5, cost="meanvar").segment(s)
        mean = PELT(min_size=5, cost="mean").segment(s)

        assert len(meanvar) > len(truth), f"expected over-segmentation, got {meanvar}"
        assert len(mean) == len(truth), f"the matched cost should be exact, got {mean}"

        # Both still LOCATE every true changepoint. The difference is entirely in the spurious extras,
        # so an error-only report would rate the two as equally good.
        for cps in (meanvar, mean):
            for t in truth:
                assert min(abs(c - t) for c in cps) <= 3, (cps, t)

        # And a stiffer penalty recovers the right count, which is the honest fix when the cost cannot
        # be changed: the penalty is not a nuisance parameter to leave at its default.
        stiffer = PELT(min_size=5, cost="meanvar", penalty=60.0).segment(s)
        assert len(stiffer) == len(truth), stiffer

    def test_finds_nothing_in_a_stationary_record(self):
        s = series(np.random.default_rng(3).normal(size=(400, 1)))
        assert len(PELT(min_size=5).segment(s)) == 0

    def test_a_higher_penalty_never_returns_more_changepoints(self):
        s, _ = piecewise(None, [0.0, 3.0, 0.0, 3.0, 0.0], n_each=60, seed=4)
        counts = [len(PELT(penalty=p, min_size=5).segment(s)) for p in (2.0, 20.0, 200.0, 2000.0)]
        assert all(b <= a for a, b in zip(counts, counts[1:])), counts
        assert counts[0] > counts[-1], f"the penalty must actually bite: {counts}"

    def test_min_size_is_respected(self):
        s, _ = piecewise(None, [0.0, 5.0, 0.0], n_each=50, seed=5)
        found = PELT(min_size=20, penalty=5.0).segment(s)
        boundaries = np.concatenate([[0], found, [s.n]])
        assert np.all(np.diff(boundaries) >= 20), boundaries

    def test_a_multivariate_change_in_one_channel_is_found(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(300, 4))
        x[150:, 2] += 4.0
        found = PELT(min_size=10).segment(series(x))
        assert len(found) >= 1
        assert min(abs(f - 150) for f in found) <= 5

    def test_the_mean_cost_does_not_explain_a_level_shift_as_a_variance_change(self):
        # Why both costs exist. A pure level shift should be found by both, but "meanvar" is free to
        # attribute a shift to variance, which fits well and means something different.
        rng = np.random.default_rng(8)
        x = np.concatenate([rng.normal(0, 1, 150), rng.normal(3, 1, 150)]).reshape(-1, 1)
        found = PELT(min_size=10, cost="mean").segment(series(x))
        assert len(found) == 1 and abs(found[0] - 150) <= 4, found

    def test_a_variance_change_with_no_level_change_needs_meanvar(self):
        rng = np.random.default_rng(9)
        x = np.concatenate([rng.normal(0, 1, 200), rng.normal(0, 4, 200)]).reshape(-1, 1)
        meanvar = PELT(min_size=10, cost="meanvar").segment(series(x))
        assert len(meanvar) >= 1 and min(abs(f - 200) for f in meanvar) <= 8, meanvar


class TestRobustness:
    def test_incomplete_samples_are_dropped_and_indices_still_refer_to_real_samples(self):
        # A reported changepoint must always point at an observation that exists.
        rng = np.random.default_rng(10)
        x = rng.normal(size=(300, 1))
        x[150:] += 5.0
        x[60:90] = np.nan
        s = series(x)
        found = PELT(min_size=10).segment(s)
        assert len(found) >= 1
        assert np.all(np.isfinite(s.x[found])), "a changepoint landed on a dropped sample"
        assert min(abs(f - 150) for f in found) <= 6

    def test_a_series_with_no_complete_samples_is_rejected(self):
        with pytest.raises(ValueError, match="no complete samples"):
            PELT().segment(series(np.full(50, np.nan)))

    def test_a_constant_series_does_not_produce_infinities(self):
        found = PELT(min_size=5).segment(series(np.zeros(200)))
        assert len(found) == 0

    def test_meanvar_with_min_size_one_is_rejected(self):
        # A variance cannot be estimated from a single point, and letting it through produces segments
        # whose cost is a function of the floor rather than of the data.
        with pytest.raises(ValueError, match="min_size of at least 2"):
            PELT(cost="meanvar", min_size=1)

    def test_unknown_cost_is_rejected(self):
        with pytest.raises(ValueError, match="unknown cost"):
            PELT(cost="magic")


class TestSegmentationError:
    def test_it_returns_the_count_alongside_the_error(self):
        # The count is not optional. A segmentation cut into fifty pieces will almost always have one
        # landing near the true onset, and reporting only the error presents that as an excellent result.
        error, count = segmentation_error(np.array([10, 97, 200]), np.arange(300.0), true_onset=100.0)
        assert error == pytest.approx(3.0)
        assert count == 3

    def test_a_shotgun_segmentation_shows_up_in_the_count_not_the_error(self):
        shotgun = np.arange(0, 300, 5)
        error, count = segmentation_error(shotgun, np.arange(300.0), true_onset=100.0)
        assert error == 0.0, "the error alone makes this look perfect"
        assert count == 60, "the count is what reveals it"

    def test_no_changepoints_gives_infinite_error_not_zero(self):
        error, count = segmentation_error(np.empty(0, dtype=int), np.arange(100.0), 50.0)
        assert error == float("inf") and count == 0

    def test_it_uses_the_time_axis_rather_than_sample_indices(self):
        # Irregular sampling is the norm on fleet telemetry, so an error in samples is not an error in
        # time and the two must not be confused.
        t = np.arange(100.0) * 10.0            # ten time units per sample
        error, _ = segmentation_error(np.array([50]), t, true_onset=520.0)
        assert error == pytest.approx(20.0)


def test_the_module_exposes_no_detect_method():
    # PELT is retrospective: it sees the whole record before deciding where the onset was. Wiring it into
    # an online detection metric would score a method with access to the future, and it would score
    # spectacularly. The absence of `detect` is a deliberate part of the design, so it is asserted.
    assert not hasattr(PELT, "detect"), "PELT must not offer an online detection interface"
