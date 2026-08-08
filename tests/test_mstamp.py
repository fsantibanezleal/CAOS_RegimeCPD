"""Tests for mSTAMP.

The load-bearing tests are the ones that check the DIMENSION SUBSET, not just the profile value. A method
that finds the right window while naming the wrong channels has failed at the job this rung exists for,
and it would still pass every test that only looked at the statistic.

MASS is verified against a direct z-normalised Euclidean distance computed the slow, obvious way. An FFT
correlation is exactly the kind of code that returns plausible numbers when its indexing is off by one.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimecpd import Series
from regimecpd.mstamp import MatrixProfile, mass_distance_profile


def series(values, names=None):
    v = np.asarray(values, dtype=float)
    v = v.reshape(-1, 1) if v.ndim == 1 else v
    names = names or tuple("abcdefgh"[: v.shape[1]])
    return Series(np.arange(len(v), dtype=float), v, names)


def slow_znorm_distance(query, window):
    """The definition, written out. Used to check MASS rather than to be fast."""
    q = (query - query.mean()) / (query.std() if query.std() > 1e-10 else 1.0)
    w = (window - window.mean()) / (window.std() if window.std() > 1e-10 else 1.0)
    return float(np.sqrt(np.sum((q - w) ** 2)))


class TestMASS:
    def test_matches_a_directly_computed_znormalised_distance(self):
        # THE test for the FFT path. An off-by-one in the correlation lag produces distances that are
        # entirely plausible and systematically wrong.
        rng = np.random.default_rng(0)
        x = rng.normal(size=400)
        m = 32
        from regimecpd.mstamp import _sliding_stats
        mean, sigma = _sliding_stats(x, m)

        fast = mass_distance_profile(x[100:100 + m], x, mean, sigma)
        for j in (0, 7, 55, 100, 250, len(fast) - 1):
            slow = slow_znorm_distance(x[100:100 + m], x[j:j + m])
            assert fast[j] == pytest.approx(slow, abs=1e-6), f"lag {j}"

    def test_a_window_matches_itself_at_distance_zero(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=300)
        m = 25
        from regimecpd.mstamp import _sliding_stats
        mean, sigma = _sliding_stats(x, m)
        d = mass_distance_profile(x[80:80 + m], x, mean, sigma)
        assert d[80] == pytest.approx(0.0, abs=1e-6)
        assert int(np.argmin(d)) == 80

    def test_z_normalisation_makes_it_scale_and_offset_invariant(self):
        # A pattern is the same pattern after a gain change or a bias shift. That is the property that
        # makes a matrix profile useful across channels measured in different units.
        rng = np.random.default_rng(2)
        base = rng.normal(size=200)
        x = np.concatenate([base, 5.0 * base + 100.0])
        m = 40
        from regimecpd.mstamp import _sliding_stats
        mean, sigma = _sliding_stats(x, m)
        d = mass_distance_profile(x[20:20 + m], x, mean, sigma)
        assert d[220] == pytest.approx(0.0, abs=1e-5), "the rescaled copy must be a perfect match"

    def test_a_flat_window_is_handled_rather_than_dividing_by_zero(self):
        x = np.concatenate([np.random.default_rng(3).normal(size=200), np.zeros(100)])
        m = 30
        from regimecpd.mstamp import _sliding_stats
        mean, sigma = _sliding_stats(x, m)
        d = mass_distance_profile(x[10:10 + m], x, mean, sigma)
        assert np.all(np.isfinite(d))
        assert d[230] == pytest.approx(np.sqrt(2.0 * m)), "a flat window matches no shape"

    def test_a_flat_query_matches_a_flat_window(self):
        x = np.concatenate([np.zeros(80), np.random.default_rng(4).normal(size=200), np.zeros(80)])
        m = 30
        from regimecpd.mstamp import _sliding_stats
        mean, sigma = _sliding_stats(x, m)
        d = mass_distance_profile(x[10:10 + m], x, mean, sigma)
        assert d[300] == pytest.approx(0.0), "flat against flat is a real match, not a worst case"


class TestSlidingStats:
    def test_matches_the_window_by_window_computation(self):
        from regimecpd.mstamp import _sliding_stats
        rng = np.random.default_rng(5)
        x = rng.normal(size=200)
        m = 17
        mean, sigma = _sliding_stats(x, m)
        assert len(mean) == len(x) - m + 1
        for i in (0, 1, 50, len(mean) - 1):
            assert mean[i] == pytest.approx(x[i:i + m].mean(), abs=1e-10)
            assert sigma[i] == pytest.approx(x[i:i + m].std(), abs=1e-8)


class TestDiscordDetection:
    def test_a_planted_discord_has_the_largest_profile_value(self):
        # A repeating pattern plus one stretch that appears nowhere else. The discord is the odd one out,
        # by construction.
        rng = np.random.default_rng(6)
        m = 40
        motif = np.sin(np.linspace(0, 4 * np.pi, m))
        x = np.tile(motif, 12) + rng.normal(0, 0.05, 12 * m)
        x[5 * m:6 * m] = rng.normal(0, 1.0, m)          # the discord

        out = MatrixProfile(window=m).detect(series(x), k=1)
        peak = int(np.nanargmax(out.statistic))
        assert 5 * m - m // 2 <= peak <= 6 * m, f"discord peak at {peak}, planted at {5 * m}"

    def test_the_statistic_is_larger_when_more_anomalous_with_no_sign_flip(self):
        rng = np.random.default_rng(7)
        m = 30
        motif = np.sin(np.linspace(0, 2 * np.pi, m))
        clean = np.tile(motif, 10) + rng.normal(0, 0.05, 10 * m)
        odd = clean.copy()
        odd[4 * m:5 * m] = rng.normal(0, 1.0, m)

        mp = MatrixProfile(window=m)
        assert np.nanmax(mp.detect(series(odd)).statistic) > np.nanmax(mp.detect(series(clean)).statistic)

    def test_the_statistic_sits_at_the_window_start(self):
        # Placing it at the centre or the end shifts every reported onset by a fixed amount, which is
        # invisible in a plot and fatal in a delay metric.
        m = 40
        out = MatrixProfile(window=m).detect(series(np.random.default_rng(8).normal(size=400)))
        assert np.all(np.isnan(out.statistic[-(m - 1):])), "no window begins in the last m-1 samples"
        assert np.isfinite(out.statistic[0])

    def test_the_exclusion_zone_raises_the_profile_but_far_less_than_folklore_claims(self):
        # A MEASURED correction to the usual framing, kept as a test so the number stays honest.
        #
        # The standard telling is that without an exclusion zone every subsequence matches the one beside
        # it, the profile collapses to zero, and the method reports a perfectly ordinary record no matter
        # what is in it. That is the claim this test was originally written to assert, and it failed.
        #
        # What is actually measured on a stochastic smooth signal, window 40, median profile:
        #
        #   smoothing   lag-1 corr   no zone   zone 0.5   ratio
        #      30         0.959       0.291      0.322     0.91
        #     100         0.987       0.323      0.374     0.87
        #     300         0.998       0.296      0.334     0.89
        #
        # About a tenth, not a collapse, and it saturates by a zone of 4 samples. z-normalisation is the
        # reason: it removes the mean and scale, so what remains is pure shape, and the shape of a window
        # changes measurably from one sample to the next even when the raw signal barely moves.
        #
        # The practical consequence is worth carrying into the product: for this class of data the
        # exclusion zone is NOT the critical parameter, the WINDOW LENGTH is. A default of 0.5 is kept
        # because it costs nothing and matters for near-exactly-repeating motifs, which is the case the
        # folklore comes from.
        rng = np.random.default_rng(9)
        m = 40
        smooth = np.convolve(rng.normal(size=1200), np.ones(100) / 100.0, mode="same")
        s = series(smooth)

        medians = [float(np.nanmedian(MatrixProfile(window=m, exclusion=e).detect(s).statistic))
                   for e in (0.0, 0.1, 0.5, 1.0)]

        assert medians[0] < medians[1], "excluding the neighbour must raise the profile"
        assert all(b >= a - 1e-12 for a, b in zip(medians, medians[1:])), (
            f"the profile must be non-decreasing in the zone size: {medians}"
        )
        assert 0.7 < medians[0] / medians[2] < 0.98, (
            f"the effect is real but modest; if this moves, re-measure the table above: {medians}"
        )
        # Doubling the zone from half a window to a full window moves the median by well under a
        # percent, which is the saturation the table above shows.
        assert medians[3] / medians[2] < 1.01, f"the effect should be saturated by a half-window: {medians}"

    def test_the_exclusion_zone_is_exactly_what_was_asked_for(self):
        # There is no floor of 1 on the zone: exclusion=0 must mean "only the self-match is removed",
        # which is the literature's no-exclusion baseline. A silent floor made the parameter impossible
        # to test at its own boundary, and it was removed for that reason.
        rng = np.random.default_rng(20)
        m = 20
        x = rng.normal(size=400)
        s = series(x)
        zero = MatrixProfile(window=m, exclusion=0.0).detect(s).statistic
        # With only the self-match removed, a window's nearest neighbour may legitimately be adjacent.
        # With a real zone it cannot be, so the profile must be strictly larger somewhere.
        wide = MatrixProfile(window=m, exclusion=1.0).detect(s).statistic
        assert np.nanmin(zero) <= np.nanmin(wide)
        assert np.nanmedian(wide) > np.nanmedian(zero)


class TestDimensionSubsets:
    @staticmethod
    def _four_channel_discord():
        rng = np.random.default_rng(10)
        m = 40
        motif = np.sin(np.linspace(0, 4 * np.pi, m))
        n_rep = 12
        cols = [np.tile(motif, n_rep) + rng.normal(0, 0.05, n_rep * m) for _ in range(4)]
        x = np.column_stack(cols)
        # The discord is planted in channels b and d only; a and c keep repeating cleanly.
        x[5 * m:6 * m, 1] = rng.normal(0, 1.0, m)
        x[5 * m:6 * m, 3] = rng.normal(0, 1.0, m)
        return series(x, ("a", "b", "c", "d")), m

    def test_anomalous_channels_names_the_channels_the_discord_lives_in(self):
        # THE job of this rung. A method that finds the right window while naming the wrong channels has
        # failed, and would still pass every test that only looked at the statistic.
        s, m = self._four_channel_discord()
        mp = MatrixProfile(window=m)
        out = mp.detect(s, k=4)
        peak = int(np.nanargmax(out.statistic))
        assert set(mp.anomalous_channels(out, peak).top(2)) == {"b", "d"}, \
            mp.anomalous_channels(out, peak).ranked()

    def test_the_MATCHING_subset_names_exactly_the_wrong_channels_at_a_discord(self):
        # A finding, pinned so it cannot be rediscovered. mSTAMP's k-dimensional subset is the k channels
        # with the SMALLEST distances, because the algorithm was designed for motifs. At a discord that
        # names the channels which still look normal.
        #
        # Reading it as attribution reports the two HEALTHY channels as the culprits, with a perfectly
        # sensible-looking discord peak beside it. Both halves are asserted here so the distinction
        # between matching_subset and anomalous_channels stays load-bearing.
        s, m = self._four_channel_discord()
        mp = MatrixProfile(window=m)
        out = mp.detect(s, k=4)
        peak = int(np.nanargmax(out.statistic))

        assert set(mp.matching_subset(out, peak, k=2).top(2)) == {"a", "c"}, \
            "the matching subset should name the still-clean channels"
        assert set(mp.anomalous_channels(out, peak).top(2)) == {"b", "d"}, \
            "the discord attribution should name the faulted ones"

    def test_a_partial_fault_is_invisible_at_low_k_and_shows_at_high_k(self):
        # The same asymmetry, in the statistic rather than the attribution. P_k uses the k SMALLEST
        # distances, so a fault confined to two of four channels is hidden at k=2 by the two clean
        # channels and only appears at k=d, where the failing channels can no longer be excluded.
        s, m = self._four_channel_discord()
        mp = MatrixProfile(window=m)
        peaks = {}
        for k in (2, 4):
            stat = mp.detect(s, k=k).statistic
            quiet = np.nanmedian(stat)
            peaks[k] = np.nanmax(stat[5 * m - m // 2:6 * m]) / max(quiet, 1e-9)
        assert peaks[4] > peaks[2], f"a high k must expose the partial fault: {peaks}"

    def test_the_subset_is_binary_and_of_the_requested_size(self):
        rng = np.random.default_rng(11)
        m = 30
        x = rng.normal(size=(600, 4))
        mp = MatrixProfile(window=m)
        out = mp.detect(series(x), k=3)
        att = mp.matching_subset(out, 100, k=3)
        assert set(np.unique(att.scores)) <= {0.0, 1.0}, "a subset membership is not a weight"
        assert int(att.scores.sum()) == 3

    def test_the_profile_is_returned_for_every_k(self):
        # Choosing k is the user's job. A fault in two channels and a fault in nine are different events.
        rng = np.random.default_rng(12)
        x = rng.normal(size=(400, 3))
        profile, dims = MatrixProfile(window=30).compute(series(x))
        assert profile.shape == (3, 400 - 30 + 1)
        assert dims.shape == (3, 400 - 30 + 1, 3)
        for k in range(3):
            assert np.all(dims[k].sum(axis=1) == k + 1)

    def test_the_profile_is_non_decreasing_in_k(self):
        # Adding a dimension can only add a distance at least as large as the smallest already included,
        # so the running mean over the sorted distances cannot fall. A violation means the sort or the
        # cumulative mean is wrong.
        rng = np.random.default_rng(13)
        x = rng.normal(size=(400, 4))
        profile, _ = MatrixProfile(window=30).compute(series(x))
        finite = np.all(np.isfinite(profile), axis=0)
        for k in range(3):
            assert np.all(profile[k + 1, finite] >= profile[k, finite] - 1e-9)

    def test_normalisation_stops_a_noisy_channel_from_owning_every_subset(self):
        # Without it, a channel that happens to be noisier contributes larger distances at every k and
        # dominates the subset selection through its variance rather than through carrying the pattern.
        rng = np.random.default_rng(14)
        m = 40
        n = 600
        quiet = rng.normal(0, 0.1, (n, 2))
        loud = rng.normal(0, 50.0, (n, 1))
        x = np.column_stack([quiet, loud])
        s = series(x, ("q1", "q2", "loud"))

        mp = MatrixProfile(window=m, normalise=True)
        out = mp.detect(s, k=1)
        picks = [mp.matching_subset(out, i, k=1).top(1)[0] for i in range(0, 400, 40)]
        assert picks.count("loud") < len(picks), f"the loud channel dominated anyway: {picks}"

    def test_attribution_on_a_foreign_detection_is_rejected(self):
        from regimecpd import Detection
        det = Detection(np.arange(5.0), np.zeros(5), "not-mstamp")
        with pytest.raises(ValueError, match="no dimension subsets"):
            MatrixProfile.matching_subset(det, 0)
        with pytest.raises(ValueError, match="no dimension subsets"):
            MatrixProfile.anomalous_channels(det, 0)


class TestGuards:
    def test_a_gapped_series_is_rejected_rather_than_silently_filled(self):
        # z-normalisation is undefined over a gap, and quietly interpolating would invent the very shape
        # the method is about to measure.
        x = np.random.default_rng(15).normal(size=(400, 1))
        x[100:120] = np.nan
        with pytest.raises(ValueError, match="complete series"):
            MatrixProfile(window=30).detect(series(x))

    def test_a_window_longer_than_half_the_record_is_rejected(self):
        with pytest.raises(ValueError, match="too short"):
            MatrixProfile(window=200).detect(series(np.random.default_rng(16).normal(size=300)))

    def test_a_tiny_window_is_rejected(self):
        with pytest.raises(ValueError, match="at least 4"):
            MatrixProfile(window=2).detect(series(np.random.default_rng(17).normal(size=300)))

    def test_k_outside_the_channel_count_is_rejected(self):
        s = series(np.random.default_rng(18).normal(size=(400, 3)))
        for bad in (0, 4):
            with pytest.raises(ValueError, match="k must be"):
                MatrixProfile(window=30).detect(s, k=bad)
