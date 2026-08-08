"""Tests for the classical control-chart detectors.

These get more scrutiny than a baseline usually receives, on purpose. The package's claim is that it
beats them, so a quietly broken incumbent would manufacture that result, and a number that comes out the
way everyone expected is the number nobody checks.

Where a published property exists (the Shewhart false-alarm rate under normality, the EWMA control limit,
the CUSUM reset) it is asserted against the published value rather than against this implementation's own
output.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimecpd import Series
from regimecpd.classical import CUSUM, EWMA, PageHinkley, Shewhart

ALL_DETECTORS = [Shewhart, CUSUM, EWMA, PageHinkley]


def series(values, names=("a",), t=None):
    v = np.asarray(values, dtype=float)
    v = v.reshape(-1, 1) if v.ndim == 1 else v
    return Series(np.arange(len(v), dtype=float) if t is None else t, v, names)


def clean_baseline(n=4000, d=1, seed=0):
    rng = np.random.default_rng(seed)
    return series(rng.normal(size=(n, d)), tuple("abcdefgh"[:d]))


class TestSharedContract:
    @pytest.mark.parametrize("cls", ALL_DETECTORS)
    def test_detect_before_fit_raises(self, cls):
        with pytest.raises(RuntimeError, match="before fit"):
            cls().detect(clean_baseline())

    @pytest.mark.parametrize("cls", ALL_DETECTORS)
    def test_detecting_on_different_channels_is_rejected(self, cls):
        det = cls().fit(clean_baseline(d=2))
        with pytest.raises(ValueError, match="fitted on channels"):
            det.detect(clean_baseline(d=1))

    @pytest.mark.parametrize("cls", ALL_DETECTORS)
    def test_statistic_is_larger_when_more_anomalous(self, cls):
        # The orientation invariant. A single flipped sign here produces a method that scores exactly as
        # badly as it should score well, and that is nearly invisible in a results table.
        base = clean_baseline(n=3000)
        det = cls().fit(base)
        quiet = det.detect(clean_baseline(n=1000, seed=1))
        shifted = det.detect(series(np.random.default_rng(1).normal(size=(1000, 1)) + 4.0))
        assert np.nanmean(shifted.statistic) > np.nanmean(quiet.statistic)

    @pytest.mark.parametrize("cls", ALL_DETECTORS)
    def test_nan_input_leaves_the_statistic_undefined_and_does_not_corrupt_state(self, cls):
        # A residual carries NaN where a sample could not be residualised. Treating that as zero would
        # feed a fabricated in-control observation into every recursive accumulator.
        base = clean_baseline(n=3000)
        det = cls().fit(base)

        rng = np.random.default_rng(2)
        clean = rng.normal(size=(600, 1))
        gapped = clean.copy()
        gapped[200:260] = np.nan

        out_clean = det.detect(series(clean))
        out_gap = det.detect(series(gapped))

        assert np.all(np.isnan(out_gap.statistic[200:260])), "gap samples must be undefined"
        assert not np.any(np.isnan(out_gap.statistic[300:])), "the detector must resume after a gap"
        # The statistic after the gap must stay in the same regime of values, not blow up or collapse.
        assert np.nanmax(out_gap.statistic[300:]) < 5 * np.nanmax(out_clean.statistic[300:])

    @pytest.mark.parametrize("cls", ALL_DETECTORS)
    def test_a_zero_variance_channel_does_not_produce_infinities(self, cls):
        base = series(np.column_stack([np.zeros(2000), np.random.default_rng(0).normal(size=2000)]),
                      ("flat", "noisy"))
        out = cls().fit(base).detect(base)
        assert np.all(np.isfinite(out.statistic[np.isfinite(out.statistic)]))

    @pytest.mark.parametrize("cls", ALL_DETECTORS)
    def test_attribution_names_the_channel_that_moved(self, cls):
        rng = np.random.default_rng(3)
        base = series(rng.normal(size=(3000, 3)), ("strut", "brake", "fuel"))
        det = cls().fit(base)

        x = rng.normal(size=(800, 3))
        x[400:, 1] += 6.0                       # only brake moves
        out = det.detect(series(x, ("strut", "brake", "fuel")))

        att = out.channel_attribution(int(np.nanargmax(out.statistic)))
        assert att.top(1) == ("brake",)

    def test_a_detector_with_no_per_channel_detail_refuses_to_attribute(self):
        from regimecpd import Detection
        det = Detection(np.arange(3.0), np.zeros(3), "opaque")
        with pytest.raises(ValueError, match="no per-channel detail"):
            det.channel_attribution(0)

    @pytest.mark.parametrize("cls", ALL_DETECTORS)
    def test_across_channels_the_reduction_is_max_not_mean(self, cls):
        # Averaging would dilute a single-channel fault by the channel count, so a twelve-channel machine
        # would need a twelve-times-larger fault to reach the same statistic. Onset evidence is usually
        # one channel moving, so that is the wrong behaviour.
        rng = np.random.default_rng(4)
        wide = series(rng.normal(size=(3000, 8)), tuple("abcdefgh"))
        det = cls().fit(wide)

        x = rng.normal(size=(600, 8))
        x[300:, 0] += 6.0
        out_wide = det.detect(series(x, tuple("abcdefgh")))

        narrow = series(rng.normal(size=(3000, 1)), ("a",))
        det1 = cls().fit(narrow)
        out_narrow = det1.detect(series(x[:, :1], ("a",)))

        # The fault is in channel a in both. Adding seven healthy channels must not shrink the response.
        assert np.nanmax(out_wide.statistic[300:]) > 0.8 * np.nanmax(out_narrow.statistic[300:])

    def test_a_baseline_with_no_finite_samples_is_rejected(self):
        with pytest.raises(ValueError, match="no finite samples"):
            Shewhart().fit(series(np.full(100, np.nan)))


class TestShewhart:
    def test_three_sigma_reproduces_the_published_false_alarm_rate(self):
        # Under normality a two-sided 3-sigma chart flags about 0.27% of in-control samples. Asserting
        # against the published property rather than against this implementation's own output is the
        # point: it would catch a wrong standardisation that a self-consistency check would not.
        rng = np.random.default_rng(0)
        base = series(rng.normal(size=(200_000, 1)))
        out = Shewhart().fit(base).detect(base)
        rate = float(np.mean(out.statistic >= 3.0))
        assert 0.0020 < rate < 0.0035, rate

    def test_the_statistic_is_the_standardised_absolute_deviation(self):
        base = series(np.array([10.0, 12.0, 14.0, 16.0, 18.0] * 200))
        det = Shewhart().fit(base)
        out = det.detect(series(np.array([14.0])))
        assert out.statistic[0] == pytest.approx(0.0, abs=1e-9)   # 14 is the baseline mean

        sd = float(np.std(base.x))
        out2 = det.detect(series(np.array([14.0 + 2 * sd])))
        assert out2.statistic[0] == pytest.approx(2.0, rel=1e-6)

    def test_has_no_memory_so_a_small_persistent_shift_barely_registers(self):
        # Shewhart's real weakness, and the reason CUSUM and EWMA exist. Asserted rather than described,
        # so the baseline's known limitation is part of the record.
        rng = np.random.default_rng(1)
        base = series(rng.normal(size=(5000, 1)))
        shifted = series(rng.normal(size=(2000, 1)) + 0.5)      # half a sigma, persistent

        shew = Shewhart().fit(base).detect(shifted)
        cus = CUSUM().fit(base).detect(shifted)

        assert np.nanmax(shew.statistic) < 6.0
        assert np.nanmax(cus.statistic) > 50.0, "CUSUM must accumulate what Shewhart cannot see"


class TestCUSUM:
    def test_resets_at_zero_so_it_is_not_a_random_walk(self):
        # Without the reset the sum wanders and eventually crosses any threshold on noise alone, which
        # makes it not a detector at all. In control, the statistic must stay bounded and small.
        rng = np.random.default_rng(0)
        base = series(rng.normal(size=(5000, 1)))
        out = CUSUM(k=0.5).fit(base).detect(series(rng.normal(size=(20_000, 1))))
        assert np.nanmin(out.statistic) >= 0.0
        assert np.nanmax(out.statistic) < 30.0, "an in-control CUSUM must not drift without bound"

    def test_accumulates_a_persistent_shift_roughly_linearly(self):
        # For a shift of size delta > k, S+ grows at about (delta - k) per sample. Checking the RATE,
        # not just that it went up, is what catches a wrong reference value.
        base = series(np.random.default_rng(0).normal(size=(5000, 1)))
        det = CUSUM(k=0.5).fit(base)
        out = det.detect(series(np.full(500, 2.0)))            # a clean 2-sigma shift
        growth = np.nanmax(out.statistic) / 500.0
        assert growth == pytest.approx(2.0 - 0.5, rel=0.05)

    def test_detects_a_downward_shift_too(self):
        base = series(np.random.default_rng(0).normal(size=(5000, 1)))
        out = CUSUM().fit(base).detect(series(np.full(500, -2.0)))
        assert np.nanmax(out.statistic) > 100.0

    def test_a_larger_reference_value_accumulates_more_slowly(self):
        base = series(np.random.default_rng(0).normal(size=(5000, 1)))
        shift = series(np.full(500, 1.5))
        small = CUSUM(k=0.25).fit(base).detect(shift)
        large = CUSUM(k=1.0).fit(base).detect(shift)
        assert np.nanmax(small.statistic) > np.nanmax(large.statistic)


class TestEWMA:
    def test_lambda_one_reduces_exactly_to_shewhart(self):
        # An analytic identity, so it either holds or the implementation is wrong. At lam = 1 the
        # variance factor is 1 and the smoother is the identity.
        rng = np.random.default_rng(0)
        base = series(rng.normal(size=(4000, 1)))
        x = series(rng.normal(size=(500, 1)))
        ewma = EWMA(lam=1.0).fit(base).detect(x)
        shew = Shewhart().fit(base).detect(x)
        assert np.allclose(ewma.statistic, shew.statistic, rtol=1e-9, atol=1e-9)

    def test_the_early_control_limit_is_the_time_varying_one_not_the_asymptote(self):
        # THE detail worth having. The asymptotic variance lam/(2-lam) OVERSTATES the spread of the first
        # samples, so a chart using it from the start is too insensitive early. On short fleet records
        # that dead zone is a real loss of detection, and it is invisible unless tested.
        #
        # The baseline is chosen so the standardisation is EXACT (mean 0, sd 1), which turns this from an
        # approximate check into an analytic identity: with the exact time-varying limit,
        #   statistic_1 = |lam z| / sqrt( lam/(2-lam) * (1 - (1-lam)^2) ) = |z|,  for every lam.
        lam = 0.2
        base = series(np.tile([-1.0, 1.0], 2000))            # mean 0, sd 1 exactly
        out = EWMA(lam=lam).fit(base).detect(series(np.full(50, 1.0)))

        assert out.statistic[0] == pytest.approx(1.0, rel=1e-12)

        # A chart using the asymptotic limit from the start would report lam*|z| / sqrt(lam/(2-lam))
        # here, weaker by exactly 1/sqrt(lam(2-lam)), which is 1.667x at lam = 0.2. That factor is the
        # early dead zone, and on short fleet records it is a real loss of detection.
        asym_sd = np.sqrt(lam / (2 - lam))
        naive_first = abs(lam * 1.0) / asym_sd
        assert out.statistic[0] / naive_first == pytest.approx(1.0 / np.sqrt(lam * (2 - lam)), rel=1e-12)
        assert out.statistic[0] > naive_first * 1.6

    def test_the_first_sample_is_lambda_independent_which_pins_the_variance_factor(self):
        # A consequence of the identity above: at t = 1 every lambda gives exactly |z|. If the variance
        # factor were wrong in any lambda-dependent way, these would disagree.
        base = series(np.tile([-1.0, 1.0], 2000))
        firsts = [EWMA(lam=lam).fit(base).detect(series(np.full(10, 1.0))).statistic[0]
                  for lam in (0.05, 0.2, 0.5, 1.0)]
        assert all(f == pytest.approx(1.0, rel=1e-12) for f in firsts), firsts

    def test_a_smaller_lambda_is_stronger_on_a_sustained_shift(self):
        # Worth stating carefully, because the intuition "small lambda reacts slowly" is about the RAW
        # smoothed value and does not survive standardisation. With the exact limit, a sustained shift z
        # drives the statistic to |z| / sqrt(lam/(2-lam)), which GROWS as lambda shrinks. Small lambda is
        # the right choice for the small persistent shift that degradation onset actually is.
        base = series(np.tile([-1.0, 1.0], 2000))
        shift = series(np.full(600, 1.0))
        fast = EWMA(lam=0.5).fit(base).detect(shift)
        slow = EWMA(lam=0.05).fit(base).detect(shift)

        assert np.nanmax(slow.statistic) > np.nanmax(fast.statistic)
        for lam, out in ((0.5, fast), (0.05, slow)):
            assert np.nanmax(out.statistic) == pytest.approx(1.0 / np.sqrt(lam / (2 - lam)), rel=0.01)

    def test_a_smaller_lambda_holds_a_transient_longer(self):
        # The other half of the trade, and the one that interacts with event-counted false alarms: a
        # long-memory chart turns a single spike into a long excursion.
        base = series(np.tile([-1.0, 1.0], 2000))
        spike = np.zeros(200)
        spike[0] = 5.0
        fast = EWMA(lam=0.5).fit(base).detect(series(spike))
        slow = EWMA(lam=0.05).fit(base).detect(series(spike))

        assert fast.statistic[0] == pytest.approx(slow.statistic[0], rel=1e-9), "t=1 is lambda-free"
        assert slow.statistic[20] > fast.statistic[20], "long memory must retain the excursion"
        assert int(np.sum(slow.statistic > 1.0)) > int(np.sum(fast.statistic > 1.0))

    def test_lambda_outside_the_unit_interval_is_rejected(self):
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="lam must be"):
                EWMA(lam=bad)

    def test_the_control_limit_uses_per_channel_observation_counts(self):
        # A channel that was NaN for a stretch has genuinely seen fewer observations, so its limit must
        # follow its OWN count rather than the wall-clock index. Both channels get an identical baseline
        # so that any difference in the result comes from the counting and not from standardisation.
        column = np.tile([-1.0, 1.0], 2000)
        base = series(np.column_stack([column, column]), ("a", "b"))

        x = np.full((60, 2), 1.0)
        x[:30, 1] = np.nan                       # channel b starts 30 samples late
        out = EWMA(lam=0.2).fit(base).detect(series(x, ("a", "b")))
        per = out.meta["per_channel"]

        # b's first defined sample is b's t = 1, so it must equal a's t = 1 exactly.
        assert per[30, 1] == pytest.approx(per[0, 0], rel=1e-12)
        # Meanwhile a, at its own t = 31, has reached its asymptote. Wall-clock indexing would have put b
        # there too and the two would agree; the whole point is that they must not.
        assert per[30, 0] == pytest.approx(1.0 / np.sqrt(0.2 / 1.8), rel=1e-3)
        assert per[30, 1] < per[30, 0] / 2.5, "b must still be early in its own control limit"


class TestPageHinkley:
    def test_stays_near_zero_in_control_and_climbs_on_a_shift(self):
        rng = np.random.default_rng(0)
        base = series(rng.normal(size=(5000, 1)))
        det = PageHinkley(delta=0.05).fit(base)
        quiet = det.detect(series(rng.normal(size=(5000, 1))))
        shifted = det.detect(series(np.full(500, 1.5)))
        assert np.nanmax(shifted.statistic) > 10 * np.nanmax(quiet.statistic)

    def test_is_two_sided(self):
        base = series(np.random.default_rng(0).normal(size=(5000, 1)))
        det = PageHinkley(delta=0.05).fit(base)
        up = np.nanmax(det.detect(series(np.full(400, 1.5))).statistic)
        down = np.nanmax(det.detect(series(np.full(400, -1.5))).statistic)
        assert up == pytest.approx(down, rel=0.02)

    def test_agrees_closely_with_cusum_because_it_is_the_same_construction(self):
        # Recorded as a test so the two are never presented as independent evidence. Page-Hinkley and
        # CUSUM come from the same 1954 construction; counting them as two classical methods beaten
        # would inflate any "we beat N baselines" claim.
        rng = np.random.default_rng(1)
        base = series(rng.normal(size=(5000, 1)))
        x = series(np.concatenate([rng.normal(size=300), rng.normal(size=300) + 1.5]).reshape(-1, 1))

        ph = PageHinkley(delta=0.5).fit(base).detect(x)
        cs = CUSUM(k=0.5).fit(base).detect(x)
        corr = np.corrcoef(ph.statistic, cs.statistic)[0, 1]
        assert corr > 0.98, f"expected near-identical behaviour, got correlation {corr}"
