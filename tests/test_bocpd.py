"""Tests for Bayesian online changepoint detection.

The load-bearing tests are the ones that check the run-length posterior itself rather than the reduced
statistic. A detector that fires in roughly the right place while carrying a meaningless posterior would
pass a detection test and fail the reason this method was chosen.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimecpd import Series
from regimecpd.bocpd import BOCPD, StudentTUPM


def series(values, names=None, t=None):
    v = np.asarray(values, dtype=float)
    v = v.reshape(-1, 1) if v.ndim == 1 else v
    names = names or tuple("abcdefgh"[: v.shape[1]])
    return Series(np.arange(len(v), dtype=float) if t is None else t, v, names)


def step_series(n=600, change_at=300, size=4.0, seed=0, d=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    x[change_at:] += size
    return series(x)


class TestRunLengthPosterior:
    def test_the_posterior_is_a_distribution_at_every_step(self):
        out = BOCPD(hazard_rate=100.0).detect(step_series())
        post = out.meta["run_length_posterior"]
        mass = post.sum(axis=1)
        assert np.allclose(mass, 1.0, atol=1e-6), f"mass ranges {mass.min()} to {mass.max()}"
        assert np.all(post >= 0.0)

    def test_the_run_length_grows_with_time_while_nothing_changes(self):
        # In a stationary stretch the most probable run length should track elapsed time. If it does not,
        # the recursion is not accumulating evidence and the posterior means nothing.
        out = BOCPD(hazard_rate=500.0).detect(series(np.random.default_rng(1).normal(size=(400, 1))))
        modes = np.argmax(out.meta["run_length_posterior"], axis=1)
        assert modes[50] > 20, modes[50]
        assert modes[300] > modes[100] > modes[50]

    def test_the_run_length_collapses_at_a_change(self):
        # THE behaviour that distinguishes this from a threshold. After a step, the mass moves back to a
        # short run: the model now believes it has only seen a few observations of the current state.
        change_at = 300
        out = BOCPD(hazard_rate=200.0).detect(step_series(change_at=change_at, size=6.0))
        modes = np.argmax(out.meta["run_length_posterior"], axis=1)
        assert modes[change_at - 5] > 200, "setup: a long run should have accumulated"
        assert np.min(modes[change_at:change_at + 20]) < 10, "the run length must collapse"

    def test_the_posterior_recovers_and_regrows_after_the_change(self):
        change_at = 300
        out = BOCPD(hazard_rate=200.0).detect(step_series(n=800, change_at=change_at, size=6.0))
        modes = np.argmax(out.meta["run_length_posterior"], axis=1)
        assert modes[750] > 300, "the run length must start accumulating again"

    def test_the_posterior_is_wider_for_a_gradual_change_than_an_abrupt_one(self):
        # The honest consequence of the independent-parameters assumption: a slow drift never gives
        # strong evidence for a changepoint at any particular instant, and the method says so through a
        # diffuse posterior rather than by being quietly wrong.
        rng = np.random.default_rng(2)
        n = 800
        abrupt = rng.normal(size=(n, 1)).copy()
        abrupt[400:] += 5.0
        gradual = rng.normal(size=(n, 1)).copy()
        gradual[:, 0] += np.clip((np.arange(n) - 400) / 200.0, 0, 1) * 5.0

        det = BOCPD(hazard_rate=200.0, statistic="surprise")
        peak_abrupt = np.nanmax(det.detect(series(abrupt)).statistic[400:460])
        peak_gradual = np.nanmax(det.detect(series(gradual)).statistic[400:600])
        assert peak_abrupt > peak_gradual, (peak_abrupt, peak_gradual)


class TestDetection:
    def test_the_statistic_peaks_near_the_true_change(self):
        change_at = 300
        out = BOCPD(hazard_rate=200.0).detect(step_series(change_at=change_at, size=6.0))
        peak = int(np.nanargmax(out.statistic))
        assert 0 <= peak - change_at <= 8, f"peak at {peak}, change at {change_at}"

    def test_the_statistic_is_larger_when_more_anomalous(self):
        # The orientation invariant, checked for both usable statistics.
        for stat in ("short_run", "surprise"):
            det = BOCPD(hazard_rate=200.0, statistic=stat)
            quiet = det.detect(series(np.random.default_rng(3).normal(size=(600, 1))))
            changed = det.detect(step_series(size=6.0, seed=3))
            assert np.nanmax(changed.statistic) > np.nanmax(quiet.statistic), stat

    def test_a_bigger_step_gives_a_more_confident_changepoint(self):
        det = BOCPD(hazard_rate=200.0)
        small = det.detect(step_series(size=1.0, seed=4))
        large = det.detect(step_series(size=8.0, seed=4))
        assert np.nanmax(large.statistic) > np.nanmax(small.statistic)

    def test_every_statistic_is_always_available_in_meta(self):
        out = BOCPD(statistic="short_run").detect(step_series())
        for key in ("short_run_probability", "changepoint_probability", "surprise"):
            assert key in out.meta
        assert np.allclose(out.statistic, out.meta["short_run_probability"], equal_nan=True)

    def test_the_short_run_statistic_is_a_probability(self):
        out = BOCPD().detect(step_series())
        finite = out.statistic[np.isfinite(out.statistic)]
        assert finite.min() >= 0.0 and finite.max() <= 1.0 + 1e-9

    def test_the_changepoint_probability_is_IDENTICALLY_the_hazard_rate(self):
        # Recorded so it is never rediscovered. Under a constant hazard both branches of the recursion
        # weight the same predictive, so the hazard factors straight out and p(r_t = 0 | x_1:t) = H at
        # every step, on every dataset. Reading the paper's headline quantity as a detection statistic
        # therefore gives a flat line, with a perfectly sensible-looking posterior plotted beside it.
        for rate in (100.0, 250.0, 1000.0):
            out = BOCPD(hazard_rate=rate, statistic="changepoint").detect(step_series(size=8.0))
            finite = out.statistic[np.isfinite(out.statistic)]
            assert np.allclose(finite, 1.0 / rate, rtol=1e-9), (rate, finite.min(), finite.max())

    def test_the_short_run_probability_is_the_informative_version(self):
        # The same idea, made usable: after a real change, mass collapses onto short runs and stays there
        # for several steps, which is where the evidence actually arrives.
        out = BOCPD(hazard_rate=200.0, statistic="short_run").detect(step_series(size=8.0))
        assert np.nanmax(out.statistic[300:340]) > 0.9
        assert np.nanmedian(out.statistic[100:290]) < 0.2

    def test_surprise_is_unbounded_and_spikes_at_the_change(self):
        # The reason the second statistic exists: the changepoint probability saturates near 1, which
        # leaves an alarm-budget curve no resolution at the sensitive end.
        out = BOCPD(hazard_rate=200.0, statistic="surprise").detect(step_series(size=8.0))
        assert np.nanmax(out.statistic) > 5.0
        assert np.nanmax(out.statistic[290:320]) == np.nanmax(out.statistic)

    def test_unknown_statistic_is_rejected(self):
        with pytest.raises(ValueError, match="unknown statistic"):
            BOCPD(statistic="magic")

    def test_an_implausible_hazard_rate_is_rejected(self):
        # hazard_rate is a prior mean run length in SAMPLES. A value at or below 1 means a change every
        # sample, which is not a prior anyone means to express.
        with pytest.raises(ValueError, match="prior mean run length"):
            BOCPD(hazard_rate=0.004)


class TestMultivariate:
    def test_channels_share_one_run_length_posterior(self):
        # What the product needs: one onset time for the machine, not one per channel.
        out = BOCPD(hazard_rate=200.0).detect(step_series(d=3, size=4.0))
        assert out.meta["run_length_posterior"].ndim == 2
        assert np.allclose(out.meta["run_length_posterior"].sum(axis=1), 1.0, atol=1e-6)

    def test_a_change_in_one_channel_of_several_is_still_found(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(700, 4))
        x[350:, 2] += 6.0                       # only the third channel steps
        out = BOCPD(hazard_rate=200.0).detect(series(x))
        peak = int(np.nanargmax(out.statistic))
        assert 0 <= peak - 350 <= 10, peak

    def test_more_channels_moving_gives_a_more_confident_changepoint(self):
        rng = np.random.default_rng(6)
        base = rng.normal(size=(700, 4))
        one, four = base.copy(), base.copy()
        one[350:, 0] += 3.0
        four[350:, :] += 3.0
        det = BOCPD(hazard_rate=200.0)
        assert np.nanmax(det.detect(series(four)).statistic) >= np.nanmax(det.detect(series(one)).statistic)


class TestRobustness:
    def test_nan_samples_are_skipped_rather_than_treated_as_observations(self):
        # Updating on an incomplete observation would be inventing data. Skipping keeps the recursion
        # exact for the samples that do exist.
        rng = np.random.default_rng(7)
        x = rng.normal(size=(600, 1))
        x[500:] += 6.0
        gapped = x.copy()
        gapped[100:150] = np.nan

        out = BOCPD(hazard_rate=200.0).detect(series(gapped))
        assert np.all(np.isnan(out.statistic[100:150]))
        assert not np.any(np.isnan(out.statistic[200:]))
        peak = int(np.nanargmax(out.statistic))
        assert 0 <= peak - 500 <= 10, f"the change must still be found, peak at {peak}"

    def test_pruning_keeps_the_posterior_a_distribution(self):
        out = BOCPD(hazard_rate=100.0, max_runs=40, prune_threshold=1e-3).detect(
            series(np.random.default_rng(8).normal(size=(1200, 1))))
        post = out.meta["run_length_posterior"]
        assert post.shape[1] == 41
        assert np.allclose(post.sum(axis=1), 1.0, atol=1e-3)

    def test_a_tight_run_cap_still_finds_the_change(self):
        out = BOCPD(hazard_rate=200.0, max_runs=50).detect(step_series(size=6.0))
        peak = int(np.nanargmax(out.statistic))
        assert 0 <= peak - 300 <= 10, peak

    def test_an_extreme_outlier_does_not_destroy_the_recursion(self):
        # The Student-t predictive should absorb this without the joint underflowing to nothing.
        rng = np.random.default_rng(9)
        x = rng.normal(size=(400, 1))
        x[200] = 1e6
        out = BOCPD(hazard_rate=200.0).detect(series(x))
        post = out.meta["run_length_posterior"]
        assert np.allclose(post[250:].sum(axis=1), 1.0, atol=1e-6), "the recursion must survive"
        assert np.all(np.isfinite(out.statistic[out.meta["warmup"]:])), "only the warm-up may be undefined"

    def test_a_constant_series_does_not_produce_nan(self):
        out = BOCPD(hazard_rate=200.0).detect(series(np.zeros(300)))
        assert np.all(np.isfinite(out.statistic[out.meta["warmup"]:]))

    def test_the_warmup_is_undefined_rather_than_a_spuriously_high_score(self):
        # Without it, p(r <= w) is structurally 1 at the start of every record and the largest value of
        # the statistic on ANY record lands at index 0, which is the record beginning and not a fault.
        det = BOCPD(hazard_rate=200.0, warmup=25)
        out = det.detect(step_series(size=6.0))
        assert np.all(np.isnan(out.statistic[:25]))
        assert np.isfinite(out.statistic[25])
        assert int(np.nanargmax(out.statistic)) > 25


class TestUPM:
    def test_the_precomputed_gamma_terms_are_exact_not_an_approximation(self):
        # The precomputation is a performance decision that must not cost accuracy: after r observations
        # the shape parameter is alpha0 + r/2, determined by the run length alone.
        import math as m
        upm = StudentTUPM(alpha0=1.0).start(d=1, max_runs=20)
        for r in (0, 1, 5, 20):
            nu = 2.0 * (1.0 + r / 2.0)
            expected = m.lgamma((nu + 1) / 2) - m.lgamma(nu / 2) - 0.5 * m.log(nu * m.pi)
            assert upm._const[r] == pytest.approx(expected, rel=1e-14)

    def test_the_predictive_has_heavier_tails_than_a_gaussian(self):
        # The reason for the conjugate predictive rather than a plugged-in variance: with few
        # observations behind a run, a slightly unusual sample must not read as a changepoint.
        upm = StudentTUPM().start(d=1, max_runs=10)
        far = float(upm.log_pred_prob(np.array([4.0]))[0])
        gaussian = -0.5 * np.log(2 * np.pi) - 0.5 * 4.0 ** 2
        assert far > gaussian, "the Student-t must assign more mass far from the mean than a Gaussian"

    def test_updating_shifts_the_predictive_toward_the_observations(self):
        upm = StudentTUPM().start(d=1, max_runs=10)
        for _ in range(5):
            upm.update(np.array([3.0]))
        # The longest-running hypothesis has seen five observations at 3.0 and should now prefer 3 to 0.
        assert upm.log_pred_prob(np.array([3.0]))[-1] > upm.log_pred_prob(np.array([0.0]))[-1]
