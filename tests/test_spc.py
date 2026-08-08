"""Tests for PCA-based multivariate SPC.

The two statistics are tested against faults that each is supposed to catch and the other is supposed to
miss, which is the only way to show they are not two spellings of one thing. Contributions are tested by
injecting a fault into a known variable and requiring that variable back.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from regimecpd import Series
from regimecpd.spc import PCAMonitor, chi2_quantile, normal_quantile


def correlated_baseline(n=5000, seed=0):
    """Four channels with real correlation structure, so the residual subspace is not trivial.

    a and b move together; c and d move together. That structure is what SPE monitors, and breaking it is
    a fault no single channel's own range would reveal.
    """
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    x = np.column_stack([
        f1 + 0.10 * rng.normal(size=n),
        f1 + 0.10 * rng.normal(size=n),
        f2 + 0.10 * rng.normal(size=n),
        f2 + 0.10 * rng.normal(size=n),
    ])
    return Series(np.arange(float(n)), x, ("a", "b", "c", "d"))


class TestQuantiles:
    def test_normal_quantile_matches_published_values(self):
        # Checked against standard published quantiles, not against this implementation's own output.
        for p, expected in [(0.975, 1.959963985), (0.95, 1.644853627), (0.99, 2.326347874),
                            (0.5, 0.0), (0.025, -1.959963985), (0.999, 3.090232306)]:
            assert normal_quantile(p) == pytest.approx(expected, abs=1e-9)

    def test_normal_quantile_is_symmetric_and_monotone(self):
        assert normal_quantile(0.3) == pytest.approx(-normal_quantile(0.7), abs=1e-12)
        values = [normal_quantile(p) for p in (0.01, 0.1, 0.4, 0.6, 0.9, 0.99)]
        assert all(b > a for a, b in zip(values, values[1:]))

    def test_normal_quantile_rejects_values_outside_the_open_unit_interval(self):
        for bad in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError, match="must be in"):
                normal_quantile(bad)

    def test_chi2_quantile_is_close_to_published_values(self):
        # Wilson-Hilferty, so an approximation. Published 95% points: 11.070 (dof 5), 18.307 (dof 10),
        # 31.410 (dof 20). Tolerance is stated at the accuracy the approximation actually has, rather
        # than being loosened until it passes.
        for dof, expected in [(5, 11.0705), (10, 18.3070), (20, 31.4104)]:
            assert chi2_quantile(0.95, dof) == pytest.approx(expected, rel=0.005)

    def test_chi2_quantile_rejects_zero_degrees_of_freedom(self):
        with pytest.raises(ValueError, match="at least 1"):
            chi2_quantile(0.95, 0)


class TestFit:
    def test_component_count_follows_the_variance_target(self):
        base = correlated_baseline()
        # Two latent factors dominate, so a 95% target should retain about two of four components.
        monitor = PCAMonitor(variance_target=0.95).fit(base)
        assert monitor.n_components_ == 2
        assert monitor.meta_["explained_variance"] >= 0.95

    def test_an_explicit_component_count_is_honoured(self):
        monitor = PCAMonitor(n_components=3).fit(correlated_baseline())
        assert monitor.n_components_ == 3

    def test_a_zero_variance_direction_is_pushed_into_the_residual_subspace(self):
        # Retaining it would divide by a numerically zero eigenvalue in T-squared and turn rounding noise
        # into an unbounded statistic. SPE handles that direction correctly instead.
        rng = np.random.default_rng(0)
        f = rng.normal(size=3000)
        x = np.column_stack([f, f, f])          # rank 1: two directions carry no variance at all
        base = Series(np.arange(3000.0), x, ("a", "b", "c"))
        monitor = PCAMonitor(n_components=3).fit(base)
        assert monitor.n_components_ == 1
        assert np.all(np.isfinite(monitor.t2(base)))

    def test_a_baseline_with_no_variance_at_all_is_rejected(self):
        base = Series(np.arange(100.0), np.ones((100, 2)), ("a", "b"))
        with pytest.raises(ValueError, match="non-zero variance"):
            PCAMonitor().fit(base)

    def test_a_baseline_with_too_few_complete_samples_is_rejected(self):
        x = np.full((100, 2), np.nan)
        x[0] = [1.0, 2.0]
        with pytest.raises(ValueError, match="fewer than two complete"):
            PCAMonitor().fit(Series(np.arange(100.0), x, ("a", "b")))

    def test_autoscaling_stops_the_largest_unit_from_deciding_the_components(self):
        # Without autoscaling the component structure reports the instrumentation rather than the machine.
        rng = np.random.default_rng(1)
        shared = rng.normal(size=4000)
        x = np.column_stack([shared * 1000.0 + rng.normal(0, 100, 4000),   # huge units
                             shared + 0.1 * rng.normal(size=4000),
                             rng.normal(size=4000)])
        base = Series(np.arange(4000.0), x, ("big", "small", "noise"))
        monitor = PCAMonitor(n_components=1).fit(base)
        loading = np.abs(monitor.loadings_[:, 0])
        # The first component must load on BOTH correlated channels, not just the one in big units.
        assert loading[1] > 0.5 * loading[0], f"autoscaling failed, loadings {loading}"

    def test_using_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="before fit"):
            PCAMonitor().t2(correlated_baseline())

    def test_unknown_statistic_is_rejected(self):
        with pytest.raises(ValueError, match="unknown statistic"):
            PCAMonitor(statistic="magic")


class TestT2AndSPEAreDifferentInstruments:
    def test_an_in_model_excursion_raises_t2_and_leaves_spe_flat(self):
        # A machine pushed far along a direction it already varies in. In-model, but extreme.
        base = correlated_baseline()
        monitor = PCAMonitor(variance_target=0.95).fit(base)

        rng = np.random.default_rng(5)
        f1 = rng.normal(size=400) + 6.0                 # push the FIRST latent factor
        f2 = rng.normal(size=400)
        x = np.column_stack([f1 + 0.10 * rng.normal(size=400), f1 + 0.10 * rng.normal(size=400),
                             f2 + 0.10 * rng.normal(size=400), f2 + 0.10 * rng.normal(size=400)])
        faulty = Series(np.arange(400.0), x, ("a", "b", "c", "d"))

        base_t2, base_spe = monitor.t2(base), monitor.spe(base)
        assert np.median(monitor.t2(faulty)) > 20 * np.median(base_t2)
        assert np.median(monitor.spe(faulty)) < 3 * np.median(base_spe), "SPE should barely move"

    def test_a_broken_correlation_raises_spe_and_leaves_t2_flat(self):
        # THE fault T-squared cannot see. a and b used to move together; now b is independent. Every
        # channel stays inside its usual range, so nothing about the individual magnitudes is unusual.
        base = correlated_baseline()
        monitor = PCAMonitor(variance_target=0.95).fit(base)

        rng = np.random.default_rng(6)
        f1, f2 = rng.normal(size=400), rng.normal(size=400)
        x = np.column_stack([f1 + 0.10 * rng.normal(size=400),
                             rng.normal(size=400),          # b decoupled from a, same marginal spread
                             f2 + 0.10 * rng.normal(size=400),
                             f2 + 0.10 * rng.normal(size=400)])
        faulty = Series(np.arange(400.0), x, ("a", "b", "c", "d"))

        assert np.median(monitor.spe(faulty)) > 20 * np.median(monitor.spe(base))
        assert np.median(monitor.t2(faulty)) < 3 * np.median(monitor.t2(base)), "T2 should barely move"


class TestControlLimits:
    def test_the_spe_limit_flags_about_alpha_of_in_control_samples(self):
        # Jackson-Mudholkar, checked against the property it claims rather than against itself.
        base = correlated_baseline(n=20000, seed=2)
        monitor = PCAMonitor(variance_target=0.95).fit(base)
        held_out = correlated_baseline(n=20000, seed=3)
        for alpha in (0.05, 0.01):
            rate = float(np.mean(monitor.spe(held_out) > monitor.spe_limit(alpha)))
            assert 0.2 * alpha < rate < 4 * alpha, f"alpha={alpha} gave {rate}"

    def test_the_t2_limit_flags_roughly_alpha_of_in_control_samples(self):
        base = correlated_baseline(n=20000, seed=4)
        monitor = PCAMonitor(variance_target=0.95).fit(base)
        held_out = correlated_baseline(n=20000, seed=5)
        rate = float(np.mean(monitor.t2(held_out) > monitor.t2_limit(0.05)))
        assert 0.01 < rate < 0.12, rate

    def test_the_spe_limit_is_zero_when_no_residual_subspace_remains(self):
        base = correlated_baseline()
        monitor = PCAMonitor(n_components=4).fit(base)
        assert monitor.spe_limit(0.05) == 0.0

    def test_a_tighter_alpha_gives_a_higher_limit(self):
        monitor = PCAMonitor(variance_target=0.95).fit(correlated_baseline())
        assert monitor.spe_limit(0.01) > monitor.spe_limit(0.05)
        assert monitor.t2_limit(0.01) > monitor.t2_limit(0.05)


class TestDetect:
    def test_the_reported_statistic_follows_the_configuration(self):
        base = correlated_baseline()
        for stat, method in [("spe", "pca-spe"), ("t2", "pca-t2"), ("combined", "pca-combined")]:
            out = PCAMonitor(statistic=stat).fit(base).detect(base)
            assert out.method == method

    def test_both_statistics_are_always_available_in_meta(self):
        # So a caller can look at the other subspace without recomputing the projection, and so a result
        # can never be reported without the half that would have contradicted it.
        out = PCAMonitor(statistic="spe").fit(correlated_baseline()).detect(correlated_baseline())
        assert "t2" in out.meta and "spe" in out.meta
        assert len(out.meta["t2"]) == len(out.meta["spe"])

    def test_incomplete_samples_are_undefined_rather_than_imputed(self):
        base = correlated_baseline()
        monitor = PCAMonitor().fit(base)
        x = base.x[:200].copy()
        x[50:60, 1] = np.nan
        out = monitor.detect(Series(np.arange(200.0), x, base.names))
        assert np.all(np.isnan(out.statistic[50:60]))
        assert np.all(np.isfinite(out.statistic[100:]))

    def test_combined_puts_the_two_statistics_on_a_comparable_scale(self):
        base = correlated_baseline()
        monitor = PCAMonitor(statistic="combined").fit(base)
        out = monitor.detect(base)
        expected = np.fmax(out.meta["t2"] / out.meta["t2_limit"], out.meta["spe"] / out.meta["spe_limit"])
        assert np.allclose(out.statistic, expected, equal_nan=True)


class TestContributions:
    def test_spe_contributions_decompose_spe_exactly(self):
        # An identity, so it either holds or the implementation is wrong.
        base = correlated_baseline()
        monitor = PCAMonitor(variance_target=0.95).fit(base)
        spe = monitor.spe(base)
        for i in (0, 137, 4999):
            att = monitor.contributions(base, i, kind="spe")
            assert float(att.scores.sum()) == pytest.approx(spe[i], rel=1e-10)

    def test_spe_contributions_name_the_decoupled_channel(self):
        base = correlated_baseline()
        monitor = PCAMonitor(variance_target=0.95).fit(base)

        rng = np.random.default_rng(7)
        f1, f2 = rng.normal(size=200), rng.normal(size=200)
        x = np.column_stack([f1, f1 + 4.0, f2, f2])       # b breaks away from a by a clear margin
        faulty = Series(np.arange(200.0), x, ("a", "b", "c", "d"))

        idx = int(np.nanargmax(monitor.spe(faulty)))
        assert monitor.contributions(faulty, idx, kind="spe").top(2)[0] in {"a", "b"}

    def test_t2_contributions_name_the_channel_that_was_pushed(self):
        base = correlated_baseline()
        monitor = PCAMonitor(variance_target=0.95).fit(base)

        rng = np.random.default_rng(8)
        f1 = rng.normal(size=200) + 8.0
        f2 = rng.normal(size=200)
        x = np.column_stack([f1, f1, f2, f2])
        faulty = Series(np.arange(200.0), x, ("a", "b", "c", "d"))

        att = monitor.contributions(faulty, int(np.nanargmax(monitor.t2(faulty))), kind="t2")
        assert set(att.top(2)) == {"a", "b"}, att.ranked()

    def test_contributions_smear_onto_a_correlated_healthy_channel(self):
        # The caveat, DEMONSTRATED rather than only written down. Only `a` is faulted, yet `b` picks up a
        # substantial contribution purely because it is correlated with `a`. This is why a contribution
        # narrows a candidate set and does not identify a cause.
        base = correlated_baseline()
        monitor = PCAMonitor(variance_target=0.95).fit(base)

        rng = np.random.default_rng(9)
        f1, f2 = rng.normal(size=200), rng.normal(size=200)
        x = np.column_stack([f1 + 10.0, f1, f2, f2])       # ONLY a is faulted
        faulty = Series(np.arange(200.0), x, ("a", "b", "c", "d"))

        att = monitor.contributions(faulty, int(np.nanargmax(monitor.t2(faulty))), kind="t2")
        ranked = dict(att.ranked())
        assert ranked["b"] > 0.2 * ranked["a"], (
            "if b no longer smears, the caveat in the docs needs revisiting"
        )

    def test_an_incomplete_sample_has_no_decomposition(self):
        base = correlated_baseline()
        monitor = PCAMonitor().fit(base)
        x = base.x[:10].copy()
        x[3, 0] = np.nan
        with pytest.raises(ValueError, match="not complete"):
            monitor.contributions(Series(np.arange(10.0), x, base.names), 3)

    def test_unknown_contribution_kind_is_rejected(self):
        monitor = PCAMonitor().fit(correlated_baseline())
        with pytest.raises(ValueError, match="unknown kind"):
            monitor.contributions(correlated_baseline(), 0, kind="magic")


def test_normal_quantile_agrees_with_erfc_across_the_range():
    # An independent check of the approximation over its whole domain, using a different function.
    for p in (1e-6, 1e-3, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1 - 1e-6):
        x = normal_quantile(p)
        assert 0.5 * math.erfc(-x / math.sqrt(2)) == pytest.approx(p, rel=1e-9, abs=1e-12)
