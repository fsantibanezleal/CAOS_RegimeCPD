"""Tests for regime segmentation and the within-regime residual.

The central test in this file is `test_residual_is_regime_flat_while_the_raw_channel_is_not`. It measures
the thing the whole package claims to do, on data where the regime effect is built in and the answer is
known by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimecpd import RegimeLabels, Series
from regimecpd.regime import ContextScaler, DiscreteRegimes, KMeansRegimes, kmeans_inertia_sweep
from regimecpd.residual import RegimeResidualizer, make_arms


def synth_machine(n=3000, seed=0, fault_at=None, fault_size=3.0, n_regimes=4):
    """A machine with a genuinely emergent regime confound.

    Context is a load level and a grade. Both monitored channels respond to BOTH context variables, so
    the regime effect appears in the channels because the physics puts it there, not because a step was
    pasted in. That distinction is the whole point: a detector run on the raw channels should fire on
    regime changes, and it should do so for the same reason it would on a real machine.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(float(n))

    load = rng.integers(0, 2, size=n).astype(float)            # empty / loaded
    grade = rng.integers(0, n_regimes // 2, size=n).astype(float)   # flat / ramp

    # Channels move with context. These coefficients are large relative to the noise on purpose: that is
    # the regime confound, and on a real truck it dominates the fault signal too.
    strut = 100.0 + 40.0 * load + 8.0 * grade + rng.normal(0, 1.0, n)
    brake = 60.0 + 5.0 * load + 25.0 * grade + rng.normal(0, 1.0, n)

    if fault_at is not None:
        # The fault is a small persistent drift, which is what degradation onset looks like. It is much
        # smaller than the regime swing, which is what makes raw-channel detection hard.
        ramp = np.clip((np.arange(n) - fault_at) / 200.0, 0.0, 1.0)
        strut = strut + fault_size * ramp

    x = np.column_stack([load, grade, strut, brake])
    return Series(t, x, ("load", "grade", "strut", "brake"), unit_id=f"m{seed}")


def between_regime_share(values, labels):
    """Fraction of a channel's variance that lives BETWEEN regimes rather than within them.

    High means the channel is mostly reporting which regime the machine is in. That is exactly what a
    change detector would latch onto.
    """
    ok = np.isfinite(values) & (labels >= 0)
    v, lab = values[ok], labels[ok]
    if v.size < 2:
        return float("nan")
    grand = v.mean()
    between = sum(np.count_nonzero(lab == k) * (v[lab == k].mean() - grand) ** 2
                  for k in np.unique(lab))
    return float(between / np.sum((v - grand) ** 2))


class TestContextScaler:
    def test_a_constant_feature_gets_scale_one_not_zero(self):
        # Dividing by zero spread turns a feature that carries no information into infinities that then
        # dominate every distance computed over the context.
        scaler = ContextScaler().fit(np.column_stack([np.ones(10), np.arange(10.0)]))
        assert scaler.scale_[0] == 1.0
        assert np.all(np.isfinite(scaler.transform(np.column_stack([np.ones(3), np.arange(3.0)]))))

    def test_scale_comes_from_the_fit_not_from_the_data_being_transformed(self):
        # Re-deriving scale at transform time would make the monitored record define its own normal.
        scaler = ContextScaler().fit(np.arange(100.0).reshape(-1, 1))
        far = scaler.transform(np.array([[1000.0]]))
        assert far[0, 0] > 10, "an out-of-baseline value must stay far away after scaling"

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="before fit"):
            ContextScaler().transform(np.arange(3.0))


class TestDiscreteRegimes:
    def test_distinct_combinations_become_regimes(self):
        ctx = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0]], dtype=float)
        labels = DiscreteRegimes().fit_label(ctx)
        assert labels.n_regimes == 4
        assert labels.labels[0] == labels.labels[4]
        assert labels.coverage == 1.0

    def test_an_unseen_combination_is_minus_one_not_the_nearest(self):
        seg = DiscreteRegimes().fit(np.array([[0, 0], [1, 1]], dtype=float))
        labels = seg.label(np.array([[0, 0], [5, 5]], dtype=float))
        assert labels.labels.tolist() == [0, -1]

    def test_rounding_stops_float_noise_from_making_every_sample_its_own_regime(self):
        # Real "discrete" settings arrive as floats with noise in the last digits. Without rounding this
        # produces one regime per sample, which looks like it worked and is the same as no regimes.
        rng = np.random.default_rng(0)
        ctx = np.repeat(np.array([[0.0], [1.0]]), 50, axis=0) + rng.normal(0, 1e-6, (100, 1))
        assert DiscreteRegimes(decimals=3).fit_label(ctx).n_regimes == 2
        assert DiscreteRegimes(decimals=12).fit_label(ctx).n_regimes > 50


class TestKMeansRegimes:
    def test_recovers_well_separated_regimes(self):
        rng = np.random.default_rng(0)
        centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
        ctx = np.repeat(centres, 200, axis=0) + rng.normal(0, 0.3, (800, 2))
        labels = KMeansRegimes(n_regimes=4, seed=0).fit_label(ctx)
        # Every true group must map onto exactly one cluster.
        for g in range(4):
            block = labels.labels[g * 200:(g + 1) * 200]
            assert len(set(block.tolist())) == 1, f"group {g} was split across clusters"
        assert len(set(labels.labels.tolist())) == 4

    def test_is_deterministic_for_a_given_seed(self):
        # A segmentation that moves between runs makes every downstream number unreproducible.
        rng = np.random.default_rng(1)
        ctx = rng.normal(size=(500, 3))
        a = KMeansRegimes(n_regimes=5, seed=7).fit_label(ctx)
        b = KMeansRegimes(n_regimes=5, seed=7).fit_label(ctx)
        assert a.labels.tolist() == b.labels.tolist()

    def test_context_far_outside_the_baseline_is_marked_unassigned(self):
        # THE reason the novelty radius exists. Without it k-means hands a confident label to a machine
        # operating somewhere the baseline never went, and the residual that follows is meaningless
        # while looking exactly like every other residual.
        rng = np.random.default_rng(2)
        base = np.repeat(np.array([[0.0], [10.0]]), 300, axis=0) + rng.normal(0, 0.5, (600, 1))
        seg = KMeansRegimes(n_regimes=2, seed=0).fit(base)

        labels = seg.label(np.array([[0.0], [10.0], [500.0], [-400.0]]))
        assert labels.labels[0] >= 0 and labels.labels[1] >= 0
        assert labels.labels[2] == -1 and labels.labels[3] == -1

    def test_ordinary_baseline_variation_stays_inside_its_regime(self):
        # The radius must not be so tight that normal operation reads as novel; that would empty the
        # residual of most of its record and quietly disable the method.
        rng = np.random.default_rng(3)
        base = rng.normal(0, 1.0, (2000, 2))
        seg = KMeansRegimes(n_regimes=3, seed=0).fit(base)
        held_out = rng.normal(0, 1.0, (2000, 2))
        assert seg.label(held_out).coverage > 0.95

    def test_more_regimes_than_samples_is_rejected(self):
        with pytest.raises(ValueError, match="cannot fit"):
            KMeansRegimes(n_regimes=10).fit(np.zeros((3, 1)))

    def test_inertia_sweep_falls_with_k(self):
        rng = np.random.default_rng(4)
        ctx = rng.normal(size=(400, 2))
        sweep = kmeans_inertia_sweep(ctx, [1, 2, 4, 8], seed=0)
        values = [v for _, v in sweep]
        assert all(b <= a + 1e-9 for a, b in zip(values, values[1:]))

    def test_label_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="before fit"):
            KMeansRegimes(n_regimes=2).label(np.zeros((5, 1)))


class TestResidual:
    def test_residual_is_regime_flat_while_the_raw_channel_is_not(self):
        # THE central property. On healthy data with a strong emergent regime effect, most of a raw
        # channel's variance is BETWEEN regimes: the channel is largely reporting which regime the
        # machine is in, which is what a change detector on raw channels latches onto. After
        # residualising against the baseline, almost none of it is.
        base = synth_machine(n=4000, seed=0)
        mon = synth_machine(n=4000, seed=1)

        raw, res, labels = make_arms(base, mon, context_names=("load", "grade"), n_regimes=4)

        raw_share = between_regime_share(raw.channel("strut"), labels.labels)
        res_share = between_regime_share(res.channel("strut"), labels.labels)

        assert raw_share > 0.90, f"setup is wrong, the raw channel is not regime-dominated ({raw_share})"
        assert res_share < 0.02, f"the residual still carries the regime ({res_share})"

    def test_residual_is_standardised_within_each_regime(self):
        base = synth_machine(n=4000, seed=0)
        mon = synth_machine(n=4000, seed=1)
        _, res, labels = make_arms(base, mon, context_names=("load", "grade"), n_regimes=4)

        for k in np.unique(labels.labels[labels.labels >= 0]):
            v = res.channel("brake")[labels.labels == k]
            v = v[np.isfinite(v)]
            assert abs(v.mean()) < 0.15, f"regime {k} residual mean {v.mean()}"
            assert 0.8 < v.std() < 1.25, f"regime {k} residual sd {v.std()}"

    def test_a_fault_survives_residualisation(self):
        # Regime conditioning must remove the regime, not the signal. If the residual also flattened the
        # fault, the method would be useless in the most flattering possible way.
        base = synth_machine(n=4000, seed=0)
        mon = synth_machine(n=4000, seed=1, fault_at=2000, fault_size=6.0)
        _, res, labels = make_arms(base, mon, context_names=("load", "grade"), n_regimes=4)

        strut = res.channel("strut")
        before = np.nanmean(strut[:1800])
        after = np.nanmean(strut[2600:])
        assert after - before > 2.0, f"the fault was flattened away: {before} -> {after}"

    def test_fitting_on_the_faulty_record_leaks_and_shrinks_the_fault(self):
        # The leakage trap, demonstrated rather than asserted. Fitting the residual model on the record
        # being monitored folds the fault into the definition of normal. Nothing raises; the fault just
        # gets smaller, which is invisible in every output.
        mon = synth_machine(n=4000, seed=1, fault_at=2000, fault_size=6.0)
        base = synth_machine(n=4000, seed=0)

        _, honest, labels = make_arms(base, mon, context_names=("load", "grade"), n_regimes=4)

        ctx = mon.select(("load", "grade")).x
        seg = KMeansRegimes(n_regimes=4, seed=0).fit(ctx)
        leaked_labels = seg.label(ctx)
        leaked = RegimeResidualizer().fit_transform(mon.select(("strut", "brake")), leaked_labels)

        def step(values):
            return float(np.nanmean(values[2600:]) - np.nanmean(values[:1800]))

        assert step(leaked.r[:, 0]) < step(honest.channel("strut")) * 0.75, (
            "leakage should visibly shrink the fault; if it does not, this test no longer demonstrates it"
        )

    def test_unusable_regimes_become_nan_rather_than_a_plausible_number(self):
        # A regime with too few baseline samples has no trustworthy mean. Filling it from a pooled or
        # nearest-regime estimate would manufacture a plausible number for the one case the method
        # cannot speak to, and it would be indistinguishable from a real residual downstream.
        #
        # The baseline is deliberately UNBALANCED: one operating context is rare, which is the realistic
        # situation (a truck spends little time in an unusual configuration) and the one that produces a
        # regime with too few samples to characterise.
        rng = np.random.default_rng(9)
        n_common, n_rare = 900, 12
        ctx_common = np.zeros((n_common, 1))
        ctx_common[n_common // 2:] = 10.0            # two well-populated regimes
        ctx_rare = np.full((n_rare, 1), 25.0)        # one barely-seen regime
        ctx_b = np.vstack([ctx_common, ctx_rare])
        chan_b = 5.0 * ctx_b[:, 0] + rng.normal(0, 1.0, len(ctx_b))
        base = Series(np.arange(float(len(ctx_b))), chan_b, ("chan",))

        seg = KMeansRegimes(n_regimes=3, seed=0).fit(ctx_b)
        base_labels = seg.label(ctx_b)
        counts = np.bincount(base_labels.labels[base_labels.labels >= 0], minlength=3)
        assert counts.min() < 100 < counts.max(), f"setup: regimes should be unbalanced, got {counts}"

        resid = RegimeResidualizer(min_samples=100).fit(base, base_labels)
        assert len(resid.usable_) < 3, "setup: the rare regime should have been rejected"

        out = resid.transform(base, base_labels)
        unusable = out.regimes.labels < 0
        assert unusable.any(), "the rare regime's samples should be reported unusable"
        assert np.all(np.isnan(out.r[unusable])), "unusable samples must be NaN"
        assert np.all(np.isfinite(out.r[~unusable])), "usable samples must be finite"

    def test_no_usable_regime_at_all_raises_rather_than_returning_nothing(self):
        base = synth_machine(n=400, seed=0)
        ctx = base.select(("load", "grade")).x
        seg = KMeansRegimes(n_regimes=4, seed=0).fit(ctx)
        with pytest.raises(ValueError, match="min_samples"):
            RegimeResidualizer(min_samples=100_000).fit(base.select(("strut",)), seg.label(ctx))

    def test_linear_removes_within_regime_context_dependence_that_zscore_leaves(self):
        # When the regime is coarse, context still varies inside it and the channel still follows.
        # zscore cannot see that; the per-regime regression can.
        rng = np.random.default_rng(5)
        n = 4000
        t = np.arange(float(n))
        load = rng.integers(0, 2, n).astype(float)
        grade = rng.uniform(0, 10, n)                 # CONTINUOUS, so 2 regimes cannot capture it
        chan = 50.0 + 30.0 * load + 4.0 * grade + rng.normal(0, 0.5, n)
        series = Series(t, np.column_stack([load, grade, chan]), ("load", "grade", "chan"))

        base = Series(t[:2000], series.x[:2000], series.names)
        mon = Series(t[2000:], series.x[2000:], series.names)

        _, z_arm, labels = make_arms(base, mon, ("load", "grade"), n_regimes=2, method="zscore")
        _, l_arm, _ = make_arms(base, mon, ("load", "grade"), n_regimes=2, method="linear")

        g = mon.channel("grade")
        z_corr = abs(np.corrcoef(z_arm.channel("chan")[labels.labels >= 0], g[labels.labels >= 0])[0, 1])
        l_corr = abs(np.corrcoef(l_arm.channel("chan")[labels.labels >= 0], g[labels.labels >= 0])[0, 1])
        assert z_corr > 0.5, f"setup: zscore should leave the grade dependence ({z_corr})"
        assert l_corr < 0.15, f"linear should have removed it ({l_corr})"

    def test_linear_without_context_is_rejected(self):
        base = synth_machine(n=1000, seed=0)
        ctx = base.select(("load", "grade")).x
        labels = KMeansRegimes(n_regimes=2, seed=0).fit_label(ctx)
        with pytest.raises(ValueError, match="needs context"):
            RegimeResidualizer(method="linear").fit(base.select(("strut",)), labels)

    def test_unknown_method_is_rejected(self):
        base = synth_machine(n=1000, seed=0)
        labels = RegimeLabels(np.zeros(1000, dtype=int), 1, "test")
        with pytest.raises(ValueError, match="unknown method"):
            RegimeResidualizer(method="magic").fit(base.select(("strut",)), labels)

    def test_transform_with_different_channels_is_rejected(self):
        base = synth_machine(n=1000, seed=0)
        labels = RegimeLabels(np.zeros(1000, dtype=int), 1, "test")
        resid = RegimeResidualizer(min_samples=10).fit(base.select(("strut", "brake")), labels)
        with pytest.raises(ValueError, match="fitted on channels"):
            resid.transform(base.select(("strut",)), labels)


class TestMakeArms:
    def test_both_arms_share_channels_time_and_length(self):
        # The comparison is only meaningful if the two arms differ in exactly one thing.
        base, mon = synth_machine(n=2000, seed=0), synth_machine(n=2000, seed=1)
        raw, res, _ = make_arms(base, mon, ("load", "grade"), n_regimes=4)
        assert raw.names == res.names == ("strut", "brake")
        assert raw.n == res.n == mon.n
        assert np.array_equal(raw.t, res.t)

    def test_context_channels_are_excluded_from_monitoring_by_default(self):
        # Their residual is zero by construction, so monitoring them would add a channel that can never
        # carry a fault and would dilute every multivariate statistic.
        base, mon = synth_machine(n=2000, seed=0), synth_machine(n=2000, seed=1)
        raw, _, _ = make_arms(base, mon, ("load", "grade"), n_regimes=4)
        assert "load" not in raw.names and "grade" not in raw.names

    def test_the_raw_arm_is_left_untouched(self):
        # Standardising the raw arm would be a one-regime correction, which blurs the contrast being
        # measured and quietly moves the baseline toward the residual arm.
        base, mon = synth_machine(n=2000, seed=0), synth_machine(n=2000, seed=1)
        raw, _, _ = make_arms(base, mon, ("load", "grade"), n_regimes=4)
        assert np.allclose(raw.channel("strut"), mon.channel("strut"))

    def test_discrete_route_needs_no_n_regimes_and_clustered_route_does(self):
        base, mon = synth_machine(n=2000, seed=0), synth_machine(n=2000, seed=1)
        _, res, labels = make_arms(base, mon, ("load", "grade"), discrete=True)
        assert labels.method == "discrete"
        assert res.n == mon.n
        with pytest.raises(ValueError, match="n_regimes is required"):
            make_arms(base, mon, ("load", "grade"))

    def test_monitoring_nothing_is_rejected(self):
        base, mon = synth_machine(n=500, seed=0), synth_machine(n=500, seed=1)
        with pytest.raises(ValueError, match="no channels left"):
            make_arms(base, mon, ("load", "grade", "strut", "brake"), n_regimes=2)
