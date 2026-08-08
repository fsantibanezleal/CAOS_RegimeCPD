"""Tests for the healthy-only novelty detectors.

The most important test here is the ORIENTATION one, run for every model. scikit-learn returns higher
scores for more normal samples and this package returns higher for more anomalous, so a missed flip
produces a detector that scores exactly as badly as it should score well. On a benchmark table that reads
as "this approach does not work for this problem" rather than as a bug, which is the worst possible way
for it to fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimecpd import Series
from regimecpd.novelty import AutoencoderDetector, IsolationForestDetector, OneClassSVMDetector

sklearn = pytest.importorskip("sklearn")

SK_MODELS = [
    lambda **kw: IsolationForestDetector(n_estimators=60, seed=0, **kw),
    lambda **kw: OneClassSVMDetector(nu=0.05, **kw),
]


def series(values, names=None):
    v = np.asarray(values, dtype=float)
    v = v.reshape(-1, 1) if v.ndim == 1 else v
    names = names or tuple("abcdefgh"[: v.shape[1]])
    return Series(np.arange(len(v), dtype=float), v, names)


def healthy(n=1500, d=3, seed=0):
    """Correlated healthy operation: three channels driven by one latent factor."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=n)
    return series(np.column_stack([f + 0.2 * rng.normal(size=n) for _ in range(d)]))


class TestOrientation:
    @pytest.mark.parametrize("make", SK_MODELS, ids=["isolation-forest", "one-class-svm"])
    def test_the_statistic_is_larger_for_anomalous_samples(self, make):
        # THE test. sklearn scores higher for more NORMAL; this package needs the opposite.
        base = healthy(seed=0)
        det = make().fit(base)

        quiet = det.detect(healthy(n=400, seed=1)).statistic
        rng = np.random.default_rng(2)
        odd = det.detect(series(rng.normal(0, 1, (400, 3)) + 6.0)).statistic

        assert np.nanmedian(odd) > np.nanmedian(quiet), (
            "the sign flip is missing: this detector would score exactly as badly as it should score well"
        )

    @pytest.mark.parametrize("make", SK_MODELS, ids=["isolation-forest", "one-class-svm"])
    def test_a_broken_correlation_scores_above_normal(self, make):
        # The interesting fault: every channel stays inside its usual range, only the relationship
        # between them breaks. A per-channel threshold cannot see this at all.
        base = healthy(seed=0)
        det = make().fit(base)

        rng = np.random.default_rng(3)
        n = 400
        f = rng.normal(size=n)
        broken = series(np.column_stack([f + 0.2 * rng.normal(size=n),
                                         rng.normal(size=n),           # decoupled
                                         f + 0.2 * rng.normal(size=n)]))
        assert np.nanmedian(det.detect(broken).statistic) > \
            np.nanmedian(det.detect(healthy(n=n, seed=4)).statistic)


class TestSharedContract:
    @pytest.mark.parametrize("make", SK_MODELS, ids=["isolation-forest", "one-class-svm"])
    def test_detect_before_fit_raises(self, make):
        with pytest.raises(RuntimeError, match="before fit"):
            make().detect(healthy())

    @pytest.mark.parametrize("make", SK_MODELS, ids=["isolation-forest", "one-class-svm"])
    def test_scoring_different_channels_is_rejected(self, make):
        det = make().fit(healthy(d=3))
        with pytest.raises(ValueError, match="fitted on channels"):
            det.detect(healthy(d=2))

    @pytest.mark.parametrize("make", SK_MODELS, ids=["isolation-forest", "one-class-svm"])
    def test_incomplete_samples_are_undefined_rather_than_imputed(self, make):
        det = make().fit(healthy())
        x = healthy(n=300, seed=5).x.copy()
        x[100:120, 1] = np.nan
        out = det.detect(series(x))
        assert np.all(np.isnan(out.statistic[100:120]))
        assert np.all(np.isfinite(out.statistic[200:]))

    @pytest.mark.parametrize("make", SK_MODELS, ids=["isolation-forest", "one-class-svm"])
    def test_a_too_short_baseline_is_rejected(self, make):
        with pytest.raises(ValueError, match="too few complete samples"):
            make().fit(series(np.full((5, 3), np.nan)))


class TestWindowing:
    def test_a_window_lets_the_model_see_temporal_structure(self):
        # Without a window each sample is judged in isolation, so a channel sitting at an ordinary value
        # is ordinary no matter how it got there.
        #
        # The anomaly here is a burst of FAST oscillation with the same amplitude as the slow baseline.
        # Its marginal distribution is essentially unchanged, so a per-sample model has nothing to see;
        # its trajectory is completely different, so a windowed model does. A first attempt swapped two
        # blocks of the smooth signal instead, and that failed to separate the two models at all, because
        # a smooth stretch moved elsewhere is still a smooth stretch.
        rng = np.random.default_rng(6)
        n = 3000
        smooth = np.convolve(rng.normal(size=n), np.ones(20) / 20.0, mode="same")
        smooth = smooth / smooth.std()
        base = series(np.column_stack([smooth, smooth + 0.1 * rng.normal(size=n)]))

        test_smooth = smooth[:600].copy()
        burst = np.sin(np.arange(60) * 1.6)
        burst = burst / burst.std()
        test_smooth[300:360] = burst
        test = np.column_stack([test_smooth, test_smooth + 0.1 * rng.normal(size=600)])

        # The marginal distribution is nearly untouched, which is what makes this a temporal-only signal.
        assert abs(test_smooth[300:360].std() - 1.0) < 0.15
        assert abs(test_smooth[300:360].mean()) < 0.5

        flat = IsolationForestDetector(n_estimators=120, seed=0, window=1).fit(base)
        stacked = IsolationForestDetector(n_estimators=120, seed=0, window=12).fit(base)

        def contrast(det):
            # MEDIAN over the burst against the median elsewhere, not the max. A max over sixty-odd
            # samples is dominated by whichever noise draw happened to be largest, and it gave the
            # per-sample model a spurious signal on the first attempt at this test.
            s = det.detect(series(test)).statistic
            inside = np.nanmedian(s[305:360])
            outside = np.nanmedian(np.concatenate([s[:295], s[370:]]))
            return float(inside - outside)

        c_flat, c_stacked = contrast(flat), contrast(stacked)
        assert c_stacked > 3 * max(c_flat, 1e-6), (
            f"the windowed model must see what the per-sample one cannot: {c_stacked} vs {c_flat}"
        )

    def test_the_statistic_sits_at_the_window_END(self):
        # These are ONLINE detectors: when the window closes, its last sample is the newest one and the
        # earliest point at which the evidence exists. mSTAMP uses the window START for the opposite
        # reason, being retrospective, and the two conventions are deliberate.
        det = IsolationForestDetector(n_estimators=40, seed=0, window=10).fit(healthy())
        out = det.detect(healthy(n=200, seed=7))
        assert np.all(np.isnan(out.statistic[:9])), "no window has closed yet"
        assert np.isfinite(out.statistic[9])
        assert np.isfinite(out.statistic[-1]), "the last sample closes a window"

    def test_a_series_shorter_than_the_window_is_rejected(self):
        det = IsolationForestDetector(n_estimators=40, seed=0, window=50).fit(healthy())
        with pytest.raises(ValueError, match="no complete window"):
            det.detect(healthy(n=20, seed=8))


class TestOneClassSVM:
    def test_subsampling_a_large_baseline_is_recorded_not_silent(self):
        # A model quietly trained on a fraction of the baseline is a different model, so the count it
        # actually used is reported.
        det = OneClassSVMDetector(nu=0.05, max_train=300, seed=0).fit(healthy(n=2000))
        out = det.detect(healthy(n=100, seed=9))
        assert out.meta["n_train"] == 300


class TestAutoencoder:
    def test_a_bottleneck_wider_than_the_input_is_rejected(self):
        # Without a bottleneck the network learns the identity, reconstructs a fault perfectly, and
        # produces a detector that never fires. That failure looks like a well-trained model.
        pytest.importorskip("torch")
        with pytest.raises(ValueError, match="bottleneck"):
            AutoencoderDetector(hidden=(8, 16)).fit(healthy(d=3))

    def test_reconstruction_error_is_larger_for_anomalous_samples(self):
        pytest.importorskip("torch")
        base = healthy(n=2000, seed=0)
        det = AutoencoderDetector(hidden=(16, 2), epochs=120, seed=0).fit(base)

        quiet = det.detect(healthy(n=400, seed=1)).statistic
        rng = np.random.default_rng(2)
        odd = det.detect(series(rng.normal(0, 1, (400, 3)) + 8.0)).statistic
        assert np.nanmedian(odd) > 5 * np.nanmedian(quiet)

    def test_it_learns_the_correlation_structure_not_just_the_range(self):
        # The learned analogue of SPE: a sample that breaks the relationship between channels while
        # staying inside every channel's own range must still reconstruct badly.
        pytest.importorskip("torch")
        base = healthy(n=2000, seed=0)
        det = AutoencoderDetector(hidden=(16, 1), epochs=200, seed=0).fit(base)

        rng = np.random.default_rng(3)
        n = 400
        f = rng.normal(size=n)
        broken = series(np.column_stack([f + 0.2 * rng.normal(size=n),
                                         rng.normal(size=n),
                                         f + 0.2 * rng.normal(size=n)]))
        assert np.nanmedian(det.detect(broken).statistic) > \
            3 * np.nanmedian(det.detect(healthy(n=n, seed=4)).statistic)

    def test_it_runs_on_the_gpu_when_one_is_available(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device on this machine")
        det = AutoencoderDetector(hidden=(16, 2), epochs=30, seed=0).fit(healthy(n=800))
        assert det.device_ == "cuda"
        out = det.detect(healthy(n=200, seed=5))
        assert out.meta["device"] == "cuda"
        assert np.all(np.isfinite(out.statistic))

    def test_cpu_and_gpu_agree_on_the_ordering_of_samples(self):
        # Exact values differ between devices in the last digits, so behaviour is asserted rather than
        # equality: both must rank the same samples as the most anomalous.
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device on this machine")
        base = healthy(n=1200, seed=0)
        rng = np.random.default_rng(6)
        test = series(np.vstack([healthy(n=200, seed=7).x, rng.normal(0, 1, (50, 3)) + 8.0]))

        cpu = AutoencoderDetector(hidden=(16, 2), epochs=80, seed=0, device="cpu").fit(base)
        gpu = AutoencoderDetector(hidden=(16, 2), epochs=80, seed=0, device="cuda").fit(base)
        s_cpu, s_gpu = cpu.detect(test).statistic, gpu.detect(test).statistic

        assert np.nanmedian(s_cpu[200:]) > np.nanmedian(s_cpu[:200])
        assert np.nanmedian(s_gpu[200:]) > np.nanmedian(s_gpu[:200])
        top_cpu = set(np.argsort(-np.nan_to_num(s_cpu))[:40])
        top_gpu = set(np.argsort(-np.nan_to_num(s_gpu))[:40])
        assert len(top_cpu & top_gpu) >= 30, "the two devices should broadly agree on what is anomalous"

    def test_detect_before_fit_raises(self):
        pytest.importorskip("torch")
        with pytest.raises(RuntimeError, match="before fit"):
            AutoencoderDetector().detect(healthy())


def test_a_missing_optional_dependency_names_the_extra_to_install():
    # A missing optional dependency should say what to install, not raise ModuleNotFoundError from three
    # frames down inside somebody else's package.
    from regimecpd.novelty import _require
    with pytest.raises(ImportError, match=r'pip install "regimecpd\[learned\]"'):
        _require("a_module_that_does_not_exist", "learned")
