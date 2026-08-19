"""The deep tier: Deep SVDD and the LSTM encoder-decoder.

The tests that matter here are not "does it run". Both methods have a documented way of being
silently useless: Deep SVDD can collapse its hypersphere and score everything identically, and a
reconstruction model can learn the identity and reconstruct faults as happily as healthy data. Both
failures produce a plausible number and no error, which is this package's recurring theme.

The reverse-order decoding of EncDec-AD gets its own behavioural test, because a forward-order
implementation would look correct on every smoke check while being a different published method.
"""
from __future__ import annotations

import numpy as np
import pytest

from regimecpd import Series
from regimecpd.deep import DeepSVDDDetector, LSTMAutoencoderDetector

torch = pytest.importorskip("torch")


def _correlated(n: int = 900, d: int = 5, seed: int = 0) -> np.ndarray:
    """Healthy data with a real correlation to break, so a fault is not just a mean shift."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    x[:, 1] = 0.9 * x[:, 0] + rng.normal(scale=0.1, size=n)
    return x


def _series(x: np.ndarray) -> Series:
    return Series(np.arange(len(x), dtype=float), x, tuple(f"c{i}" for i in range(x.shape[1])))


class TestDeepSVDD:
    def test_the_network_has_no_bias_anywhere(self):
        # Proposition 2 of Ruff et al. 2018: with a bias, the collapsed constant map is an OPTIMAL
        # solution, so this is not a style preference, it is what keeps the objective meaningful.
        import torch.nn as nn
        det = DeepSVDDDetector(window=4, epochs=2, pretrain_epochs=0, seed=0).fit(_series(_correlated()))
        linears = [m for m in det.model_.modules() if isinstance(m, nn.Linear)]
        assert linears, "no Linear layers found; the assertion below would be vacuous"
        assert all(m.bias is None for m in linears)

    def test_no_centre_coordinate_sits_at_zero(self):
        # Proposition 1: a zero coordinate is trivially matched by zero weights. The reference
        # implementation leaves EXACTLY-zero coordinates at zero; this one must not.
        det = DeepSVDDDetector(window=4, epochs=2, pretrain_epochs=0, eps=0.1, seed=0)
        det.fit(_series(_correlated()))
        assert float(det.center_.abs().min()) >= 0.1 - 1e-6

    def test_a_trained_model_has_not_collapsed(self):
        det = DeepSVDDDetector(window=4, epochs=20, pretrain_epochs=5, seed=0)
        det.fit(_series(_correlated()))
        meta = det.detect(_series(_correlated())).meta
        assert meta["representation_variance"] > 1e-6, (
            "the representations have essentially no variance: the network learned a constant map, "
            "which is hypersphere collapse and makes the detector blind")

    def test_the_statistic_is_larger_for_anomalous_windows(self):
        healthy = _correlated()
        broken = healthy.copy()
        broken[700:, 1] *= -1.0          # break the correlation, leaving every marginal unchanged
        det = DeepSVDDDetector(window=6, epochs=20, pretrain_epochs=5, seed=0).fit(_series(healthy))
        stat = det.detect(_series(broken)).statistic
        assert np.nanmean(stat[705:]) > 1.3 * np.nanmean(stat[:690])

    def test_it_declares_its_shape_and_device(self):
        # A downstream study groups rungs by shape, so the shape travels with the detection rather
        # than living in a lookup table that can drift away from the code.
        meta = DeepSVDDDetector(window=4, epochs=2, pretrain_epochs=0).fit(
            _series(_correlated())).detect(_series(_correlated())).meta
        assert meta["shape"] == "boundary"
        assert meta["device"] in ("cpu", "cuda")

    def test_detect_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            DeepSVDDDetector().detect(_series(_correlated()))


class TestLSTMAutoencoder:
    def test_a_length_one_window_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            LSTMAutoencoderDetector(window=1)

    def test_an_unknown_score_is_rejected(self):
        with pytest.raises(ValueError, match="mahalanobis"):
            LSTMAutoencoderDetector(window=4, score="euclidean")

    def test_the_decoder_reconstructs_in_reverse_which_favours_the_LAST_step(self):
        """The published method's signature, and the one a forward-order build would fail.

        EncDec-AD copies the encoder's FINAL state into the decoder and emits the reconstruction of
        the window's LAST sample first, with no input at all. So the last timestep sits closest to
        the state that encodes it and is reconstructed best; a forward-order implementation would
        favour the FIRST timestep instead. This asserts the asymmetry rather than the ordering of a
        tensor, so it survives any refactor that keeps the method correct.
        """
        det = LSTMAutoencoderDetector(window=8, hidden=16, epochs=40, seed=0)
        x = _correlated(n=700)
        det.fit(_series(x))
        flat, _ = det._features(_series(x))
        seq = torch.tensor(det._sequences(flat), dtype=torch.float32, device=det.device_)
        with torch.no_grad():
            recon = det._reconstruct(torch, det.model_, seq, teacher_forcing=False)
            per_step = ((recon - seq) ** 2).mean(dim=(0, 2)).cpu().numpy()
        assert per_step[-1] < per_step[0], (
            f"the LAST timestep should reconstruct best under reverse-order decoding, got "
            f"first={per_step[0]:.5f} last={per_step[-1]:.5f}; a forward-order decoder inverts this")

    def test_the_statistic_is_larger_for_anomalous_windows(self):
        healthy = _correlated()
        broken = healthy.copy()
        broken[700:, 1] *= -1.0
        det = LSTMAutoencoderDetector(window=6, hidden=16, epochs=30, seed=0).fit(_series(healthy))
        stat = det.detect(_series(broken)).statistic
        assert np.nanmean(stat[705:]) > 1.2 * np.nanmean(stat[:690])

    def test_the_PUBLISHED_score_sees_a_broken_correlation_and_plain_MSE_does_not(self):
        """Measured, and the reason the paper's Mahalanobis step is not decoration.

        The injected fault flips one channel's correlation with another and leaves every marginal
        distribution untouched. The network still reconstructs each channel's own range, so the mean
        squared error barely moves; the quadratic form on the error vector, weighted by the inverse
        covariance of HEALTHY errors, is what makes the broken joint structure visible.

        Ratios of anomalous mean to healthy mean, three seeds, measured 2026-08-19:
        mahalanobis 1.378 / 1.363 / 1.730, mse 0.999 / 1.080 / 1.126. So this is not a seed
        accident, and a reimplementation that "simplifies" the score to MSE would keep passing a
        smoke test while losing the detector's actual sensitivity on this fault class.
        """
        healthy = _correlated()
        broken = healthy.copy()
        broken[700:, 1] *= -1.0
        ratios = {}
        for score in ("mahalanobis", "mse"):
            det = LSTMAutoencoderDetector(window=6, hidden=16, epochs=30, score=score, seed=0)
            stat = det.fit(_series(healthy)).detect(_series(broken)).statistic
            ratios[score] = float(np.nanmean(stat[705:]) / np.nanmean(stat[:690]))
        assert ratios["mahalanobis"] > 1.2, ratios
        assert ratios["mahalanobis"] > ratios["mse"] + 0.15, ratios

    def test_a_dead_channel_does_not_produce_an_unbounded_statistic(self):
        # The covariance of the error vectors is singular when a channel never moves. Without the
        # ridge, its inverse turns rounding noise into an enormous quadratic form: the same
        # degenerate-scale defect this package has fixed in three other modules.
        x = _correlated()
        x[:, 3] = 21610.0                      # constant to within nothing at all
        det = LSTMAutoencoderDetector(window=4, hidden=8, epochs=10, seed=0).fit(_series(x))
        stat = det.detect(_series(x)).statistic
        assert np.all(np.isfinite(stat[~np.isnan(stat)]))
        assert np.nanmax(stat) < 1e6

    def test_it_declares_its_shape_and_device(self):
        meta = LSTMAutoencoderDetector(window=4, hidden=8, epochs=5).fit(
            _series(_correlated())).detect(_series(_correlated())).meta
        assert meta["shape"] == "reconstruction"
        assert meta["device"] in ("cpu", "cuda")

    def test_detect_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            LSTMAutoencoderDetector(window=4).detect(_series(_correlated()))


class TestTheShapePair:
    def test_the_two_declare_OPPOSITE_shapes(self):
        """The whole reason both exist: they are a matched pair for a shape hypothesis.

        If a refactor ever made these agree, the downstream experiment would silently stop being a
        test of anything, so the opposition is pinned here rather than assumed.
        """
        x = _series(_correlated())
        a = DeepSVDDDetector(window=4, epochs=2, pretrain_epochs=0).fit(x).detect(x).meta["shape"]
        b = LSTMAutoencoderDetector(window=4, hidden=8, epochs=2).fit(x).detect(x).meta["shape"]
        assert {a, b} == {"boundary", "reconstruction"}
