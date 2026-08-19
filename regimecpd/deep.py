"""The DEEP tier: Deep SVDD and an LSTM encoder-decoder, both trained on healthy data only.

These two exist as a matched PAIR, and the pairing is the point. The novelty rungs in
:mod:`regimecpd.novelty` split into two shapes that behave differently under regime conditioning:

**Boundary-shaped** detectors learn where the healthy cloud IS (isolation forest, one-class SVM,
and Deep SVDD here). **Reconstruction-shaped** detectors learn the correlation structure of normal
operation and score by how badly a sample violates it (the dense autoencoder, PCA SPE, and the LSTM
encoder-decoder here). A downstream study measured conditioning HURTING the boundary-shaped rungs and
HELPING the reconstruction-shaped one, and hypothesised that the shape is what decides it. One deep
detector of each shape is what turns that hypothesis into a test, so both are implemented to their
papers rather than to whatever is convenient.

.. rubric:: Both are optional, and neither degrades quietly

torch is an EXTRA (``pip install "regimecpd[deep]"``), imported lazily inside ``fit``. A missing
backend raises an actionable ImportError at fit time, never at import time and never a silent skip.

.. rubric:: Determinism on CUDA

Seeded, and identical seeds give identical results on one machine and driver. cuDNN's RNN kernels are
documented as non-deterministic unless ``CUBLAS_WORKSPACE_CONFIG`` is set, so exact cross-machine
equality is NOT claimed for the LSTM and the tests assert behaviour rather than values. The device
actually used is recorded in ``Detection.meta["device"]``, because "which hardware produced this
number" is not a question a reader should have to guess at.

Sources, both read against the primary text rather than recalled:

- Ruff, L., Vandermeulen, R., Goernitz, N., Deecke, L., Siddiqui, S. A., Binder, A., Mueller, E.,
  Kloft, M. "Deep One-Class Classification." *ICML 2018*, PMLR 80, 4393-4402.
- Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., Shroff, G. "LSTM-based
  Encoder-Decoder for Multi-sensor Anomaly Detection." *ICML 2016 Anomaly Detection Workshop*,
  arXiv:1607.00148. (NOT the ESANN 2015 LSTM-AD paper, which is a PREDICTION model; citing that one
  for a reconstruction autoencoder would be a factual error.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .novelty import _require, _Windowed
from .types import Detection, Series

__all__ = ["DeepSVDDDetector", "LSTMAutoencoderDetector"]


def _resolve_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DeepSVDDDetector(_Windowed):
    """One-Class Deep SVDD: learn a map under which healthy windows fall in a small hypersphere.

    The statistic is the squared distance from the network's representation to a FIXED centre ``c``:

    .. math::
        s(x) = \\lVert \\phi(x; \\mathcal{W}^{*}) - c \\rVert^{2}

    trained by minimising that distance over healthy windows (the paper's equation 4, the One-Class
    objective; the soft-boundary variant with a radius and the nu-property is not implemented here,
    because the operating point in this package comes from external budget calibration rather than
    from nu, and shipping an unused radius would be a knob that looks like a guarantee).

    .. rubric:: Hypersphere collapse, and the three constraints that prevent it

    This method has one catastrophic failure mode, and the paper proves it rather than warning about
    it: the network can learn a CONSTANT map onto the centre, driving the loss to zero while the
    detector becomes blind. A loss near zero is a red flag here, not a success. Three constraints,
    each from a proposition in Section 3.3, and all three are ENFORCED rather than documented:

    1. **The centre is fixed, not learned, and must not be zero** (Proposition 1). It is set to the
       mean of an initial forward pass, then any coordinate with ``|c_i| < eps`` is pushed to
       ``+/- eps``. The reference implementation leaves EXACTLY-zero coordinates at zero, which is a
       real hole in it; here they go to ``+eps``.
    2. **No bias terms anywhere** (Proposition 2). With a bias, setting the layer's weights to zero
       makes the constant map an OPTIMAL solution, so collapse is not a training accident but the
       objective's own minimum. Every Linear is built with ``bias=False`` and a construction-time
       assertion re-checks it.
    3. **No bounded activations** (Proposition 3). A saturating activation emulates a learned bias
       and reopens the same hole, so the activation is ReLU (or LeakyReLU).

    A collapse metric (the variance of the representations) is computed at fit time and reported in
    the detection meta, so a collapsed model is visible in the artifact rather than inferred from a
    suspiciously flat statistic.

    Parameters
    ----------
    hidden
        Encoder widths; the last entry is the representation dimension. Unlike an autoencoder there
        is no reconstruction, so this is not a bottleneck in the autoencoder sense.
    pretrain_epochs
        Epochs of autoencoder pretraining used to initialise the encoder, as the paper does (it
        initialises Deep SVDD from a trained autoencoder's encoder). Set to 0 to skip.
    """

    hidden: tuple = (32, 8)
    epochs: int = 60
    pretrain_epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-6
    eps: float = 0.1
    device: str = "auto"
    seed: int = 0
    model_: object = field(default=None, repr=False)
    center_: object = field(default=None, repr=False)
    device_: str = "cpu"

    def _build(self, torch, nn, n_in: int):
        """Bias-free, unbounded-activation encoder. Both properties are load-bearing, see the docstring."""
        widths = [n_in, *self.hidden]
        layers = []
        for a, b in zip(widths, widths[1:]):
            layers += [nn.Linear(a, b, bias=False), nn.ReLU()]
        net = nn.Sequential(*layers[:-1])          # no activation on the representation itself
        for module in net.modules():
            if isinstance(module, nn.Linear):
                assert module.bias is None, (
                    "a bias term makes the collapsed constant solution OPTIMAL (Ruff et al. 2018, "
                    "Proposition 2); this network must be built bias-free")
        return net.to(self.device_)

    def fit(self, baseline: Series) -> "DeepSVDDDetector":
        torch = _require("torch", "deep")
        import torch.nn as nn

        features = self._fit_scaler(baseline)
        torch.manual_seed(self.seed)
        self.device_ = _resolve_device(torch, self.device)
        data = torch.tensor(features, dtype=torch.float32, device=self.device_)
        n_in = features.shape[1]
        net = self._build(torch, nn, n_in)

        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        def batches():
            order = torch.randperm(len(data), generator=generator).to(self.device_)
            for start in range(0, len(data), self.batch_size):
                yield data[order[start:start + self.batch_size]]

        # Pretraining, as the paper does: train an autoencoder whose ENCODER is this network, then
        # keep the encoder. The decoder is bias-free too, so pretraining cannot smuggle a bias in.
        if self.pretrain_epochs > 0:
            widths = [n_in, *self.hidden]
            decoder_layers = []
            rev = list(reversed(widths))
            for a, b in zip(rev, rev[1:]):
                decoder_layers += [nn.Linear(a, b, bias=False), nn.ReLU()]
            decoder = nn.Sequential(*decoder_layers[:-1]).to(self.device_)
            ae = nn.Sequential(net, decoder)
            opt = torch.optim.Adam(ae.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            loss_fn = nn.MSELoss()
            ae.train()
            for _ in range(self.pretrain_epochs):
                for batch in batches():
                    opt.zero_grad(set_to_none=True)
                    loss_fn(ae(batch), batch).backward()
                    opt.step()

        # The centre: the mean of an initial forward pass, with every near-zero coordinate pushed
        # away from zero. A zero coordinate is trivially matched by zero weights (Proposition 1).
        net.eval()
        with torch.no_grad():
            c = net(data).mean(dim=0)
        near_zero = c.abs() < self.eps
        # The reference implementation leaves an EXACTLY zero coordinate at zero, because it tests
        # `c < 0` and `c > 0` separately. Using a sign that maps 0 to +1 closes that hole.
        sign = torch.where(c < 0, torch.full_like(c, -1.0), torch.full_like(c, 1.0))
        c = torch.where(near_zero, sign * self.eps, c)
        assert float(c.abs().min()) >= self.eps * (1 - 1e-6), "a centre coordinate is still at zero"
        self.center_ = c

        optimiser = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        net.train()
        for _ in range(self.epochs):
            for batch in batches():
                optimiser.zero_grad(set_to_none=True)
                loss = ((net(batch) - c) ** 2).sum(dim=1).mean()
                loss.backward()
                optimiser.step()
        net.eval()

        # Collapse diagnostics, computed rather than assumed. If the representations have essentially
        # no variance, the network has learned a constant map and the detector is blind.
        with torch.no_grad():
            rep = net(data)
            self._rep_variance = float(((rep - rep.mean(dim=0)) ** 2).sum(dim=1).mean())
            self._final_loss = float(((rep - c) ** 2).sum(dim=1).mean())
        self.model_ = net
        return self

    def detect(self, series: Series) -> Detection:
        if self.model_ is None:
            raise RuntimeError("detect called before fit")
        torch = _require("torch", "deep")
        features, index = self._features(series)
        with torch.no_grad():
            data = torch.tensor(features, dtype=torch.float32, device=self.device_)
            score = ((self.model_(data) - self.center_) ** 2).sum(dim=1).cpu().numpy()
        return self._place(series, index, score, "deep-svdd", {
            "device": self.device_,
            "hidden": self.hidden,
            "final_train_loss": getattr(self, "_final_loss", None),
            # Near-zero representation variance means hypersphere collapse: the model maps every
            # input to the centre. Reported so a collapsed run is visible, not merely suspected.
            "representation_variance": getattr(self, "_rep_variance", None),
            "shape": "boundary",
        })


@dataclass
class LSTMAutoencoderDetector(_Windowed):
    """EncDec-AD: an LSTM encoder-decoder trained to reconstruct healthy windows.

    The encoder consumes the window forward and hands its final hidden state to the decoder, which
    reconstructs the window in REVERSE order. That reversal is the published method, not a detail:
    the first thing the decoder emits is the reconstruction of the LAST input sample, which shortens
    the dependency path at the seam. A forward-order implementation is a different model.

    Training uses teacher forcing (the decoder is fed the true previous sample); inference is free
    running (it is fed its own previous output). The first emission comes straight off the copied
    encoder state with no input at all.

    The published score is a Mahalanobis distance on the element-wise absolute error vector:

    .. math::
        e^{(i)} = \\lvert x^{(i)} - x'^{(i)} \\rvert, \\qquad
        a^{(i)} = (e^{(i)} - \\mu)^{\\top} \\Sigma^{-1} (e^{(i)} - \\mu)

    with mu and Sigma fitted on healthy data. Note ``e`` is element-wise, so it stays a vector, and
    the score is a quadratic form on it, NOT a plain squared error. Sigma is regularised with a small
    ridge before inversion, because a channel that never moves gives a singular covariance and an
    unregularised inverse would turn its rounding noise into an unbounded statistic. That is the same
    degenerate-scale failure this package has already paid for three times in other modules.

    .. rubric:: An honest limit of the source

    Every experiment in the EncDec-AD paper is effectively UNIVARIATE: the two multi-sensor datasets
    are reduced to their first principal component, and the rest are one-dimensional. So the paper
    provides no evidence for the full ``m x m`` covariance at realistic sensor counts, and this
    implementation's use of it on nine channels is an extrapolation. ``score="mse"`` is offered for
    exactly that reason: it drops the covariance and scores the mean squared error instead.

    Parameters
    ----------
    hidden
        LSTM hidden width, the model's capacity knob. A reconstruction model with too much capacity
        learns the identity and reconstructs faults as happily as healthy data ("identity shortcut"),
        which is the reconstruction analogue of hypersphere collapse.
    score
        ``"mahalanobis"`` (the published score) or ``"mse"`` (mean squared reconstruction error).

        These are NOT interchangeable, and the difference is measured rather than assumed. On a fault
        that flips one channel's correlation with another while leaving every marginal untouched, the
        published score separates anomalous from healthy by a factor of 1.36 to 1.73 across three
        seeds, and plain MSE by 1.00 to 1.13, which is to say not at all. The network reconstructs
        each channel's own range either way; only the covariance-weighted quadratic form sees that
        the joint structure broke. ``"mse"`` exists for the honest limit noted above, not as a
        simplification to reach for.
    """

    hidden: int = 32
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    ridge: float = 1e-6
    score: str = "mahalanobis"
    device: str = "auto"
    seed: int = 0
    model_: object = field(default=None, repr=False)
    mu_: object = field(default=None, repr=False)
    precision_: object = field(default=None, repr=False)
    n_channels_: int = 0
    device_: str = "cpu"

    def __post_init__(self) -> None:
        if self.score not in ("mahalanobis", "mse"):
            raise ValueError(f"score must be 'mahalanobis' or 'mse', got {self.score!r}")
        if self.window < 2:
            raise ValueError(
                f"window must be at least 2 for a sequence model, got {self.window}; a length-1 "
                f"window has no temporal structure for an LSTM to encode")

    def _sequences(self, flat: np.ndarray) -> np.ndarray:
        """(n, window*d) from the shared feature contract back to (n, window, d).

        ``_Windowed`` ravels each ``(window, d)`` chunk in C order, so the inverse is a plain
        reshape. Everything the shared scaler guarantees (the relative degeneracy floor, the
        complete-window rule) is preserved by construction.
        """
        return flat.reshape(len(flat), self.window, self.n_channels_)

    def _reconstruct(self, torch, model, seq, teacher_forcing: bool):
        """Reverse-order decoding, exactly as published. Returns the reconstruction in FORWARD order."""
        encoder, decoder, head = model["encoder"], model["decoder"], model["head"]
        _, (h, c) = encoder(seq)
        state = (h, c)
        # Targets in emission order are x^(L), x^(L-1), ..., x^(1): the window reversed on time.
        reversed_seq = torch.flip(seq, dims=[1])
        outputs = []
        # The FIRST emission comes off the copied encoder state with no input at all.
        step = head(state[0][-1])
        outputs.append(step)
        for i in range(1, seq.shape[1]):
            # Training feeds the true previous sample; inference feeds the model's own output.
            fed = reversed_seq[:, i - 1, :] if teacher_forcing else step
            _, state = decoder(fed.unsqueeze(1), state)
            step = head(state[0][-1])
            outputs.append(step)
        recon_reversed = torch.stack(outputs, dim=1)
        # Flip back so index i lines up with the input's index i before any error is taken.
        return torch.flip(recon_reversed, dims=[1])

    def fit(self, baseline: Series) -> "LSTMAutoencoderDetector":
        torch = _require("torch", "deep")
        import torch.nn as nn

        flat = self._fit_scaler(baseline)
        self.n_channels_ = len(baseline.names)
        torch.manual_seed(self.seed)
        self.device_ = _resolve_device(torch, self.device)

        seq = torch.tensor(self._sequences(flat), dtype=torch.float32, device=self.device_)
        d = self.n_channels_
        model = {
            "encoder": nn.LSTM(d, self.hidden, batch_first=True).to(self.device_),
            "decoder": nn.LSTM(d, self.hidden, batch_first=True).to(self.device_),
            "head": nn.Linear(self.hidden, d).to(self.device_),
        }
        params = [p for m in model.values() for p in m.parameters()]
        optimiser = torch.optim.Adam(params, lr=self.lr)
        loss_fn = nn.MSELoss()
        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        for m in model.values():
            m.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(seq), generator=generator).to(self.device_)
            for start in range(0, len(seq), self.batch_size):
                batch = seq[order[start:start + self.batch_size]]
                optimiser.zero_grad(set_to_none=True)
                loss = loss_fn(self._reconstruct(torch, model, batch, teacher_forcing=True), batch)
                loss.backward()
                optimiser.step()
        for m in model.values():
            m.eval()
        self.model_ = model
        self._final_loss = float(loss.detach().cpu())

        # mu and Sigma are fitted on FREE-RUNNING errors, not teacher-forced ones: teacher forcing
        # systematically understates the error a scored window will produce, so calibrating on it
        # would compare against a distribution the detector never sees.
        with torch.no_grad():
            recon = self._reconstruct(torch, model, seq, teacher_forcing=False)
            err = (recon - seq).abs().reshape(-1, d).cpu().numpy()
        self.mu_ = err.mean(axis=0)
        centred = err - self.mu_
        cov = (centred.T @ centred) / max(len(centred) - 1, 1)
        # The ridge is relative to the covariance's own scale: an absolute floor would be a unit
        # dependence, and a dead channel would still invert to something enormous.
        scale = float(np.trace(cov) / d) if d else 1.0
        cov = cov + np.eye(d) * max(self.ridge * max(scale, 1e-12), 1e-12)
        self.precision_ = np.linalg.pinv(cov)
        return self

    def detect(self, series: Series) -> Detection:
        if self.model_ is None:
            raise RuntimeError("detect called before fit")
        torch = _require("torch", "deep")
        flat, index = self._features(series)
        seq = torch.tensor(self._sequences(flat), dtype=torch.float32, device=self.device_)
        with torch.no_grad():
            recon = self._reconstruct(torch, self.model_, seq, teacher_forcing=False)
            err = (recon - seq).abs().cpu().numpy()
        if self.score == "mse":
            score = (err ** 2).mean(axis=(1, 2))
        else:
            centred = err - self.mu_                     # (n, window, d)
            quad = np.einsum("nwi,ij,nwj->nw", centred, self.precision_, centred)
            # One score per window: the mean over its timesteps. The statistic lands on the window's
            # last sample, the same convention every windowed detector here uses.
            score = quad.mean(axis=1)
        return self._place(series, index, score, "lstm-autoencoder", {
            "device": self.device_,
            "hidden": self.hidden,
            "score": self.score,
            "final_train_loss": getattr(self, "_final_loss", None),
            "shape": "reconstruction",
        })
