"""Healthy-only novelty detectors: isolation forest, one-class SVM, and an autoencoder.

These learn what normal looks like from a healthy baseline and score how far a new sample sits from it.
No fault labels are needed, which matters because a fleet has thousands of healthy machine-years and a
handful of labelled failures.

.. rubric:: The sign, which is where these go wrong

scikit-learn's novelty estimators return **higher scores for more normal** samples. Every other detector
in this package returns higher for more anomalous. The flip happens once, here, at the source.

This is worth calling out rather than doing quietly. A missed flip produces a detector that scores exactly
as badly as it should score well: its alarm-budget curve looks like a plausible weak method rather than
like an error, and on a benchmark table it reads as "this approach does not work for this problem". There
is a test asserting the orientation for every model in this module.

.. rubric:: Optional dependencies, and why the core does not carry them

The core of this package is numpy and nothing else. A monitoring library that pulls in a full ML stack to
compute a CUSUM has misplaced its boundary, so:

- ``pip install "regimecpd[learned]"`` adds scikit-learn for the isolation forest and one-class SVM.
- ``pip install "regimecpd[deep]"`` adds PyTorch for the autoencoder, which uses CUDA when it is there.

Each import happens inside the method that needs it, with an error naming the extra. A missing optional
dependency should say what to install, not raise ``ModuleNotFoundError`` from three frames down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scaling import robust_scale
from .types import Detection, Series

__all__ = ["IsolationForestDetector", "OneClassSVMDetector", "AutoencoderDetector"]


def _require(module: str, extra: str):
    try:
        return __import__(module)
    except ImportError as exc:
        raise ImportError(
            f"{module} is needed for this detector and is not installed. "
            f'Install it with: pip install "regimecpd[{extra}]"'
        ) from exc


@dataclass
class _Windowed:
    """Shared feature construction: standardise against the baseline, then optionally stack a window.

    ``window > 1`` stacks that many consecutive samples into one feature vector, which is what lets these
    models see temporal structure at all. Without it they judge each sample in isolation and cannot
    distinguish a channel sitting at an unusual value from a channel that got there in an unusual way.

    The statistic for a stacked window is placed at the window's **last** sample, not its first. These are
    online detectors: at the moment the window closes, that is the newest sample and the earliest point at
    which the evidence exists. mSTAMP places its statistic at the window start for the opposite reason,
    being retrospective over the whole record, and the two conventions are deliberate rather than
    inconsistent.
    """

    window: int = 1
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    names_: tuple = ()

    def _fit_scaler(self, baseline: Series) -> np.ndarray:
        x = baseline.x
        rows = np.all(np.isfinite(x), axis=1)
        if np.count_nonzero(rows) < max(2, self.window):
            raise ValueError("the baseline has too few complete samples to fit")
        xb = x[rows]
        self.mean_ = xb.mean(axis=0)
        spread = xb.std(axis=0)
        self.scale_ = robust_scale(spread, self.mean_)
        self.names_ = baseline.names
        return self._stack((xb - self.mean_) / self.scale_)

    def _features(self, series: Series) -> "tuple[np.ndarray, np.ndarray]":
        """Return ``(features, index)`` where ``index`` maps each feature row to a sample index."""
        if self.mean_ is None:
            raise RuntimeError("detect called before fit")
        if series.names != self.names_:
            raise ValueError(f"fitted on channels {self.names_}, asked to score {series.names}")
        z = (series.x - self.mean_) / self.scale_
        w = self.window
        if w == 1:
            ok = np.flatnonzero(np.all(np.isfinite(z), axis=1))
            return z[ok], ok
        stacked, index = [], []
        for i in range(w - 1, len(z)):
            chunk = z[i - w + 1: i + 1]
            if np.all(np.isfinite(chunk)):
                stacked.append(chunk.ravel())
                index.append(i)      # the window's LAST sample: see the class docstring
        if not stacked:
            raise ValueError("no complete window of the requested length exists in this series")
        return np.asarray(stacked), np.asarray(index, dtype=int)

    def _stack(self, z: np.ndarray) -> np.ndarray:
        w = self.window
        if w == 1:
            return z
        return np.asarray([z[i - w + 1: i + 1].ravel() for i in range(w - 1, len(z))])

    def _place(self, series: Series, index: np.ndarray, scores: np.ndarray,
               method: str, meta: dict) -> Detection:
        statistic = np.full(series.n, np.nan)
        statistic[index] = scores
        return Detection(series.t, statistic, method, {**meta, "window": self.window})


@dataclass
class IsolationForestDetector(_Windowed):
    """Isolation forest trained on healthy data only.

    Isolates points by random axis-aligned splits: anomalies need fewer splits to separate, so the mean
    path length over a forest is short for them. No distributional assumption and linear time.

    Its scores are **not calibrated probabilities**, which is precisely the gap the conformal layer fills.
    Pairing the two is a design decision rather than an accident.

    Source: Liu, F. T., Ting, K. M., Zhou, Z.-H. "Isolation Forest." *ICDM 2008*, pp. 413-422.
    doi:10.1109/ICDM.2008.17
    """

    n_estimators: int = 200
    max_samples: "int | str" = "auto"
    seed: int = 0
    model_: object = field(default=None, repr=False)

    def fit(self, baseline: Series) -> "IsolationForestDetector":
        _require("sklearn", "learned")
        from sklearn.ensemble import IsolationForest

        features = self._fit_scaler(baseline)
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators, max_samples=self.max_samples,
            random_state=self.seed, contamination="auto",
        ).fit(features)
        return self

    def detect(self, series: Series) -> Detection:
        if self.model_ is None:
            raise RuntimeError("detect called before fit")
        features, index = self._features(series)
        # NEGATED: sklearn's score_samples is higher for more normal. See the module docstring.
        scores = -self.model_.score_samples(features)
        return self._place(series, index, scores, "isolation-forest",
                           {"n_estimators": self.n_estimators})


@dataclass
class OneClassSVMDetector(_Windowed):
    """One-class SVM with an RBF kernel, trained on healthy data only.

    Fits a boundary enclosing most of the baseline. Sharper than an isolation forest when normal
    operation is a compact, curved region; considerably slower, since training is superlinear in the
    baseline size, which is why ``max_train`` exists.

    Source: Scholkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., Williamson, R. C. "Estimating
    the Support of a High-Dimensional Distribution." *Neural Computation* **13**(7), 1443-1471 (2001).
    **UNVERIFIED:** not opened against a primary source during the research pass behind this package.
    """

    nu: float = 0.05
    gamma: "str | float" = "scale"
    max_train: int = 5000
    seed: int = 0
    model_: object = field(default=None, repr=False)

    def fit(self, baseline: Series) -> "OneClassSVMDetector":
        _require("sklearn", "learned")
        from sklearn.svm import OneClassSVM

        features = self._fit_scaler(baseline)
        if len(features) > self.max_train:
            # Subsampled deterministically, and the fact is recorded in meta rather than left implicit:
            # a model silently trained on a fraction of the baseline is a different model.
            rng = np.random.default_rng(self.seed)
            features = features[rng.choice(len(features), self.max_train, replace=False)]
        self.model_ = OneClassSVM(nu=self.nu, gamma=self.gamma, kernel="rbf").fit(features)
        self._n_train = len(features)
        return self

    def detect(self, series: Series) -> Detection:
        if self.model_ is None:
            raise RuntimeError("detect called before fit")
        features, index = self._features(series)
        # NEGATED, same reason as the isolation forest.
        scores = -self.model_.decision_function(features)
        return self._place(series, index, scores, "one-class-svm",
                           {"nu": self.nu, "n_train": getattr(self, "_n_train", None)})


@dataclass
class AutoencoderDetector(_Windowed):
    """Dense autoencoder trained on healthy data only; the statistic is reconstruction error.

    A bottleneck forces the network to learn the correlation structure of normal operation. A sample that
    respects that structure reconstructs well; one that breaks it does not. This is the learned analogue
    of the SPE statistic in :mod:`regimecpd.spc`, with a nonlinear manifold in place of a linear subspace,
    and the two are worth comparing directly for exactly that reason.

    No sign flip: reconstruction error is already larger for more anomalous.

    Uses CUDA when it is available. Determinism is set from ``seed``, though results on GPU can still
    differ in the last digits between driver versions, so tests assert behaviour rather than exact values.

    Parameters
    ----------
    hidden
        Encoder widths. The last entry is the bottleneck, and it is the only architectural choice that
        really matters: a bottleneck as wide as the input learns the identity and reconstructs a fault
        perfectly, which produces a detector that never fires. There is a test for that failure.
    """

    hidden: tuple = (32, 8)
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "auto"
    seed: int = 0
    model_: object = field(default=None, repr=False)
    device_: str = "cpu"

    def _resolve_device(self, torch) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, baseline: Series) -> "AutoencoderDetector":
        torch = _require("torch", "deep")
        import torch.nn as nn

        features = self._fit_scaler(baseline)
        n_in = features.shape[1]
        if self.hidden and self.hidden[-1] >= n_in:
            raise ValueError(
                f"the bottleneck ({self.hidden[-1]}) is not narrower than the input ({n_in}); "
                f"an autoencoder without a bottleneck learns the identity and reconstructs faults "
                f"perfectly, giving a detector that never fires"
            )

        torch.manual_seed(self.seed)
        self.device_ = self._resolve_device(torch)

        widths = [n_in, *self.hidden]
        encoder, decoder = [], []
        for a, b in zip(widths, widths[1:]):
            encoder += [nn.Linear(a, b), nn.ReLU()]
        for a, b in zip(reversed(widths), list(reversed(widths))[1:]):
            decoder += [nn.Linear(a, b), nn.ReLU()]
        decoder = decoder[:-1]          # no activation on the output: the target is standardised, so it
        model = nn.Sequential(*encoder[:-1], *decoder).to(self.device_)   # is signed and unbounded

        data = torch.tensor(features, dtype=torch.float32, device=self.device_)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        model.train()
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        for _ in range(self.epochs):
            order = torch.randperm(len(data), generator=generator).to(self.device_)
            for start in range(0, len(data), self.batch_size):
                batch = data[order[start:start + self.batch_size]]
                optimiser.zero_grad(set_to_none=True)
                loss = loss_fn(model(batch), batch)
                loss.backward()
                optimiser.step()
        model.eval()
        self.model_ = model
        self._final_loss = float(loss.detach().cpu())
        return self

    def detect(self, series: Series) -> Detection:
        if self.model_ is None:
            raise RuntimeError("detect called before fit")
        torch = _require("torch", "deep")
        features, index = self._features(series)
        with torch.no_grad():
            data = torch.tensor(features, dtype=torch.float32, device=self.device_)
            error = ((self.model_(data) - data) ** 2).mean(dim=1).cpu().numpy()
        return self._place(series, index, error, "autoencoder", {
            "device": self.device_, "hidden": self.hidden,
            "final_train_loss": getattr(self, "_final_loss", None),
        })
