"""Operating-regime segmentation: deciding what context a sample was recorded in.

This is the stage the rest of the package is downstream of, and the one the central claim is about. A
haul truck climbing a ramp loaded and the same truck returning empty are two different machines as far as
any channel is concerned. Comparing a loaded reading against an empty baseline produces an excursion that
is real, large and entirely uninformative.

Two routes, because the two situations are genuinely different and the difference is worth measuring:

**Observed regime** (:class:`DiscreteRegimes`). The context is handed to you. NASA C-MAPSS carries three
operational setting columns; a truck fleet might carry loaded/empty and a gear indicator. Regimes are the
distinct observed combinations.

**Discovered regime** (:class:`KMeansRegimes`). Nobody tells you the context, so it is clustered out of
context features. This is the realistic case, and it is strictly harder.

Running both on the same data and reporting the gap between them measures **the price of not being told
the regime**. A package that only implemented the observed route would flatter itself on any benchmark
that happens to ship its operating conditions as columns, and C-MAPSS is exactly such a benchmark.

Both routes mark a sample that fits no known regime as ``-1`` rather than snapping it to the nearest one.
See :class:`~regimecpd.types.RegimeLabels` for why that matters.

.. rubric:: The leakage trap

A regime model is fitted on a **healthy baseline** and then applied to the monitored record. Fitting it
on the record being monitored folds the fault into the definition of normal, and the residual then has
the fault subtracted out of it. The failure is quiet: everything runs, the residual looks well behaved,
and detection performance collapses for a reason that is invisible in the outputs. ``fit`` and
``label`` are separate calls specifically so that the baseline is an explicit choice rather than a
default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scaling import robust_scale
from .types import RegimeLabels

__all__ = ["ContextScaler", "DiscreteRegimes", "KMeansRegimes", "kmeans_inertia_sweep"]


@dataclass
class ContextScaler:
    """Centre and scale context features before any distance is computed over them.

    Without this, a context feature measured in Pascals dominates one measured in radians purely through
    its units, and the clustering silently becomes a clustering of that one feature. Scale is stored from
    the fit so that the transform applied to monitored data is the same one the baseline defined.

    A feature with zero spread in the baseline is given a scale of 1 rather than 0. Dividing by its
    spread would produce infinities from a feature that carries no information at all.
    """

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, context: np.ndarray) -> "ContextScaler":
        c = np.atleast_2d(np.asarray(context, dtype=float))
        if c.shape[0] == 1 and c.size > 1 and np.ndim(context) == 1:
            c = c.T
        self.mean_ = c.mean(axis=0)
        spread = c.std(axis=0)
        self.scale_ = robust_scale(spread, self.mean_)
        return self

    def transform(self, context: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("ContextScaler.transform called before fit")
        c = np.asarray(context, dtype=float)
        c = c.reshape(-1, 1) if c.ndim == 1 else c
        return (c - self.mean_) / self.scale_

    def fit_transform(self, context: np.ndarray) -> np.ndarray:
        return self.fit(context).transform(context)


def _as_context(context: np.ndarray) -> np.ndarray:
    c = np.asarray(context, dtype=float)
    return c.reshape(-1, 1) if c.ndim == 1 else c


@dataclass
class DiscreteRegimes:
    """Regimes as the distinct observed combinations of a handed-to-you context.

    Context values are rounded to ``decimals`` before combinations are formed. Real "discrete" settings
    arrive as floats with noise in the last digits, and without rounding every sample becomes its own
    regime, which is the same as having no regimes at all while looking like it worked.

    A combination not seen during fitting is labelled ``-1``. That is the honest answer: there is no
    fitted baseline for a context that never occurred.
    """

    decimals: int = 3
    combinations_: dict = field(default_factory=dict)

    @property
    def n_regimes(self) -> int:
        return len(self.combinations_)

    def _keys(self, context: np.ndarray) -> list[tuple]:
        c = np.round(_as_context(context), self.decimals)
        return [tuple(row) for row in c]

    def fit(self, context: np.ndarray) -> "DiscreteRegimes":
        seen: dict = {}
        for key in self._keys(context):
            if key not in seen:
                seen[key] = len(seen)
        self.combinations_ = seen
        return self

    def label(self, context: np.ndarray) -> RegimeLabels:
        if not self.combinations_:
            raise RuntimeError("DiscreteRegimes.label called before fit")
        labels = np.array([self.combinations_.get(k, -1) for k in self._keys(context)], dtype=int)
        return RegimeLabels(labels, max(self.n_regimes, 1), "discrete",
                            {"decimals": self.decimals, "n_combinations": self.n_regimes})

    def fit_label(self, context: np.ndarray) -> RegimeLabels:
        return self.fit(context).label(context)


def _kmeans_plusplus(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: spread the initial centres out instead of picking k points at random.

    Random seeding regularly puts two centres inside one dense cluster and none in a sparse one, and
    Lloyd's algorithm cannot recover from that. On regime data the sparse clusters are the interesting
    operating conditions, so plain random seeding loses exactly the regimes worth having.
    """
    centres = np.empty((k, x.shape[1]), dtype=float)
    centres[0] = x[rng.integers(len(x))]
    closest = np.sum((x - centres[0]) ** 2, axis=1)
    for i in range(1, k):
        total = closest.sum()
        if not np.isfinite(total) or total <= 0:
            centres[i] = x[rng.integers(len(x))]      # degenerate: all points coincide
        else:
            centres[i] = x[rng.choice(len(x), p=closest / total)]
        closest = np.minimum(closest, np.sum((x - centres[i]) ** 2, axis=1))
    return centres


@dataclass
class KMeansRegimes:
    """Regimes discovered by clustering the context, with an explicit novelty radius.

    Plain numpy: k-means++ seeding, Lloyd iterations, ``n_init`` restarts keeping the lowest inertia.
    Deterministic given ``seed``, because a segmentation that changes between runs makes every downstream
    number unreproducible.

    Parameters
    ----------
    n_regimes
        Number of clusters. Chosen deliberately rather than searched, because the right k is a question
        about the machine (loaded/empty times a handful of grades) and not about the data. Use
        :func:`kmeans_inertia_sweep` to look at the trade-off before deciding.
    novelty_quantile
        Radius per cluster, taken as this quantile of the baseline distances to that centre. A monitored
        sample farther than ``novelty_factor`` times the radius is labelled ``-1``.
    novelty_factor
        Slack on the radius. The default of 1.5 keeps ordinary baseline variation inside the regime while
        still catching a genuinely new operating context.

    .. rubric:: Why the novelty radius exists

    Without it, k-means assigns every point to its nearest centre no matter how far away it is, so a
    machine running in a context the baseline never contained gets a confident regime label and a
    meaningless residual. That is worse than no answer, because the residual it produces looks exactly
    like every other residual. The radius turns "I have never seen this" into a state the caller can
    check via ``RegimeLabels.coverage``.
    """

    n_regimes: int
    novelty_quantile: float = 0.99
    novelty_factor: float = 1.5
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-6
    seed: int = 0
    centres_: np.ndarray | None = None
    radii_: np.ndarray | None = None
    scaler_: ContextScaler | None = None
    inertia_: float = float("nan")

    def _lloyd(self, x: np.ndarray, rng: np.random.Generator) -> "tuple[np.ndarray, float]":
        centres = _kmeans_plusplus(x, self.n_regimes, rng)
        for _ in range(self.max_iter):
            d2 = np.sum((x[:, None, :] - centres[None, :, :]) ** 2, axis=2)
            assign = np.argmin(d2, axis=1)
            moved = 0.0
            for j in range(self.n_regimes):
                members = x[assign == j]
                if len(members) == 0:
                    # An empty cluster is re-seeded at the point worst served by the current centres,
                    # rather than dropped. Dropping it silently returns fewer regimes than requested and
                    # every downstream report then describes a k it did not use.
                    new = x[np.argmax(np.min(d2, axis=1))]
                else:
                    new = members.mean(axis=0)
                moved = max(moved, float(np.linalg.norm(new - centres[j])))
                centres[j] = new
            if moved < self.tol:
                break
        d2 = np.sum((x[:, None, :] - centres[None, :, :]) ** 2, axis=2)
        return centres, float(np.sum(np.min(d2, axis=1)))

    def fit(self, context: np.ndarray) -> "KMeansRegimes":
        """Fit on a HEALTHY baseline. See the module docstring on the leakage trap."""
        c = _as_context(context)
        if len(c) < self.n_regimes:
            raise ValueError(f"cannot fit {self.n_regimes} regimes to {len(c)} samples")

        self.scaler_ = ContextScaler().fit(c)
        x = self.scaler_.transform(c)

        rng = np.random.default_rng(self.seed)
        best, best_inertia = None, np.inf
        for _ in range(max(1, self.n_init)):
            centres, inertia = self._lloyd(x, rng)
            if inertia < best_inertia:
                best, best_inertia = centres, inertia
        self.centres_, self.inertia_ = best, best_inertia

        d = np.linalg.norm(x[:, None, :] - self.centres_[None, :, :], axis=2)
        assign = np.argmin(d, axis=1)
        radii = np.empty(self.n_regimes)
        for j in range(self.n_regimes):
            member_d = d[assign == j, j]
            # A cluster with no baseline members has no radius to speak of. Zero would reject everything
            # it ever sees, so it inherits the largest radius that WAS estimated and the caller finds out
            # through coverage rather than through a silently empty regime.
            radii[j] = np.quantile(member_d, self.novelty_quantile) if member_d.size else np.nan
        if np.all(np.isnan(radii)):
            radii = np.zeros(self.n_regimes)
        radii = np.where(np.isnan(radii), np.nanmax(radii), radii)
        # A DEGENERATE radius rejects the regime it belongs to. When a cluster's baseline members all sit
        # at essentially the same point (a discrete operating condition, a held setpoint), its quantile
        # distance is roundoff, and `distance > novelty_factor * ~0` is then true for every sample the
        # cluster will ever see, including one sitting exactly on its centre. The regime is created, is
        # never assignable, and shows up only as missing coverage.
        #
        # The floor is relative to the SPREAD OF THE CENTRES, which is the scale distances live on here,
        # so it means the same thing whatever units the context arrived in.
        if self.centres_ is not None and len(self.centres_) > 1:
            spread = float(np.linalg.norm(self.centres_.std(axis=0)))
        else:
            spread = 1.0
        floor = max(spread, 1.0) * 1e-9
        self.radii_ = np.maximum(radii, floor)
        return self

    def label(self, context: np.ndarray) -> RegimeLabels:
        if self.centres_ is None or self.scaler_ is None or self.radii_ is None:
            raise RuntimeError("KMeansRegimes.label called before fit")
        x = self.scaler_.transform(_as_context(context))
        d = np.linalg.norm(x[:, None, :] - self.centres_[None, :, :], axis=2)
        nearest = np.argmin(d, axis=1)
        distance = d[np.arange(len(x)), nearest]
        outside = distance > self.novelty_factor * self.radii_[nearest]
        labels = np.where(outside, -1, nearest).astype(int)
        return RegimeLabels(labels, self.n_regimes, "kmeans", {
            "inertia": self.inertia_,
            "novelty_quantile": self.novelty_quantile,
            "novelty_factor": self.novelty_factor,
            "seed": self.seed,
        })

    def fit_label(self, context: np.ndarray) -> RegimeLabels:
        return self.fit(context).label(context)


def kmeans_inertia_sweep(context: np.ndarray, k_values: "list[int]", seed: int = 0,
                         n_init: int = 5) -> "list[tuple[int, float]]":
    """Within-cluster sum of squares for each k, so the choice of k is made with the curve in view.

    Deliberately returns numbers rather than picking a k. Automatic elbow detection on regime data
    routinely returns k = 2 for a machine with six operating conditions, because the two loaded states
    dominate the variance. The right k is a question about the machine, and the person who knows the
    machine should answer it with this curve in front of them.
    """
    out = []
    for k in k_values:
        model = KMeansRegimes(n_regimes=k, seed=seed, n_init=n_init).fit(context)
        out.append((k, model.inertia_))
    return out
