"""mSTAMP: the multidimensional matrix profile, which returns the window AND the dimension subset.

Yeh, Kavantzas and Keogh (2017). This is the second answer in the package to "which variables", reached
by a route that shares nothing with the PCA contributions in :mod:`regimecpd.spc`. That independence is
the point: agreement between two unrelated attributions is worth considerably more than either alone, and
disagreement is a signal to look harder rather than to pick a favourite.

.. rubric:: What a matrix profile is

For every subsequence of length :math:`m`, the matrix profile records the z-normalised Euclidean distance
to its nearest neighbour elsewhere in the series. A small value means "this pattern happens again" (a
motif). A **large** value means "nothing else in this record looks like this", which is a discord, and
discords are what onset detection wants.

No labels, no training set, and no threshold to tune before the fact.

.. rubric:: The multidimensional part, and why the naive generalisation fails

The obvious extension to :math:`d` channels is to sum the distance across all of them. The authors show
that this does not produce meaningful motifs except in contrived cases: a pattern that repeats in three
of twelve channels is swamped by the nine that are just noise, and the sum is dominated by the channels
carrying nothing.

mSTAMP instead computes, for each subsequence and each :math:`k`, the distance using the **best**
:math:`k` dimensions. Sorting the per-dimension distances ascending and taking a running mean gives every
:math:`k` at once:

.. math::
    D_k(i, j) = \\frac{1}{k}\\sum_{l=1}^{k} \\mathrm{sort}_l\\left( d_1(i,j), \\dots, d_d(i,j) \\right)

and :math:`P_k(i) = \\min_j D_k(i,j)` over non-trivial matches. The dimensions achieving that minimum are
the subset the pattern lives in, which is the attribution this module exists to produce.

.. rubric:: Choosing k is the user's job, not the algorithm's

The profile is returned for **every** :math:`k`, and there is deliberately no automatic selection. A
fault in two channels and a fault in nine are different events, and picking :math:`k` by an information
criterion silently commits to one reading of the data. The authors provide a minimum-description-length
heuristic; it is not implemented here, and its absence is stated rather than papered over.

Source: Yeh, C.-C. M., Kavantzas, N., Keogh, E. J. "Matrix Profile VI: Meaningful Multidimensional Motif
Discovery." *ICDM 2017*, pp. 565-574. Reference implementation: https://github.com/mcyeh/mstamp
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Attribution, Detection, Series

__all__ = ["MatrixProfile", "mass_distance_profile"]

_TINY = 1e-10


def _sliding_stats(x: np.ndarray, m: int) -> "tuple[np.ndarray, np.ndarray]":
    """Mean and standard deviation of every length-``m`` window, in O(n).

    Computed from prefix sums rather than window by window. The naive version is O(nm) and dominates the
    whole algorithm for any realistic window length.
    """
    cs = np.concatenate([[0.0], np.cumsum(x)])
    css = np.concatenate([[0.0], np.cumsum(x ** 2)])
    total = cs[m:] - cs[:-m]
    total_sq = css[m:] - css[:-m]
    mean = total / m
    var = np.maximum(total_sq / m - mean ** 2, 0.0)
    return mean, np.sqrt(var)


def mass_distance_profile(query: np.ndarray, series: np.ndarray,
                          mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """z-normalised Euclidean distance from ``query`` to every window of ``series``, via MASS.

    The sliding dot product is computed with an FFT convolution, so this is O(n log n) rather than the
    O(nm) of a direct comparison. That is what makes a full matrix profile tractable at all: the direct
    route is O(n^2 m) overall and unusable past a few hundred samples.

    A window with zero variance has no z-normalised form. Rather than dividing by zero, its distance is
    set to the maximum possible for the window length, which is the honest reading: a flat window matches
    nothing, because it has no shape to match.
    """
    m = len(query)
    n = len(series)
    q_mean = float(query.mean())
    # The SAME estimator the window statistics use. `_sliding_stats` computes sigma from prefix sums as
    # sqrt(total_sq/m - mean^2); calling `query.std()` here computed the same quantity a different way,
    # and the two were then compared against one absolute constant. On a near-constant window the two
    # estimators disagree in the last digits, so which branch fired depended on rounding.
    q_sigma = float(np.sqrt(max(float((query ** 2).mean()) - q_mean ** 2, 0.0)))

    # Correlate the reversed query with the series: numpy's FFT convolution gives all lags at once.
    product = np.fft.irfft(np.fft.rfft(series, n) * np.fft.rfft(query[::-1], n), n)[m - 1:]

    # RELATIVE floor. An absolute 1e-10 calls a window of millivolt readings flat and a window of
    # kilopascal readings live, for the same physical steadiness. The scale is the series' own spread.
    scale = max(float(np.std(series)), 1.0)
    flat = max(_TINY, scale * 1e-9)
    flat_q = q_sigma < flat
    flat_w = sigma < flat

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (product - m * q_mean * mean) / (m * q_sigma * sigma)
    corr = np.clip(np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    dist = np.sqrt(np.maximum(2 * m * (1.0 - corr), 0.0))

    worst = np.sqrt(2.0 * m)
    if flat_q:
        dist[:] = worst
    dist[flat_w] = worst
    # A flat query matched against a flat window is a genuine match, not a worst case.
    if flat_q:
        dist[flat_w] = 0.0
    return dist


@dataclass
class MatrixProfile:
    """The multidimensional matrix profile and its dimension subsets.

    Parameters
    ----------
    window
        Subsequence length ``m``, in SAMPLES. The single most consequential setting: it decides what
        counts as a pattern. Too short and everything matches something; too long and the profile is
        dominated by whatever happens to be near the ends. It should be chosen from the physics (a haul
        cycle, a thermal time constant), not swept for the best-looking answer.
    exclusion
        Trivial-match exclusion zone as a fraction of ``window``. Without it every subsequence's nearest
        neighbour is the subsequence one sample to its left, the profile is uniformly near zero, and the
        method reports that nothing anywhere is unusual. The 0.5 default follows common practice.
    normalise
        Divide each dimension's distance by the maximum possible for the window, putting all dimensions
        on [0, 1] before they are sorted. Without it, a channel that happens to be noisier contributes
        larger distances at every k and dominates the subset selection through its variance rather than
        through carrying the pattern.
    """

    window: int = 50
    exclusion: float = 0.5
    normalise: bool = True
    _nn_distance: "np.ndarray | None" = None

    def compute(self, series: Series) -> "tuple[np.ndarray, np.ndarray]":
        """Return ``(profile, dimensions)``.

        ``profile`` has shape ``(d, n_sub)``: row ``k-1`` is the k-dimensional matrix profile.
        ``dimensions`` has shape ``(d, n_sub, d)`` and is a boolean mask of which channels each
        subsequence's k-dimensional match used.
        """
        m = self.window
        x = series.x
        n, d = x.shape
        if m < 4:
            raise ValueError(f"window must be at least 4 samples, got {m}")
        if n < 2 * m:
            raise ValueError(f"a series of {n} samples is too short for a window of {m}")
        if not np.all(np.isfinite(x)):
            raise ValueError(
                "mSTAMP needs a complete series; z-normalisation is undefined over a gap. "
                "Drop or interpolate incomplete samples deliberately before calling this."
            )

        n_sub = n - m + 1
        stats = [_sliding_stats(x[:, j], m) for j in range(d)]
        worst = np.sqrt(2.0 * m)
        # No floor of 1 here. `exclusion=0` must mean exactly "only the self-match is removed", which is
        # the literature's no-exclusion baseline and the configuration that demonstrates why the zone is
        # needed at all. Flooring it at 1 silently applied a small exclusion zone even when none was
        # asked for, which made the parameter impossible to test at its own boundary.
        zone = int(np.ceil(m * self.exclusion))

        profile = np.full((d, n_sub), np.inf)
        dims = np.zeros((d, n_sub, d), dtype=bool)
        # Per-dimension distances at the FULL-dimensional nearest neighbour. This is what discord
        # attribution needs and what the k-dimensional subset above cannot give it; see
        # `anomalous_channels` for why the two are different questions.
        nn_distance = np.full((n_sub, d), np.nan)

        for i in range(n_sub):
            per_dim = np.empty((d, n_sub))
            for j in range(d):
                mean, sigma = stats[j]
                per_dim[j] = mass_distance_profile(x[i:i + m, j], x[:, j], mean, sigma)
            if self.normalise:
                per_dim /= worst

            lo, hi = max(0, i - zone), min(n_sub, i + zone + 1)
            per_dim[:, lo:hi] = np.inf

            # Sorting across DIMENSIONS for each candidate j, then a running mean, gives the best-k
            # distance for every k in one pass. This is the step the naive sum-over-all-dimensions
            # generalisation gets wrong, and it is where the dimension subset comes from.
            order = np.argsort(per_dim, axis=0)
            ordered = np.take_along_axis(per_dim, order, axis=0)
            running = np.cumsum(ordered, axis=0) / np.arange(1, d + 1)[:, None]

            for k in range(d):
                jstar = int(np.argmin(running[k]))
                profile[k, i] = running[k, jstar]
                dims[k, i, order[: k + 1, jstar]] = True

            full_star = int(np.argmin(running[d - 1]))
            if np.isfinite(running[d - 1, full_star]):
                nn_distance[i] = per_dim[:, full_star]

        profile[~np.isfinite(profile)] = np.nan
        self._nn_distance = nn_distance
        return profile, dims

    def detect(self, series: Series, k: int = 1) -> Detection:
        """The k-dimensional matrix profile as a detection statistic.

        Larger already means more anomalous: a large profile value says nothing else in the record looks
        like this window, which is the definition of a discord. No sign flip is needed and none is
        applied, which is worth stating because most of the other rungs here do need one.

        The statistic is placed at the window's START index, and the trailing ``m-1`` samples are
        undefined because no window begins there. Placing it at the centre or the end would shift every
        reported onset time by a fixed amount, which is invisible in a plot and fatal in a delay metric.
        """
        if not 1 <= k <= series.d:
            raise ValueError(f"k must be between 1 and {series.d}, got {k}")
        profile, dims = self.compute(series)
        statistic = np.full(series.n, np.nan)
        statistic[: profile.shape[1]] = profile[k - 1]
        return Detection(series.t, statistic, f"mstamp-k{k}", {
            "profile": profile,
            "dimensions": dims,
            "nn_distance": self._nn_distance,
            "window": self.window,
            "k": k,
            "names": series.names,
        })

    @staticmethod
    def matching_subset(detection: Detection, index: int, k: "int | None" = None) -> Attribution:
        """Which channels the k-dimensional MATCH at ``index`` used: mSTAMP's own subset.

        This is the quantity the paper defines, and it answers "in which k channels does this window
        resemble something else in the record". That is a **motif** question.

        Scores are binary: a channel is either in the selected subset or not.

        **Do not use this as anomaly attribution.** See :meth:`anomalous_channels`.
        """
        dims = detection.meta.get("dimensions")
        names = detection.meta.get("names")
        if dims is None or names is None:
            raise ValueError("this detection carries no dimension subsets; it did not come from mSTAMP")
        kk = detection.meta["k"] if k is None else k
        if index >= dims.shape[1]:
            raise ValueError(f"index {index} has no window; the last is {dims.shape[1] - 1}")
        return Attribution(dims[kk - 1, index].astype(float), tuple(names), f"mstamp-k{kk}-subset", index)

    @staticmethod
    def anomalous_channels(detection: Detection, index: int) -> Attribution:
        """Which channels make the window at ``index`` unlike anything else: the discord attribution.

        Per-channel z-normalised distance to the full-dimensional nearest neighbour. Large means that
        channel refuses to match, which is what "this channel is carrying the anomaly" means.

        .. rubric:: Why this is not :meth:`matching_subset`

        mSTAMP's :math:`k`-dimensional subset is the :math:`k` channels with the **smallest** distances,
        because the algorithm was designed to find motifs. At a discord that names the channels which
        still look normal, which is precisely the wrong answer.

        Measured on a four-channel record with a discord planted in channels ``b`` and ``d``: the
        ``k = 2`` matching subset returns ``{a, c}``, the two channels that are still repeating cleanly.
        Reading that as attribution would report the two healthy channels as the culprits, with a
        perfectly sensible-looking discord peak beside it. There is a test pinning both halves.

        The same asymmetry explains which :math:`k` detects a partial fault. :math:`P_k` uses the
        :math:`k` smallest distances, so a fault confined to two of four channels is invisible at
        :math:`k=2` (the two clean channels still match) and shows up at :math:`k=d`, where the failing
        channels can no longer be excluded. **For fault detection, prefer a high k.**
        """
        nn = detection.meta.get("nn_distance")
        names = detection.meta.get("names")
        if nn is None or names is None:
            raise ValueError("this detection carries no dimension subsets; it did not come from mSTAMP")
        if index >= len(nn):
            raise ValueError(f"index {index} has no window; the last is {len(nn) - 1}")
        scores = np.nan_to_num(nn[index], nan=0.0)
        return Attribution(scores, tuple(names), "mstamp-anomalous-channels", index)
