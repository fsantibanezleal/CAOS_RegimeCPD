"""Multivariate statistical process control on PCA: Hotelling T-squared, SPE (Q), and contributions.

This is the rung that answers **which variables**, which is what turns an alarm into an action. A
maintenance planner cannot dispatch against "the machine is anomalous"; they can dispatch against "strut
pressure and brake temperature are the channels carrying the excursion".

The construction splits the measurement space in two, and the split is the reason this rung exists at all
rather than being a fancier threshold:

**T-squared** lives in the retained principal subspace. It asks whether the machine is in an unusual
place along the directions of normal variation: in-model, but extreme.

.. math:: T^2_t = \\sum_{i=1}^{a} \\frac{t_{t,i}^2}{\\lambda_i}

**SPE**, also called the Q statistic, lives in the residual subspace. It asks whether the machine has
left the model altogether: a correlation structure that held throughout the baseline has broken.

.. math:: \\mathrm{SPE}_t = \\lVert x_t - \\hat{x}_t \\rVert^2 = \\lVert (I - PP^{\\top}) x_t \\rVert^2

The two catch genuinely different faults and neither subsumes the other. A sensor drifting along a
direction the machine already varies in raises T-squared and leaves SPE flat. A newly broken relationship
between two channels that used to move together raises SPE while every individual channel stays inside
its usual range, so T-squared sees nothing. Reporting only one is the common mistake.

.. rubric:: Contributions, and what they are not

Contribution plots decompose either statistic across the original variables. They are the classical
answer to "which variables", and they **smear**: a single faulty variable spreads contribution onto every
variable correlated with it, because the statistic is computed in a rotated space where the fault is no
longer aligned with one axis.

So a high contribution means "this variable is implicated", never "this variable is the cause". The
honest use is to narrow a candidate set. See Westerhuis, Gurden and Smilde (2000),
doi:10.1016/S0169-7439(00)00062-9, which treats the smearing properly rather than presenting
contributions as diagnosis.

.. rubric:: On the control limits provided here

The classical limits are implemented because they are the published contribution of these papers and
because they let the implementation be checked against a stated property rather than against its own
output. **They are not what this package uses to compare methods.** Comparisons run through the
calibration layer at a common false-alarm budget, because two methods at their own preferred limits tell
you nothing about each other.

Both limits rest on approximations, named where they are used, and both assume multivariate normality of
the baseline. On real telemetry that assumption is wrong often enough that the empirical calibration is
the more trustworthy route even setting the comparison argument aside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .scaling import robust_scale
from .types import Attribution, Detection, Series

__all__ = ["PCAMonitor", "normal_quantile", "chi2_quantile"]


# Coefficients from Peter Acklam's rational approximation to the inverse normal CDF. Implemented here
# because the core of this package is numpy-only and numpy provides no inverse error function; pulling in
# scipy so that a monitoring library can look up one quantile would misplace the dependency boundary.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)


def normal_quantile(p: float) -> float:
    """Inverse standard normal CDF, accurate to about 1.15e-9 in absolute terms.

    Refined by one Halley step against ``math.erfc``, which costs nothing and removes the approximation
    error almost entirely. Verified against published quantiles in the tests rather than against itself.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    elif p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
            (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)

    e = 0.5 * math.erfc(-x / math.sqrt(2)) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def chi2_quantile(p: float, dof: int) -> float:
    """Chi-square quantile by the Wilson-Hilferty transformation.

    .. math:: \\chi^2_{a,p} \\approx a\\left(1 - \\frac{2}{9a} + z_p\\sqrt{\\frac{2}{9a}}\\right)^3

    An approximation, and named as one. It is good to well under a percent for ``dof`` of 5 or more and
    degrades for very small ``dof``, where the chi-square is strongly skewed. Used only for the reference
    control limit, never for a comparison between methods.
    """
    if dof < 1:
        raise ValueError(f"dof must be at least 1, got {dof}")
    z = normal_quantile(p)
    return float(dof * (1 - 2.0 / (9 * dof) + z * math.sqrt(2.0 / (9 * dof))) ** 3)


@dataclass
class PCAMonitor:
    """Hotelling T-squared and SPE on a PCA fitted to a healthy baseline, with contributions.

    Parameters
    ----------
    n_components
        Number of principal components to retain. When ``None``, the smallest number reaching
        ``variance_target`` of the baseline variance is used.
    variance_target
        Explained-variance fraction used to pick ``n_components`` automatically.
    statistic
        Which statistic ``detect`` reports: ``"spe"``, ``"t2"``, or ``"combined"``.

        ``"combined"`` reports the maximum of the two after each is divided by its own reference limit,
        so they are comparable. That is a convenience for a single-number workflow and it is the weakest
        of the three: it inherits the approximations in both limits and it hides which subspace moved.
        Prefer running the two separately and reporting both, which is what the literature does and what
        makes the T-squared against SPE distinction usable.
    """

    n_components: int | None = None
    variance_target: float = 0.95
    statistic: str = "spe"
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    loadings_: np.ndarray | None = None
    eigenvalues_: np.ndarray | None = None
    residual_eigenvalues_: np.ndarray | None = None
    names_: tuple = ()
    n_baseline_: int = 0
    meta_: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.statistic not in {"spe", "t2", "combined"}:
            raise ValueError(f"unknown statistic {self.statistic!r}; use 'spe', 't2' or 'combined'")

    @property
    def n_components_(self) -> int:
        if self.eigenvalues_ is None:
            raise RuntimeError("PCAMonitor used before fit")
        return len(self.eigenvalues_)

    def fit(self, baseline: Series) -> "PCAMonitor":
        """Fit on a HEALTHY baseline. Fitting on the monitored record folds the fault into the model."""
        x = baseline.x
        rows = np.all(np.isfinite(x), axis=1)
        if np.count_nonzero(rows) < 2:
            raise ValueError("the baseline has fewer than two complete samples")
        xb = x[rows]

        self.mean_ = xb.mean(axis=0)
        spread = xb.std(axis=0)
        # Autoscaling, not covariance PCA. Without it the component structure is decided by whichever
        # channel happens to be measured in the largest units, which is a property of the instrumentation
        # rather than of the machine.
        self.scale_ = robust_scale(spread, self.mean_)
        z = (xb - self.mean_) / self.scale_

        # SVD of the centred data rather than eigendecomposition of the covariance: same answer, better
        # conditioned, and it does not require forming a matrix whose condition number is squared.
        _, singular, vt = np.linalg.svd(z, full_matrices=False)
        eigenvalues = (singular ** 2) / max(len(z) - 1, 1)

        total = eigenvalues.sum()
        if self.n_components is None:
            cumulative = np.cumsum(eigenvalues) / total if total > 0 else np.array([1.0])
            a = int(np.searchsorted(cumulative, self.variance_target) + 1)
            a = min(max(a, 1), len(eigenvalues))
        else:
            a = min(max(int(self.n_components), 1), len(eigenvalues))

        # A component with a numerically zero eigenvalue is a direction the baseline never moved in.
        # Dividing by it in T-squared would turn rounding noise into an unbounded statistic, so those
        # directions are pushed into the residual subspace, where SPE handles them correctly.
        tol = max(total, 1.0) * 1e-12
        while a > 1 and eigenvalues[a - 1] <= tol:
            a -= 1
        if eigenvalues[a - 1] <= tol:
            raise ValueError("the baseline has no direction with non-zero variance")

        self.loadings_ = vt[:a].T
        # The RESIDUAL subspace needs the same relative guard the retained one just got. When a channel is
        # an exact linear combination of the others, its residual eigenvalue is roundoff: `spe` returns
        # roundoff and `spe_limit` calibrates itself to the same roundoff, so the two stay comparable and
        # the chart looks calibrated while measuring nothing. Directions below the tolerance are recorded
        # so `spe` can drop them instead of squaring float noise.
        # Recorded so `spe_limit` can refuse to calibrate itself on roundoff. See its docstring.
        self.residual_tol_ = tol
        self.eigenvalues_ = eigenvalues[:a]
        self.residual_eigenvalues_ = eigenvalues[a:]
        self.names_ = baseline.names
        self.n_baseline_ = int(len(z))
        self.meta_ = {
            "n_components": a,
            "explained_variance": float(eigenvalues[:a].sum() / total) if total > 0 else 1.0,
        }
        return self

    def _z(self, series: Series) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("PCAMonitor used before fit")
        if series.names != self.names_:
            raise ValueError(f"fitted on channels {self.names_}, asked to use {series.names}")
        return (series.x - self.mean_) / self.scale_

    def scores(self, series: Series) -> np.ndarray:
        """Projections onto the retained components, shape ``(n, a)``."""
        return self._z(series) @ self.loadings_

    def t2(self, series: Series) -> np.ndarray:
        """Hotelling T-squared per sample. NaN where the sample is not complete."""
        z = self._z(series)
        out = np.full(len(z), np.nan)
        ok = np.all(np.isfinite(z), axis=1)
        if ok.any():
            t = z[ok] @ self.loadings_
            out[ok] = np.sum(t ** 2 / self.eigenvalues_, axis=1)
        return out

    def spe(self, series: Series) -> np.ndarray:
        """Squared prediction error (the Q statistic) per sample. NaN where the sample is not complete."""
        z = self._z(series)
        out = np.full(len(z), np.nan)
        ok = np.all(np.isfinite(z), axis=1)
        if ok.any():
            # The FULL residual, deliberately. An earlier attempt at the degeneracy fix dropped the
            # residual directions with no baseline variance; that destroys the statistic, because a break
            # in an exact linear relationship among channels lands entirely in one of those directions.
            # SPE exists to catch exactly that, so the guard belongs on the LIMIT, not here.
            residual = z[ok] - (z[ok] @ self.loadings_) @ self.loadings_.T
            out[ok] = np.sum(residual ** 2, axis=1)
        return out

    def t2_limit(self, alpha: float = 0.05) -> float:
        """Reference control limit for T-squared, by the chi-square approximation.

        Exact for a known covariance; here the covariance is estimated, so the F-distribution form is the
        stricter statement. With a baseline of a few thousand samples the difference is small, and the
        approximation is named rather than hidden. Use the calibration layer for anything comparative.
        """
        return chi2_quantile(1.0 - alpha, self.n_components_)

    def spe_limit(self, alpha: float = 0.05) -> float:
        """Reference control limit for SPE, by Jackson and Mudholkar (1979).

        .. math::
            Q_\\alpha = \\theta_1 \\left[ \\frac{c_\\alpha \\sqrt{2\\theta_2 h_0^2}}{\\theta_1}
            + 1 + \\frac{\\theta_2 h_0 (h_0 - 1)}{\\theta_1^2} \\right]^{1/h_0}

        with :math:`\\theta_i = \\sum_{j>a} \\lambda_j^i` and
        :math:`h_0 = 1 - \\frac{2\\theta_1\\theta_3}{3\\theta_2^2}`.

        Returns 0 when the residual subspace is empty (every component retained), because there is then
        no residual to exceed anything.

        Source: Jackson, J. E. and Mudholkar, G. S. "Control Procedures for Residuals Associated With
        Principal Component Analysis." *Technometrics* **21**(3), 341-349 (1979).
        doi:10.1080/00401706.1979.10489779
        """
        lam = self.residual_eigenvalues_
        if lam is None or lam.size == 0:
            return 0.0
        # A residual subspace whose eigenvalues are ROUNDOFF cannot produce a meaningful limit. When a
        # channel is an exact linear combination of the others, the residual variance is around 1e-30:
        # `spe` returns roundoff, this expansion scales itself to the same roundoff, and the two remain
        # comparable, so the chart looks calibrated while measuring nothing. NaN says "undefined", which
        # a caller can see; a tiny number says "calibrated", which is a lie.
        tol = getattr(self, "residual_tol_", 0.0)
        if float(lam.sum()) <= max(tol, 0.0):
            return float("nan")
        theta1, theta2, theta3 = lam.sum(), (lam ** 2).sum(), (lam ** 3).sum()
        if theta1 <= 0 or theta2 <= 0:
            return 0.0
        h0 = 1.0 - (2.0 * theta1 * theta3) / (3.0 * theta2 ** 2)
        if abs(h0) < 1e-6:      # the expansion degenerates; fall back to the leading term
            h0 = 1e-6
        c = normal_quantile(1.0 - alpha)
        inner = (c * math.sqrt(2.0 * theta2 * h0 ** 2) / theta1
                 + 1.0
                 + theta2 * h0 * (h0 - 1.0) / theta1 ** 2)
        return float(theta1 * max(inner, 0.0) ** (1.0 / h0))

    def detect(self, series: Series, alpha: float = 0.05) -> Detection:
        """The configured statistic as a :class:`~regimecpd.types.Detection`.

        ``meta`` always carries both ``t2`` and ``spe`` arrays, so a caller can inspect the other
        subspace without recomputing the projection.
        """
        t2 = self.t2(series)
        spe = self.spe(series)
        meta = {"t2": t2, "spe": spe, **self.meta_,
                "t2_limit": self.t2_limit(alpha), "spe_limit": self.spe_limit(alpha)}

        if self.statistic == "t2":
            return Detection(series.t, t2, "pca-t2", meta)
        if self.statistic == "spe":
            return Detection(series.t, spe, "pca-spe", meta)

        t2_lim = max(meta["t2_limit"], 1e-12)
        spe_lim = max(meta["spe_limit"], 1e-12)
        combined = np.fmax(t2 / t2_lim, spe / spe_lim)
        return Detection(series.t, combined, "pca-combined", meta)

    def contributions(self, series: Series, index: int, kind: str = "spe") -> Attribution:
        """Per-variable contribution to the statistic at one sample.

        ``kind="spe"`` gives the squared residual per variable, which decomposes SPE exactly:

        .. math:: c_j = \\left( z_{t,j} - \\hat{z}_{t,j} \\right)^2, \\qquad \\sum_j c_j = \\mathrm{SPE}_t

        ``kind="t2"`` gives the Kourti and MacGregor form, summed over retained components:

        .. math:: c_j = \\left| \\sum_{i=1}^{a} \\frac{t_{t,i}}{\\lambda_i} p_{ji} z_{t,j} \\right|

        The absolute value is taken because a signed contribution can cancel across components and report
        a variable as uninvolved when it is carrying the excursion in both directions.

        **These smear.** A single faulty variable spreads contribution onto every variable correlated
        with it. The result narrows a candidate set; it does not identify a cause. Cross-check it against
        an attribution obtained by an unrelated route before acting on it.
        """
        if kind not in {"spe", "t2"}:
            raise ValueError(f"unknown kind {kind!r}; use 'spe' or 't2'")
        z = self._z(series)[index]
        if not np.all(np.isfinite(z)):
            raise ValueError(f"sample {index} is not complete, so it has no contribution decomposition")

        if kind == "spe":
            scores = (z - (z @ self.loadings_) @ self.loadings_.T) ** 2
        else:
            t = z @ self.loadings_
            scores = np.abs(((t / self.eigenvalues_) * self.loadings_).sum(axis=1) * z)

        return Attribution(scores, self.names_, f"pca-{kind}-contribution", index)
