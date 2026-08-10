"""regimecpd: regime-conditional change-point detection for load-varying machines.

The problem this package exists for, in one paragraph. On a machine whose operating context moves, every
measured channel moves with it. A haul truck's strut pressure, fuel rate and temperatures all rise
together when it climbs a ramp loaded, and they all fall when it returns empty. Run a change detector on
those raw channels and it fires on the ramp, not on a fault. So the pipeline is not "detect"; it is
segment the operating regime, build the residual WITHIN regime, and detect on that:

    raw channels -> regime segmentation -> within-regime residual -> change point -> onset

The package provides both arms of that comparison on purpose. Running the identical detector on raw
channels and on residuals, at a FIXED false-alarm budget, is the only way to find out whether the
regime stage earns its place, and reporting that comparison honestly (including when it does not) is the
point. See ``docs/methods/01_regime-conditioning.md``.

Nothing here is truck-specific. The reference validation runs on turbofan data, which is the evidence
for that claim rather than an assertion of it.
"""

from __future__ import annotations

from .bocpd import BOCPD, StudentTUPM
from .classical import CUSUM, EWMA, PageHinkley, Shewhart
from .conformal import (
    AdaptiveConformal,
    SplitConformal,
    calibration_report,
    conformalise,
)
from .drift import ADWIN, KSWIN, ks_two_sample_pvalue
from .metrics import (
    FleetScore,
    UnitOutcome,
    UnitScore,
    alarm_budget_curve,
    bootstrap_ci,
    candidate_thresholds,
    rising_edges,
    score_fleet,
    score_unit,
    threshold_for_budget,
)
from .mstamp import MatrixProfile, mass_distance_profile
from .novelty import AutoencoderDetector, IsolationForestDetector, OneClassSVMDetector
from .pelt import PELT, optimal_partition, segmentation_error
from .regime import ContextScaler, DiscreteRegimes, KMeansRegimes, kmeans_inertia_sweep
from .residual import RegimeResidualizer, make_arms
from .scaling import degenerate_mask, robust_scale
from .spc import PCAMonitor, chi2_quantile, normal_quantile
from .types import Attribution, Detection, RegimeLabels, Residual, Series

__version__ = "0.09.001"

__all__ = [
    "__version__",
    # the data contract
    "Series",
    "RegimeLabels",
    "Residual",
    "Detection",
    "Attribution",
    # regime segmentation and the residual
    "ContextScaler",
    "DiscreteRegimes",
    "KMeansRegimes",
    "kmeans_inertia_sweep",
    "RegimeResidualizer",
    "make_arms",
    # classical detectors
    "Shewhart",
    "CUSUM",
    "EWMA",
    "PageHinkley",
    # multivariate SPC
    "PCAMonitor",
    "normal_quantile",
    "chi2_quantile",
    # Bayesian online changepoint detection
    "BOCPD",
    "StudentTUPM",
    # retrospective segmentation
    "PELT",
    "optimal_partition",
    "segmentation_error",
    # matrix profile
    "MatrixProfile",
    "mass_distance_profile",
    # streaming drift detectors
    "ADWIN",
    "KSWIN",
    "ks_two_sample_pvalue",
    # healthy-only novelty models (optional extras)
    "IsolationForestDetector",
    "OneClassSVMDetector",
    "AutoencoderDetector",
    # conformal calibration
    "SplitConformal",
    "AdaptiveConformal",
    "conformalise",
    "calibration_report",
    # scaling guard
    "robust_scale",
    "degenerate_mask",
    # measurement
    "UnitOutcome",
    "UnitScore",
    "FleetScore",
    "rising_edges",
    "score_unit",
    "score_fleet",
    "candidate_thresholds",
    "alarm_budget_curve",
    "threshold_for_budget",
    "bootstrap_ci",
]
