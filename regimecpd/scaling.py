"""One shared rule for deciding whether a channel can be standardised at all.

.. rubric:: The bug this module exists to prevent

Every scaler in this package used to guard with ``spread > 0``. That is not enough, and the failure it
lets through is severe and silent.

A channel can be constant to within floating-point noise while still having a strictly positive standard
deviation. On NASA C-MAPSS, ``sensor_06`` holds the value 21.61 for a whole baseline window and reports a
standard deviation of **7.1e-15**, on 43 of 100 units. It passes ``spread > 0``. Dividing by it turns a
deviation of one unit into a z-score of :math:`1.4\\times10^{14}`.

What that did downstream, measured rather than imagined: a CUSUM on such a channel accumulated to
**8.4e12** over a healthy stretch. One unit doing that set the fleet-wide threshold for the whole
benchmark, and detection rate at a fixed false-alarm budget collapsed from 0.79 to 0.07. Every individual
unit still looked fine. The benchmark simply reported that the method did not work.

.. rubric:: The rule

A channel is standardisable only when its spread is meaningfully above the floating-point noise floor of
its own magnitude:

.. math:: \\sigma > \\max(\\texttt{atol},\\ \\texttt{rtol} \\cdot |\\mu|)

Otherwise its scale is set to 1, which leaves it in raw units. A channel that does not move contributes a
residual near zero rather than an enormous one, which is the correct behaviour: no variation means no
information, not infinite sensitivity.

``rtol`` defaults to 1e-8, roughly the square root of double precision. That is far below any real
quantisation step (a sensor reporting to two decimals has a relative spread around 1e-4) and far above
pure floating-point noise, so it separates the two cases without judgement calls about the data.
"""

from __future__ import annotations

import numpy as np

__all__ = ["robust_scale", "degenerate_mask", "RTOL", "ATOL"]

RTOL = 1e-8
ATOL = 1e-12


def degenerate_mask(spread: np.ndarray, location: np.ndarray,
                    rtol: float = RTOL, atol: float = ATOL) -> np.ndarray:
    """True where a channel's spread is indistinguishable from floating-point noise."""
    spread = np.asarray(spread, dtype=float)
    floor = np.maximum(atol, rtol * np.abs(np.asarray(location, dtype=float)))
    return ~(np.isfinite(spread) & (spread > floor))


def robust_scale(spread: np.ndarray, location: np.ndarray,
                 rtol: float = RTOL, atol: float = ATOL) -> np.ndarray:
    """Per-channel divisor: the spread where it is real, 1 where it is noise.

    Returning 1 rather than the tiny spread is the whole point. It keeps a degenerate channel in raw
    units, where its deviations are small, instead of amplifying them by a factor of 1e14.
    """
    spread = np.asarray(spread, dtype=float)
    return np.where(degenerate_mask(spread, location, rtol, atol), 1.0, spread)
