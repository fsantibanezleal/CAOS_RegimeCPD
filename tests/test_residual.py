"""Regressions for RegimeResidualizer's degenerate-scale guard, across every residual method.

Split out from the regime tests because the defect these cover is not about segmentation at all: it is
about what the scale floor is measured RELATIVE to, and it recurs independently in each branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimecpd import RegimeLabels, RegimeResidualizer, Series


class TestDegenerateChannelUnderEveryMethod:
    """A channel that never moves must produce no residual, whichever residual method is used.

    D1 from the 2026-08-10 engine review. `robust_scale` floors the spread at `max(atol, rtol*|loc|)`,
    and the relative term is the whole point: a channel constant at 21610 must be judged against 21610.
    The `linear` branch passed the RESIDUAL mean as the location, which an intercept in the design matrix
    forces to approximately zero, so the floor collapsed to 1e-12 and a ridge-perturbed fit on a dead
    channel produced residuals with a median of 0.74 and a maximum of 5.0.

    The 301 tests around this were green: nothing raises, and every number looks plausible.
    """

    @pytest.mark.parametrize("method", ["zscore", "linear"])
    def test_a_constant_channel_stays_silent(self, method):
        rng = np.random.default_rng(0)
        n = 600
        ctx = rng.normal(size=(n, 1))
        live = 3.0 * ctx[:, 0] + rng.normal(scale=0.4, size=n)
        dead = np.full(n, 21610.0)          # never moves, and is LARGE, which is the trap
        x = np.column_stack([live, dead])
        names = ("live", "dead")

        base = Series(np.arange(n, dtype=float), x, names)
        labels = RegimeLabels(np.zeros(n, dtype=int), n_regimes=1, method="test-single")
        r = RegimeResidualizer(method=method).fit(base, labels, context=ctx)
        out = r.transform(base, labels, context=ctx)

        dead_res = np.abs(out.r[:, 1])
        # Not zero, and it cannot be: a ridge solve on a channel sitting at 21610 leaves float64
        # roundoff, and the floor divides by a relative quantity rather than by infinity. The bar is set
        # at 1e-3 because a LIVE channel's residual is order 1, so this is three orders below anything
        # that could move a multivariate statistic. Before the fix this same channel reached 5.0.
        assert np.nanmax(dead_res) < 1e-3, (
            f"method={method}: a channel that never moves produced a residual of "
            f"{np.nanmax(dead_res):.3g}; the degenerate-scale floor collapsed")
        live_res = np.abs(out.r[:, 0])
        assert np.nanmax(dead_res) < np.nanmax(live_res) / 1000.0, (
            f"method={method}: the dead channel is not negligible against the live one")

    @pytest.mark.parametrize("method", ["zscore", "linear"])
    def test_a_channel_centred_on_zero_is_still_guarded(self, method):
        # The same collapse by another route: |mean| is approximately zero for a channel oscillating
        # about zero, so a floor scaled by the signed mean vanishes even though the channel is live.
        rng = np.random.default_rng(1)
        n = 400
        ctx = rng.normal(size=(n, 1))
        x = np.column_stack([rng.normal(size=n), np.full(n, 0.0)])
        base = Series(np.arange(n, dtype=float), x, ("live", "zero"))
        labels = RegimeLabels(np.zeros(n, dtype=int), n_regimes=1, method="test-single")
        r = RegimeResidualizer(method=method).fit(base, labels, context=ctx)
        out = r.transform(base, labels, context=ctx)
        assert np.nanmax(np.abs(out.r[:, 1])) < 1e-3, f"method={method}: constant-zero channel spoke"
