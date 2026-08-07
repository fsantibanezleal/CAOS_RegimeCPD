"""Bayesian online changepoint detection (Adams and MacKay, 2007).

Returns the **run-length posterior** :math:`p(r_t \\mid x_{1:t})`, the distribution over time since the
last changepoint, rather than a binary flag. That is the whole reason to reach for this method over a
threshold: it can say how sure it is that the change started here, and how sure it is that it started
somewhere else instead.

.. rubric:: The recursion

With :math:`H(r)` the hazard function and :math:`\\pi_t^{(r)} = p(x_t \\mid \\text{run } r)` the
predictive probability under each run hypothesis:

.. math::
    p(r_t = r_{t-1}+1,\\, x_{1:t}) &= p(r_{t-1},\\, x_{1:t-1})\\, \\pi_t^{(r)}\\, (1 - H(r_{t-1}+1)) \\\\
    p(r_t = 0,\\, x_{1:t}) &= \\sum_{r_{t-1}} p(r_{t-1},\\, x_{1:t-1})\\, \\pi_t^{(r)}\\, H(r_{t-1}+1)

normalised by the evidence :math:`p(x_{1:t})` to give the posterior. Inference is **exact for the most
recent changepoint**, online, by message passing. It is not a retrospective segmentation, which is why
PELT is a separate rung rather than a substitute for this one.

.. rubric:: Why the modularity matters here specifically

Adams and MacKay factor the algorithm so the predictive model is swappable while the recursion stays
fixed. That is not a convenience in this package, it is what makes its central comparison fair.

Arm A of the comparison runs this detector on raw channels; arm B runs it on within-regime residuals.
Because the observation model is the only thing that differs, the comparison is **the same recursion, the
same hazard function, a different observation model**. Nobody can object that two different pipelines
were compared and the better-engineered one won.

.. rubric:: The assumption, stated where it can be seen

The derivation assumes the model parameters **before and after a changepoint are independent**. That is a
real modelling assumption, not a formality. It is well suited to a step change in a machine's behaviour
and poorly suited to a slow drift whose post-change state is nearly its pre-change state, where the
evidence for a changepoint at any particular instant stays weak no matter how far the drift eventually
travels. Expect BOCPD to be strong on abrupt onsets and unexceptional on gradual ones, and to say so
through a diffuse posterior rather than by being quietly wrong.

Source: Adams, R. P. and MacKay, D. J. C. "Bayesian Online Changepoint Detection."
arXiv:0710.3742 [stat.ML], 19 October 2007. A preprint with no journal version; cite it as such.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .types import Detection, Series

__all__ = ["StudentTUPM", "BOCPD"]


@dataclass
class StudentTUPM:
    """Gaussian observations with unknown mean and variance, under a Normal-Inverse-Gamma prior.

    The conjugate predictive is a Student-t, which is the right shape for this job: it has heavier tails
    than a Gaussian, so an early run with two observations behind it does not declare a changepoint every
    time a slightly unusual sample arrives. A Gaussian predictive with a plugged-in variance estimate
    produces exactly that failure, and it looks like sensitivity rather than like a modelling error.

    Channels are treated as **conditionally independent given the run length**, so the joint predictive
    is the product across channels and every channel shares ONE run-length posterior. That is what the
    product needs: one onset time for the machine, not one per channel. It does mean cross-channel
    correlation is not modelled here, which is what the PCA rung is for.

    Parameters
    ----------
    mu0, kappa0, alpha0, beta0
        Normal-Inverse-Gamma prior. ``kappa0`` is the prior strength on the mean and ``alpha0``,
        ``beta0`` on the variance. The defaults are weak on the mean and proper on the variance; inputs
        are expected to be standardised against a baseline, which is what makes a fixed prior reasonable
        across channels at all.
    """

    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0: float = 1.0
    _mu: np.ndarray | None = field(default=None, repr=False)
    _beta: np.ndarray | None = field(default=None, repr=False)
    _runs: np.ndarray | None = field(default=None, repr=False)
    _const: np.ndarray | None = field(default=None, repr=False)
    _max_runs: int = 0
    _d: int = 0

    def start(self, d: int, max_runs: int) -> "StudentTUPM":
        """Initialise for ``d`` channels, precomputing every run-length-dependent constant.

        The Student-t log density needs ``lgamma``, which numpy does not vectorise. Naively that is one
        ``lgamma`` call per run hypothesis per sample, tens of millions on a long record in pure Python.

        It is avoidable exactly: after ``r`` observations the shape parameter is ``alpha0 + r/2`` and the
        mean strength is ``kappa0 + r``, both determined by the run length alone. So every gamma term is
        precomputed once on a grid of ``r`` and looked up by run length. This is not an approximation.
        """
        self._d = d
        self._max_runs = max_runs
        self._mu = np.full((1, d), float(self.mu0))
        self._beta = np.full((1, d), float(self.beta0))
        # The run length of each hypothesis, tracked EXPLICITLY rather than inferred from array position.
        # Once the posterior is pruned the survivors are no longer a contiguous 0, 1, 2, ... block, so
        # position and run length part company. Reading the shape parameters by position after that gives
        # every hypothesis the wrong prior strength, and the symptom is a changepoint probability pinned
        # at exactly the hazard rate: the predictive stops discriminating between hypotheses, which looks
        # like a plausible constant rather than like a bug.
        self._runs = np.zeros(1, dtype=int)

        r = np.arange(max_runs + 2, dtype=float)
        alpha = self.alpha0 + r / 2.0
        nu = 2.0 * alpha
        gamma_terms = np.array([math.lgamma((v + 1.0) / 2.0) - math.lgamma(v / 2.0) for v in nu])
        self._const = gamma_terms - 0.5 * np.log(nu * math.pi)
        return self

    def _alpha_kappa(self) -> "tuple[np.ndarray, np.ndarray]":
        r = self._runs.astype(float)
        return self.alpha0 + r / 2.0, self.kappa0 + r

    def log_pred_prob(self, x: np.ndarray) -> np.ndarray:
        """Log joint predictive of one observation under each run hypothesis, shape ``(n_runs,)``."""
        alpha, kappa = self._alpha_kappa()
        # Predictive scale of the Normal-Inverse-Gamma: beta (kappa + 1) / (alpha kappa).
        scale2 = self._beta * ((kappa + 1.0) / (alpha * kappa))[:, None]
        nu = (2.0 * alpha)[:, None]
        resid2 = (x[None, :] - self._mu) ** 2
        per_channel = (self._const[np.minimum(self._runs, self._max_runs + 1)][:, None]
                       - 0.5 * np.log(scale2)
                       - ((nu + 1.0) / 2.0) * np.log1p(resid2 / (nu * scale2)))
        return per_channel.sum(axis=1)

    def update(self, x: np.ndarray) -> None:
        """Grow every run hypothesis by one observation, and prepend a fresh run at length zero."""
        _, kappa = self._alpha_kappa()
        k = kappa[:, None]
        new_mu = (k * self._mu + x[None, :]) / (k + 1.0)
        new_beta = self._beta + k * (x[None, :] - self._mu) ** 2 / (2.0 * (k + 1.0))
        self._mu = np.vstack([np.full((1, self._d), float(self.mu0)), new_mu])
        self._beta = np.vstack([np.full((1, self._d), float(self.beta0)), new_beta])
        self._runs = np.concatenate([[0], self._runs + 1])

    def keep(self, index: np.ndarray) -> None:
        """Retain only the given run hypotheses, used when the posterior is pruned."""
        self._mu = self._mu[index]
        self._beta = self._beta[index]
        self._runs = self._runs[index]

    @property
    def run_lengths(self) -> np.ndarray:
        """The run length each surviving hypothesis stands for."""
        return self._runs


@dataclass
class BOCPD:
    """Bayesian online changepoint detection with a swappable predictive model.

    Parameters
    ----------
    hazard_rate
        Constant hazard :math:`H(r) = 1/\\lambda`. ``hazard_rate`` is :math:`\\lambda`, the prior mean
        run length, expressed in SAMPLES. Setting it far below the true inter-change interval makes the
        posterior restless; setting it far above makes the detector reluctant. It is a prior, and like
        any prior it should be set from what is known about the machine rather than tuned until the
        answer looks right.
    max_runs
        Cap on retained run hypotheses. The exact posterior grows by one hypothesis per sample forever,
        so some cap is unavoidable on a long record.
    prune_threshold
        Hypotheses holding less than this posterior mass are dropped and the remainder renormalised.
    statistic
        ``"short_run"`` (default) reports :math:`p(r_t \\le w \\mid x_{1:t})`, the posterior mass on short
        runs, with :math:`w` set by ``short_run_window``. ``"surprise"`` reports
        :math:`-\\log p(x_t \\mid x_{1:t-1})`, unbounded, which often calibrates better precisely because
        it does not saturate near 1. ``"changepoint"`` reports :math:`p(r_t = 0 \\mid x_{1:t})` and is
        **deliberately useless**; see below.
    warmup
        Samples at the start of a record for which the statistic is reported as undefined.

        BOCPD has a genuine warm-up that the control charts do not, because it learns its parameters
        online rather than receiving them from a ``fit`` on a baseline. Two things are wrong at the very
        start: the posterior has accumulated no evidence yet, and more sharply,
        :math:`p(r_t \\le w)` is **structurally 1** while :math:`t \\le w`, since the run cannot be longer
        than the record. Without a warm-up, the largest value of the statistic on any record is at index
        0, which is the record beginning rather than a fault. That is a real defect this package hit and
        fixed rather than a hypothetical.

    .. rubric:: Why the changepoint probability is NOT the detection statistic

    The obvious reading of this algorithm is that it hands you :math:`p(r_t = 0 \\mid x_{1:t})`, the
    probability that a changepoint just happened, and that you threshold it. Under a **constant hazard**
    that quantity is identically the hazard rate, at every step, on every dataset:

    .. math::
        p(r_t = 0 \\mid x_{1:t}) = \\frac{\\sum_r p(r_{t-1}=r)\\,\\pi_t^{(r)}\\,H}
                                        {\\sum_r p(r_{t-1}=r)\\,\\pi_t^{(r)}} = H

    because both branches of the recursion weight the same predictive :math:`\\pi_t^{(r)}`, and the
    hazard factors straight out. The observation :math:`x_t` is evaluated under the OLD run's parameters
    in both branches, so it cannot yet distinguish them. Evidence that the run is new only accumulates on
    the following steps, as the long-run hypotheses are crushed by the predictive.

    This is not a defect in the algorithm and it is not a bug in this implementation. It is a trap in the
    reading, and it produces a detector whose statistic is a flat line at 0.005 while every posterior
    plot beside it looks perfectly sensible. ``"changepoint"`` is kept, and there is a test asserting it
    equals the hazard exactly, so that the property is recorded rather than rediscovered.

    ``"short_run"`` is the informative version of the same idea: after a real change, mass collapses onto
    short runs and stays there for several steps.

    Notes
    -----
    ``Detection.meta["run_length_posterior"]`` holds the full posterior as an ``(n, max_runs+1)`` array
    indexed by TRUE run length, so a consumer can plot the distribution rather than a line. Shipping only
    the reduced statistic would throw away the reason this method was chosen.
    """

    hazard_rate: float = 250.0
    max_runs: int = 500
    prune_threshold: float = 1e-6
    statistic: str = "short_run"
    short_run_window: int = 5
    warmup: int = 20
    upm: StudentTUPM = field(default_factory=StudentTUPM)

    def __post_init__(self) -> None:
        if self.statistic not in {"short_run", "changepoint", "surprise"}:
            raise ValueError(
                f"unknown statistic {self.statistic!r}; use 'short_run', 'surprise' or 'changepoint'"
            )
        if self.hazard_rate <= 1.0:
            raise ValueError(f"hazard_rate is a prior mean run length in samples, got {self.hazard_rate}")
        if self.short_run_window < 0:
            raise ValueError(f"short_run_window must be non-negative, got {self.short_run_window}")

    def detect(self, series: Series) -> Detection:
        """Run the recursion over ``series``.

        There is no ``fit``: BOCPD is online and learns its own parameters from the record as it goes,
        which is a genuine difference from the control charts and is why the two calibrate differently.
        Standardising the input against a healthy baseline first is still sensible, and is what the
        shared prior assumes.
        """
        hazard = 1.0 / self.hazard_rate
        upm = self.upm.start(series.d, self.max_runs)

        n = series.n
        posterior = np.zeros((n, self.max_runs + 1))
        changepoint = np.full(n, np.nan)
        surprise = np.full(n, np.nan)

        # The run-length posterior itself, held NORMALISED at every step. Carrying an unnormalised joint
        # instead means the max-subtraction used for numerical stability accumulates across steps, and
        # the evidence ratio that gives the surprise stops being a real log probability. Normalised, the
        # one-step evidence is simply the total mass before renormalising, which is exact.
        joint = np.array([1.0])

        def _scatter(mass: np.ndarray, runs: np.ndarray, row: int) -> None:
            # Mass is written at the TRUE run length, not at the array position, so a pruned posterior
            # still plots against a meaningful axis. Hypotheses past the cap are folded into the last
            # bin rather than dropped, which keeps each row a distribution.
            np.add.at(posterior[row], np.minimum(runs, self.max_runs), mass)

        for i in range(n):
            x = series.x[i]
            if not np.all(np.isfinite(x)):
                # An incomplete observation carries no information about the run length. Updating on it
                # would be inventing an observation; skipping keeps the recursion exact for the samples
                # that do exist, and the statistic is undefined here rather than fabricated.
                _scatter(joint, upm.run_lengths, i)
                continue

            log_pred = upm.log_pred_prob(x)
            # Subtracting the maximum before exponentiating: a run with a thousand observations behind it
            # can have a log predictive far from a fresh one, and the difference underflows to zero
            # without it, silently deleting hypotheses the recursion still needs.
            shift = float(log_pred.max())
            pred = np.exp(log_pred - shift)

            growth = joint * pred * (1.0 - hazard)
            born = float(np.sum(joint * pred * hazard))
            new_joint = np.concatenate([[born], growth])

            total = float(new_joint.sum())
            if total <= 0 or not np.isfinite(total):
                # Every hypothesis has become numerically impossible. Restarting from a single fresh run
                # is the honest recovery: it says "no run hypothesis explains this", rather than
                # continuing to report a posterior derived from zeros.
                upm.start(series.d, self.max_runs)
                joint = np.array([1.0])
                _scatter(joint, upm.run_lengths, i)
                continue

            # `joint` sums to one, so the one-step evidence p(x_t | x_1:t-1) is exactly total * e^shift.
            surprise[i] = -(math.log(total) + shift)
            normalised = new_joint / total
            changepoint[i] = normalised[0]

            upm.update(x)
            runs = upm.run_lengths

            keep_mask = normalised >= self.prune_threshold
            keep_mask[0] = True                       # the fresh run is never pruned
            if int(keep_mask.sum()) > self.max_runs + 1:
                cutoff = np.argsort(-normalised)[: self.max_runs + 1]
                keep_mask = np.zeros_like(keep_mask)
                keep_mask[cutoff] = True
            index = np.flatnonzero(keep_mask)

            _scatter(normalised[index] / normalised[index].sum(), runs[index], i)

            upm.keep(index)
            joint = normalised[index] / normalised[index].sum()

        w = min(self.short_run_window, self.max_runs)
        short_run = posterior[:, : w + 1].sum(axis=1)
        # Rows the recursion never scored (an incomplete observation) carry the carried-forward posterior
        # rather than a fresh measurement, so the statistic is undefined there for consistency with every
        # other detector in the package.
        short_run = np.where(np.isnan(surprise), np.nan, short_run)

        # The warm-up. Structurally, p(r <= w) is 1 while t <= w, so without this the largest value on
        # every record is at index 0 and every detection lands on the record beginning.
        cut = min(max(self.warmup, w + 1), n)
        short_run[:cut] = np.nan
        surprise[:cut] = np.nan
        changepoint[:cut] = np.nan

        value = {"short_run": short_run, "changepoint": changepoint, "surprise": surprise}[self.statistic]
        return Detection(series.t, value, f"bocpd-{self.statistic}", {
            "run_length_posterior": posterior,
            "short_run_probability": short_run,
            "changepoint_probability": changepoint,
            "surprise": surprise,
            "hazard_rate": self.hazard_rate,
            "short_run_window": w,
            "warmup": cut,
        })
