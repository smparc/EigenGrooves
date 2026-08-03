"""
Choosing the number of latent dimensions.

The original project hardcoded ``k = 5`` with no justification. Five is a
defensible guess, but "how many components actually carry signal rather than
noise" is a question with real answers, and which one you pick changes the
recommendations. This module implements three, so the choice becomes an
argument you can make rather than a constant you have to defend.

``variance``
    Smallest ``k`` whose cumulative explained variance reaches a threshold
    (default 90%). Simple, interpretable, and the threshold is arbitrary.

``elbow``
    The knee of the scree curve, found as the point of maximum perpendicular
    distance from the line joining its endpoints. No threshold to pick, but it
    assumes there *is* a knee -- on a flat spectrum the answer is meaningless.

``gavish_donoho``
    The optimal hard threshold for singular value truncation. Given the
    assumption that the data is low-rank signal plus i.i.d. Gaussian noise,
    this is the provably optimal cut point in the asymptotic mean-squared-error
    sense. It needs no threshold at all -- the aspect ratio of the matrix
    determines everything.

Reference
---------
Gavish, M., & Donoho, D. L. (2014). "The Optimal Hard Threshold for Singular
Values is 4/sqrt(3)." IEEE Transactions on Information Theory, 60(8), 5040-5053.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "STRATEGIES",
    "RankSelection",
    "rank_by_elbow",
    "rank_by_gavish_donoho",
    "rank_by_variance",
    "select_rank",
]

STRATEGIES = ("variance", "elbow", "gavish_donoho")


@dataclass(frozen=True)
class RankSelection:
    """The chosen rank plus enough context to explain the choice."""

    k: int
    strategy: str
    detail: str
    cumulative_variance: float

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"k={self.k} via {self.strategy} ({self.detail}); "
            f"retains {self.cumulative_variance * 100:.1f}% of variance"
        )


def _as_spectrum(sigma: np.ndarray) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float).ravel()
    if sigma.size and np.any(np.diff(sigma) > 1e-9 * max(sigma[0], 1.0)):
        raise ValueError("singular values must be in descending order")
    return sigma


def rank_by_variance(sigma: np.ndarray, threshold: float = 0.90) -> int:
    """Smallest ``k`` retaining at least ``threshold`` of the total variance."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    sigma = _as_spectrum(sigma)
    if sigma.size == 0:
        return 0
    energy = sigma**2
    total = energy.sum()
    if total == 0.0:
        return 0
    cumulative = np.cumsum(energy) / total
    # searchsorted finds the first index meeting the threshold; +1 converts an
    # index to a count.
    return int(np.searchsorted(cumulative, threshold) + 1)


def rank_by_elbow(sigma: np.ndarray) -> int:
    """Knee of the scree curve by maximum distance from the endpoint chord.

    Draw the line from ``(1, sigma_1)`` to ``(n, sigma_n)``; the elbow is the
    point furthest from it. Degenerate spectra (fewer than three components, or
    a perfectly straight scree line) fall back to the full rank.
    """
    sigma = _as_spectrum(sigma)
    n = sigma.size
    if n < 3:
        return int(n)

    x = np.arange(n, dtype=float)
    y = sigma.astype(float)
    # Normalise both axes so the distance is not dominated by whichever happens
    # to have the larger units.
    x_span = x[-1] - x[0]
    y_span = y[0] - y[-1]
    if x_span == 0 or y_span == 0:
        return int(n)
    xn = (x - x[0]) / x_span
    yn = (y - y[-1]) / y_span

    # Perpendicular distance to the chord from (0, 1) to (1, 0), i.e. the line
    # x + y - 1 = 0.
    distance = np.abs(xn + yn - 1.0) / np.sqrt(2.0)

    # The argmax is the corner itself -- the first component sitting on the
    # flat tail. Signal is everything *before* it, so the index doubles as the
    # count. For [100, 90, 80, 4, 3.5, ...] the corner is index 3 and the
    # answer is 3, not 4. Never return 0: one component is the floor.
    return max(int(np.argmax(distance)), 1)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration of ``y`` over ``x``.

    Written out rather than delegating to NumPy: ``np.trapezoid`` only exists
    from NumPy 2.0, and ``np.trapz`` is deprecated in it, so calling either
    would silently constrain which NumPy versions this package supports. The
    rule is three lines, and everything else numerical here is from scratch
    anyway.
    """
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def _median_marcenko_pastur(beta: float) -> float:
    """Median of the Marcenko-Pastur distribution with aspect ratio ``beta``.

    Found by bisection on the CDF, integrated numerically. Used to convert the
    observed median singular value into an estimate of the noise level.
    """
    lower = (1.0 - np.sqrt(beta)) ** 2
    upper = (1.0 + np.sqrt(beta)) ** 2

    def cdf(x: float) -> float:
        grid = np.linspace(lower, min(x, upper), 4096)
        if grid.size < 2 or x <= lower:
            return 0.0
        density = np.sqrt(np.maximum((upper - grid) * (grid - lower), 0.0)) / (
            2.0 * np.pi * beta * grid
        )
        return _trapezoid(density, grid)

    lo, hi = lower, upper
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < 0.5:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rank_by_gavish_donoho(
    sigma: np.ndarray, n_rows: int, n_cols: int, noise_sigma: float | None = None
) -> int:
    """Optimal hard threshold for singular value truncation.

    Parameters
    ----------
    sigma : np.ndarray
        Full spectrum, descending.
    n_rows, n_cols : int
        Shape of the matrix the spectrum came from. The aspect ratio
        ``beta = min/max`` drives the threshold.
    noise_sigma : float, optional
        Known noise standard deviation. If omitted it is estimated from the
        median singular value, which is the usual practical case.

    Returns
    -------
    int
        Number of singular values above the threshold.
    """
    sigma = _as_spectrum(sigma)
    if sigma.size == 0:
        return 0
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("n_rows and n_cols must be positive")

    beta = min(n_rows, n_cols) / max(n_rows, n_cols)

    if noise_sigma is not None:
        # Known-noise case: threshold = lambda(beta) * sqrt(n) * noise.
        lam = np.sqrt(
            2.0 * (beta + 1.0)
            + (8.0 * beta) / ((beta + 1.0) + np.sqrt(beta**2 + 14.0 * beta + 1.0))
        )
        cutoff = lam * np.sqrt(max(n_rows, n_cols)) * noise_sigma
    else:
        # Unknown-noise case: omega(beta) * median(sigma). Gavish & Donoho give
        # a polynomial approximation to omega; we compute it exactly instead,
        # via omega = lambda(beta) / sqrt(median of Marcenko-Pastur).
        lam = np.sqrt(
            2.0 * (beta + 1.0)
            + (8.0 * beta) / ((beta + 1.0) + np.sqrt(beta**2 + 14.0 * beta + 1.0))
        )
        mu = _median_marcenko_pastur(beta)
        omega = lam / np.sqrt(mu)
        cutoff = omega * float(np.median(sigma))

    return int(max(np.sum(sigma > cutoff), 1))


def select_rank(
    sigma: np.ndarray,
    strategy: str | int = "variance",
    n_rows: int | None = None,
    n_cols: int | None = None,
    variance_threshold: float = 0.90,
    noise_sigma: float | None = None,
) -> RankSelection:
    """Pick a rank and report how the choice was made.

    Parameters
    ----------
    sigma : np.ndarray
        The *full* spectrum, descending.
    strategy : {"variance", "elbow", "gavish_donoho"} | int
        An integer is taken as an explicit rank and passed through (clamped to
        the available range), which keeps the "I know what I want" path honest
        rather than special-cased by the caller.
    n_rows, n_cols : int, optional
        Required by ``gavish_donoho``.
    variance_threshold : float
        Used by ``variance``.
    noise_sigma : float, optional
        Used by ``gavish_donoho``.

    Returns
    -------
    RankSelection
    """
    sigma = _as_spectrum(sigma)
    n_available = sigma.size

    def _finish(k: int, name: str, detail: str) -> RankSelection:
        k = int(np.clip(k, 0, n_available))
        energy = sigma**2
        total = energy.sum()
        cum = float(energy[:k].sum() / total) if total > 0 else 0.0
        return RankSelection(k=k, strategy=name, detail=detail, cumulative_variance=cum)

    if isinstance(strategy, (int, np.integer)) and not isinstance(strategy, bool):
        requested = int(strategy)
        if requested < 0:
            raise ValueError(f"explicit rank must be non-negative, got {requested}")
        detail = "explicitly requested"
        if requested > n_available:
            detail = f"requested {requested}, clamped to available rank {n_available}"
        return _finish(requested, "explicit", detail)

    if strategy == "variance":
        k = rank_by_variance(sigma, variance_threshold)
        return _finish(k, "variance", f"threshold={variance_threshold:.0%}")

    if strategy == "elbow":
        k = rank_by_elbow(sigma)
        return _finish(k, "elbow", "max distance from scree chord")

    if strategy == "gavish_donoho":
        if n_rows is None or n_cols is None:
            raise ValueError("gavish_donoho requires n_rows and n_cols")
        k = rank_by_gavish_donoho(sigma, n_rows, n_cols, noise_sigma)
        beta = min(n_rows, n_cols) / max(n_rows, n_cols)
        known = "known" if noise_sigma is not None else "estimated"
        return _finish(k, "gavish_donoho", f"beta={beta:.3f}, {known} noise")

    raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES} or an int")
