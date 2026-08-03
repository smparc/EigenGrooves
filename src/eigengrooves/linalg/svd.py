"""
SVD front-end: three interchangeable backends behind one function.

======================  ==========================  ==============================
backend                 method                      when to reach for it
======================  ==========================  ==============================
``"jacobi"`` (default)  one-sided Jacobi            always, unless you have a
                                                    reason not to
``"eigh"``              eigendecomposition of A^T A the textbook derivation; fast
                                                    but loses small singular
                                                    values to conditioning
``"randomized"``        Halko-Martinsson-Tropp      m >> n and only the top k
                                                    components are wanted
======================  ==========================  ==============================

All three return the same ``(U, sigma, Vt)`` convention with ``sigma``
descending, and all three are checked against each other and against NumPy in
the test suite.

Sign convention
---------------
The SVD is only unique up to a simultaneous sign flip of ``u_i`` and ``v_i``.
Left unconstrained, the sign is whatever the arithmetic happened to produce,
which means a latent feature documented as "high energy, loud" can come back as
"low energy, quiet" on a different machine or a different data slice. We pin it:
the entry of largest magnitude in each right singular vector is forced positive.
That makes loadings comparable across runs, datasets and backends.
"""

from __future__ import annotations

import numpy as np

from .eigen import symmetric_eigh
from .jacobi_svd import jacobi_svd
from .randomized import randomized_svd

__all__ = ["svd", "canonicalize_signs", "explained_variance_ratio", "BACKENDS"]

BACKENDS = ("jacobi", "eigh", "randomized")


def canonicalize_signs(U: np.ndarray, Vt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fix the sign ambiguity so decompositions are comparable across runs.

    For each component, the largest-magnitude entry of the right singular
    vector is made positive, flipping the matching left singular vector to
    keep ``U Sigma V^T`` unchanged.
    """
    if Vt.shape[0] == 0:
        return U, Vt
    U = U.copy()
    Vt = Vt.copy()
    dominant = np.argmax(np.abs(Vt), axis=1)
    signs = np.sign(Vt[np.arange(Vt.shape[0]), dominant])
    signs[signs == 0] = 1.0
    Vt *= signs[:, None]
    if U.shape[1] == signs.size:
        U *= signs[None, :]
    return U, Vt


def _svd_via_eigh(
    A: np.ndarray, k: int | None, eigen_method: str = "jacobi"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD through the eigendecomposition of ``A^T A``.

    Kept because it is the derivation everyone learns first, and because having
    it side by side with the Jacobi backend is what lets the test suite
    *demonstrate* the accuracy gap rather than assert it. Not the default:
    forming ``A^T A`` squares the condition number, so components below
    ``sqrt(eps) * sigma_max`` come back as noise.
    """
    m, n = A.shape
    AtA = A.T @ A
    eigenvalues, V = symmetric_eigh(AtA, method=eigen_method)

    # Negative eigenvalues here are pure rounding error on a PSD matrix.
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    sigma = np.sqrt(eigenvalues)

    cutoff = max(m, n) * np.finfo(float).eps * (sigma[0] if sigma.size else 0.0)
    rank = int(np.sum(sigma > cutoff))
    if k is not None:
        rank = min(rank, max(int(k), 0))

    sigma = sigma[:rank]
    V = V[:, :rank]

    # u_i = A v_i / sigma_i. Safe because sigma > cutoff > 0 by construction.
    U = (A @ V) / np.where(sigma == 0.0, 1.0, sigma)
    return U, sigma, V.T


def svd(
    A: np.ndarray,
    k: int | None = None,
    backend: str = "jacobi",
    random_state: int | np.random.Generator | None = None,
    canonical_signs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the (optionally truncated) SVD ``A = U diag(sigma) V^T``.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
    k : int, optional
        Keep the leading ``k`` components. ``None`` keeps all numerically
        significant ones. Values above the numerical rank are clamped down.
    backend : {"jacobi", "eigh", "randomized"}
    random_state : int | np.random.Generator | None
        Only used by the randomized backend.
    canonical_signs : bool
        Apply the sign convention described in the module docstring.

    Returns
    -------
    U : np.ndarray, shape (m, r)
    sigma : np.ndarray, shape (r,), descending
    Vt : np.ndarray, shape (r, n)
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("input contains NaN or inf; clean the feature matrix first")
    if k is not None and k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    if backend == "jacobi":
        U, sigma, Vt = jacobi_svd(A, k=k)
    elif backend == "eigh":
        U, sigma, Vt = _svd_via_eigh(A, k=k)
    elif backend == "randomized":
        if k is None:
            raise ValueError("the randomized backend requires an explicit k")
        U, sigma, Vt = randomized_svd(A, k=k, random_state=random_state)
    else:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")

    if canonical_signs:
        U, Vt = canonicalize_signs(U, Vt)
    return U, sigma, Vt


def explained_variance_ratio(
    sigma: np.ndarray, total_sigma: np.ndarray | None = None
) -> np.ndarray:
    """Fraction of total variance carried by each component.

    Parameters
    ----------
    sigma : np.ndarray
        Singular values of interest -- typically the truncated set.
    total_sigma : np.ndarray, optional
        The *full* spectrum, used as the denominator. This argument is the
        whole point of the function.

        Omitting it normalises ``sigma`` against itself, so a truncated input
        always sums to 100%. The original version of this project did exactly
        that and reported "total variance explained: 100.0%" for every run,
        when the true figure for k=5 of 9 components was 58.7%. Pass the full
        spectrum unless you genuinely mean the within-subset shares.

    Returns
    -------
    np.ndarray, same shape as ``sigma``
    """
    sigma = np.asarray(sigma, dtype=float)
    denom_source = sigma if total_sigma is None else np.asarray(total_sigma, dtype=float)
    total = float(np.sum(denom_source**2))
    if total == 0.0:
        return np.zeros_like(sigma)
    return (sigma**2) / total
