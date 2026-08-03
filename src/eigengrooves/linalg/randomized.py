"""
Randomized SVD (Halko, Martinsson & Tropp, 2011), implemented from scratch.

For a catalogue of a few thousand songs and nine features, a dense SVD is
instantaneous and this module is unnecessary. It exists because the interesting
version of this problem is not 9 features -- it is a song-by-song or
song-by-tag matrix with millions of rows, where forming the full decomposition
is hopeless and only the leading ``k`` components are wanted anyway.

The idea: if ``A`` has rapidly decaying spectrum, a random projection of its
columns captures its dominant range with overwhelming probability. Build a
small orthonormal basis ``Q`` for that range, project ``A`` onto it, and take
an exact SVD of the resulting tiny matrix::

    Omega ~ N(0, 1), shape (n, k + p)
    Y = A Omega                      -> samples the range of A
    Q = orth(Y)                      -> orthonormal basis, shape (m, k + p)
    B = Q^T A                        -> small, shape (k + p, n)
    B = U_B Sigma V^T                -> exact SVD of the small matrix
    U = Q U_B

Cost is ``O(m n (k + p))`` instead of ``O(m n^2)``. Oversampling by ``p``
(default 10) makes the range estimate robust; power iterations sharpen the
spectral decay when it is slow, at the cost of one extra pass each.

Reference
---------
Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). "Finding Structure with
Randomness: Probabilistic Algorithms for Constructing Approximate Matrix
Decompositions." SIAM Review 53(2), 217-288.
"""

from __future__ import annotations

import numpy as np

from .jacobi_svd import jacobi_svd
from .qr import householder_qr

__all__ = ["randomized_svd"]


def randomized_svd(
    A: np.ndarray,
    k: int,
    n_oversamples: int = 10,
    n_power_iterations: int = 4,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate the leading ``k`` singular triplets of ``A``.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
    k : int
        Number of components to return.
    n_oversamples : int
        Extra dimensions sampled beyond ``k``. Larger values tighten the
        approximation; 5-10 is standard.
    n_power_iterations : int
        Power iterations ``(A A^T)^q A Omega``, which amplify spectral decay.
        Use 0 when the spectrum drops off sharply, 4-7 when it is flat.
    random_state : int | np.random.Generator | None
        Seed or generator. Pass a seed for reproducible output.

    Returns
    -------
    U, sigma, Vt : np.ndarray
        Same convention as :func:`eigengrooves.linalg.jacobi_svd.jacobi_svd`.

    Notes
    -----
    This is an *approximation*. Accuracy degrades for slowly-decaying spectra;
    the error bound is in terms of ``sigma_{k+1}``, so it is only as good as
    the matrix is genuinely low-rank. The test suite checks it against the
    deterministic solvers on matrices with known rank structure.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {A.shape}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    m, n = A.shape
    n_samples = min(k + n_oversamples, min(m, n))

    # 1. Sample the range of A.
    Omega = rng.standard_normal((n, n_samples))
    Y = A @ Omega
    Q, _ = householder_qr(Y, reduced=True)

    # 2. Power iterations. Re-orthonormalising between every application is
    #    what keeps this from collapsing onto the dominant singular vector --
    #    without it, rounding wipes out the smaller components we asked for.
    for _ in range(n_power_iterations):
        Z, _ = householder_qr(A.T @ Q, reduced=True)
        Q, _ = householder_qr(A @ Z, reduced=True)

    # 3. Project and decompose the small matrix exactly.
    B = Q.T @ A
    U_b, sigma, Vt = jacobi_svd(B)

    U = Q @ U_b
    return U[:, :k], sigma[:k], Vt[:k, :]
