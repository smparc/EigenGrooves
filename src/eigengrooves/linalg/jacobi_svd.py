"""
One-sided Jacobi SVD, implemented from scratch.

Why this algorithm
------------------
The obvious way to get an SVD is to eigendecompose ``A^T A``: its eigenvectors
are the right singular vectors and its eigenvalues are the squared singular
values. That is what the first version of this project did, and it works fine
until the feature matrix is ill-conditioned -- which audio-feature matrices are,
because energy/loudness/acousticness are strongly correlated.

The problem is that forming ``A^T A`` squares the condition number. Singular
values below ``sqrt(eps) * sigma_max`` are annihilated by rounding before the
eigensolver ever sees them. Measured on correlated synthetic data with the old
implementation: reconstruction error degraded from 1e-15 to 2.8e-6 and three of
nine components were silently dropped.

One-sided Jacobi never forms ``A^T A``. It orthogonalises the *columns of A*
directly using plane rotations, which gives singular values accurate to high
relative precision -- small ones included. This is the algorithm behind LAPACK's
``dgesvj``, and it is the most accurate SVD method known for dense matrices.

How it works
------------
Apply rotations ``A <- A J(p, q, theta)`` chosen so that columns ``p`` and ``q``
become orthogonal. Rotations are orthogonal, so ``A J`` has the same singular
values as ``A``. Repeat until every pair of columns is mutually orthogonal; at
that point ``A = U Sigma`` with ``U`` orthonormal, so::

    sigma_i = ||a_i||        U[:, i] = a_i / sigma_i        V = product of J's

The rotation for a pair is derived from the 2x2 Gram block
``[[alpha, gamma], [gamma, beta]]`` where ``alpha = a_p . a_p``,
``beta = a_q . a_q``, ``gamma = a_p . a_q`` -- exactly the Jacobi eigenvalue
rotation, but applied one-sidedly.
"""

from __future__ import annotations

import numpy as np

__all__ = ["jacobi_svd"]


def jacobi_svd(
    A: np.ndarray,
    k: int | None = None,
    max_sweeps: int = 60,
    tol: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the SVD ``A = U diag(sigma) V^T`` by one-sided Jacobi rotations.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
    k : int, optional
        Truncate to the leading ``k`` components. ``None`` keeps every
        numerically non-negligible one.
    max_sweeps : int
        A sweep visits every column pair once. Convergence is typically 6-12
        sweeps; the cap is a safety net.
    tol : float, optional
        Orthogonality tolerance on ``|a_p . a_q| / (||a_p|| ||a_q||)``.
        Defaults to ``n * eps``.

    Returns
    -------
    U : np.ndarray, shape (m, r)
    sigma : np.ndarray, shape (r,), descending, non-negative
    Vt : np.ndarray, shape (r, n) -- rows are right singular vectors
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {A.shape}")

    m, n = A.shape
    if m == 0 or n == 0:
        return np.zeros((m, 0)), np.zeros(0), np.zeros((0, n))

    # One-sided Jacobi requires at least as many rows as columns (otherwise the
    # columns cannot all be made mutually orthogonal). Transpose if needed and
    # swap the roles of U and V at the end.
    transposed = m < n
    if transposed:
        A = A.T
        m, n = n, m

    if tol is None:
        tol = n * np.finfo(float).eps

    W = A.copy()          # becomes U * Sigma
    V = np.eye(n)         # accumulates the rotations

    scale = np.linalg.norm(W)
    if scale == 0.0:
        empty_u = np.zeros((m, 0))
        empty_s = np.zeros(0)
        empty_vt = np.zeros((0, n))
        if transposed:
            return np.zeros((n, 0)), empty_s, np.zeros((0, m))
        return empty_u, empty_s, empty_vt

    for _ in range(max_sweeps):
        converged = True
        for p in range(n - 1):
            for q in range(p + 1, n):
                a_p = W[:, p]
                a_q = W[:, q]
                alpha = float(a_p @ a_p)
                beta = float(a_q @ a_q)
                gamma = float(a_p @ a_q)

                if alpha == 0.0 or beta == 0.0:
                    continue
                # Relative orthogonality measure: the cosine of the angle
                # between the two columns.
                if abs(gamma) <= tol * np.sqrt(alpha * beta):
                    continue
                converged = False

                # Smaller root of t^2 + 2*zeta*t - 1 = 0, computed in the
                # cancellation-free form.
                zeta = (beta - alpha) / (2.0 * gamma)
                if zeta >= 0:
                    t = 1.0 / (zeta + np.sqrt(1.0 + zeta * zeta))
                else:
                    t = -1.0 / (-zeta + np.sqrt(1.0 + zeta * zeta))
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = c * t

                col_p = W[:, p].copy()
                col_q = W[:, q].copy()
                W[:, p] = c * col_p - s * col_q
                W[:, q] = s * col_p + c * col_q

                v_p = V[:, p].copy()
                v_q = V[:, q].copy()
                V[:, p] = c * v_p - s * v_q
                V[:, q] = s * v_p + c * v_q

        if converged:
            break

    # Columns of W are now mutually orthogonal: their norms are the singular
    # values and their directions are the left singular vectors.
    sigma = np.linalg.norm(W, axis=0)
    order = np.argsort(-sigma, kind="stable")
    sigma = sigma[order]
    W = W[:, order]
    V = V[:, order]

    # Drop numerically-zero components. The threshold is relative to sigma_max,
    # which is the only scale-invariant choice.
    cutoff = max(m, n) * np.finfo(float).eps * (sigma[0] if sigma.size else 0.0)
    rank = int(np.sum(sigma > cutoff))
    if k is not None:
        rank = min(rank, max(int(k), 0))

    sigma = sigma[:rank]
    U = W[:, :rank] / np.where(sigma == 0.0, 1.0, sigma)
    V = V[:, :rank]

    if transposed:
        # A_original = A^T = (U S V^T)^T = V S U^T
        return V, sigma, U.T
    return U, sigma, V.T
