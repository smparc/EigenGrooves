"""
Symmetric eigensolvers, implemented from scratch.

Two algorithms, both operating on real symmetric matrices:

``jacobi_eigh``
    Cyclic two-sided Jacobi. Unconditionally convergent, and accurate to high
    *relative* precision for positive-definite matrices -- meaning even the
    tiniest eigenvalues come back with full significant digits. Slower than QR
    in the asymptotic sense, irrelevant at the sizes we use.

``qr_eigh``
    Symmetric QR iteration with Wilkinson shifts and deflation, built on the
    Householder QR in :mod:`eigengrooves.linalg.qr`.

Both replace the original project's *unshifted* QR iteration, which had two
defects worth recording since they motivated this module:

1. Its convergence test compared an absolute off-diagonal sum against 1e-10.
   Since that sum scales with ``||A||``, the test never fired on real data --
   at 5000 rows the loop burned all 1000 iterations and still exited with an
   off-diagonal mass of ~8e-3. Both solvers here use a *relative* tolerance.
2. Without shifts, convergence is linear in ``|lambda_{i+1} / lambda_i|``, which
   stalls badly on clustered spectra. Jacobi sidesteps this entirely; the QR
   path uses Wilkinson shifts, which give cubic convergence for symmetric input.
"""

from __future__ import annotations

import numpy as np

from .qr import householder_qr

__all__ = ["jacobi_eigh", "qr_eigh", "symmetric_eigh"]


def _sort_eigenpairs(values: np.ndarray, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort eigenpairs by descending eigenvalue, stably."""
    # Negate rather than reverse a sort: `[::-1]` on argsort breaks ties in a
    # confusing order, and stability matters when eigenvalues repeat.
    order = np.argsort(-values, kind="stable")
    return values[order], vectors[:, order]


def _check_symmetric(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {S.shape}")
    asym = np.linalg.norm(S - S.T)
    scale = np.linalg.norm(S)
    if scale > 0 and asym > 1e-8 * scale:
        raise ValueError(
            f"matrix is not symmetric (||S - S^T|| / ||S|| = {asym / scale:.2e})"
        )
    # Symmetrise to kill any rounding asymmetry before we start.
    return 0.5 * (S + S.T)


def jacobi_eigh(
    S: np.ndarray, max_sweeps: int = 60, tol: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose a real symmetric matrix by cyclic Jacobi rotations.

    Repeatedly applies plane rotations ``J(p, q, theta)`` chosen to annihilate
    the off-diagonal entry ``S[p, q]``. Each rotation is orthogonal, so
    ``S`` stays symmetric and its eigenvalues are preserved exactly; the
    off-diagonal Frobenius mass decreases monotonically to zero.

    Parameters
    ----------
    S : np.ndarray, shape (n, n), symmetric
    max_sweeps : int
        A "sweep" visits every off-diagonal pair once. Convergence is typically
        reached in 6-10 sweeps; the cap is a safety net, not a target.
    tol : float, optional
        Relative tolerance on off-diagonal mass. Defaults to ``n * eps``.

    Returns
    -------
    eigenvalues : np.ndarray, shape (n,), descending
    eigenvectors : np.ndarray, shape (n, n), orthonormal columns
    """
    A = _check_symmetric(S)
    n = A.shape[0]
    V = np.eye(n)

    if n == 1:
        return A[0].copy(), V

    if tol is None:
        tol = n * np.finfo(float).eps

    frob = np.linalg.norm(A)
    if frob == 0.0:
        return np.zeros(n), V

    for _ in range(max_sweeps):
        off = np.sqrt(np.sum(A**2) - np.sum(np.diag(A) ** 2))
        if off <= tol * frob:
            break

        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p, q]
                if abs(apq) <= tol * frob / n:
                    continue

                # Solve for the rotation that zeroes A[p, q]. Using the
                # smaller root of t^2 + 2*theta*t - 1 = 0 keeps |t| <= 1,
                # which is what makes the iteration numerically docile.
                theta = (A[q, q] - A[p, p]) / (2.0 * apq)
                if theta >= 0:
                    t = 1.0 / (theta + np.sqrt(1.0 + theta * theta))
                else:
                    t = -1.0 / (-theta + np.sqrt(1.0 + theta * theta))
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = t * c

                # Apply J^T A J, touching only rows/cols p and q.
                col_p = A[:, p].copy()
                col_q = A[:, q].copy()
                A[:, p] = c * col_p - s * col_q
                A[:, q] = s * col_p + c * col_q
                row_p = A[p, :].copy()
                row_q = A[q, :].copy()
                A[p, :] = c * row_p - s * row_q
                A[q, :] = s * row_p + c * row_q
                # The annihilated entries are zero by construction; assigning
                # them explicitly stops rounding dust from accumulating.
                A[p, q] = A[q, p] = 0.0

                # Accumulate the rotation into the eigenvector basis.
                v_p = V[:, p].copy()
                v_q = V[:, q].copy()
                V[:, p] = c * v_p - s * v_q
                V[:, q] = s * v_p + c * v_q

    return _sort_eigenpairs(np.diag(A).copy(), V)


def _wilkinson_shift(A: np.ndarray, m: int) -> float:
    """Wilkinson shift from the trailing 2x2 block of ``A[:m, :m]``.

    Picks the eigenvalue of that block closest to ``A[m-1, m-1]``, which is
    what buys cubic convergence on symmetric matrices.
    """
    if m < 2:
        return float(A[0, 0])
    a = A[m - 2, m - 2]
    b = A[m - 2, m - 1]
    c = A[m - 1, m - 1]
    delta = (a - c) / 2.0
    denom = abs(delta) + np.sqrt(delta * delta + b * b)
    if denom == 0.0:
        return float(c)
    sign = 1.0 if delta >= 0 else -1.0
    return float(c - (sign * b * b) / denom)


def qr_eigh(
    S: np.ndarray, max_iter: int = 500, tol: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose a real symmetric matrix by shifted QR iteration.

    At each step we factor ``A - mu I = Q R`` and form ``A <- R Q + mu I``,
    which is an orthogonal similarity transform (so the spectrum is preserved)
    that drives the matrix toward diagonal form. The Wilkinson shift ``mu``
    accelerates the trailing eigenvalue; once its off-diagonal row is
    negligible we *deflate* -- freeze that eigenvalue and continue on the
    leading submatrix.

    Returns
    -------
    eigenvalues : np.ndarray, shape (n,), descending
    eigenvectors : np.ndarray, shape (n, n), orthonormal columns
    """
    A = _check_symmetric(S)
    n = A.shape[0]
    V = np.eye(n)

    if n == 1:
        return A[0].copy(), V

    if tol is None:
        tol = n * np.finfo(float).eps

    frob = np.linalg.norm(A)
    if frob == 0.0:
        return np.zeros(n), V

    m = n  # active submatrix is A[:m, :m]
    iterations = 0
    while m > 1 and iterations < max_iter:
        # Deflate as far as possible before doing any work.
        while m > 1 and abs(A[m - 1, m - 2]) <= tol * (
            abs(A[m - 1, m - 1]) + abs(A[m - 2, m - 2]) + frob / n
        ):
            A[m - 1, m - 2] = A[m - 2, m - 1] = 0.0
            m -= 1
        if m <= 1:
            break

        mu = _wilkinson_shift(A, m)
        sub = A[:m, :m] - mu * np.eye(m)
        Q, R = householder_qr(sub, reduced=True)
        A[:m, :m] = R @ Q + mu * np.eye(m)
        # Re-symmetrise: R @ Q is symmetric in exact arithmetic only.
        A[:m, :m] = 0.5 * (A[:m, :m] + A[:m, :m].T)
        V[:, :m] = V[:, :m] @ Q
        iterations += 1

    return _sort_eigenpairs(np.diag(A).copy(), V)


def symmetric_eigh(S: np.ndarray, method: str = "jacobi") -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to a symmetric eigensolver by name."""
    if method == "jacobi":
        return jacobi_eigh(S)
    if method == "qr":
        return qr_eigh(S)
    raise ValueError(f"unknown eigensolver {method!r}; expected 'jacobi' or 'qr'")
