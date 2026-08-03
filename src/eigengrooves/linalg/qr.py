"""
QR decomposition, implemented from scratch.

Two algorithms are provided:

``householder_qr``
    Backward-stable QR via Householder reflections. This is the default and the
    one you should use for anything numerical. Orthogonality of ``Q`` is
    guaranteed to machine precision regardless of the conditioning of ``A``.

``modified_gram_schmidt``
    QR via modified Gram-Schmidt. Cheaper and easier to read, but loses
    orthogonality proportionally to ``cond(A)``. Kept because it is the
    textbook algorithm and because the test-suite uses it to *demonstrate* the
    stability gap against Householder.

Classical Gram-Schmidt is deliberately not offered: it loses orthogonality
proportionally to ``cond(A)^2`` and has no advantage over the modified variant.
"""

from __future__ import annotations

import numpy as np

__all__ = ["householder_qr", "modified_gram_schmidt", "qr"]


def householder_qr(A: np.ndarray, reduced: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``A = Q R`` using Householder reflections.

    Each step zeroes the sub-diagonal of one column by reflecting it onto a
    multiple of ``e_1``::

        v = x + sign(x_0) * ||x|| * e_1
        H = I - 2 v v^T / (v^T v)

    The reflectors are applied to the working matrix in place and accumulated
    into ``Q`` at the end, which avoids ever materialising an ``m x m``
    reflector.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
    reduced : bool
        If True return the thin factorisation ``Q`` (m, min(m, n)) and ``R``
        (min(m, n), n). If False return the full ``Q`` (m, m) and ``R`` (m, n).

    Returns
    -------
    Q, R : np.ndarray
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {A.shape}")

    m, n = A.shape
    R = A.copy()
    # Store the reflector vectors; applying them in reverse builds Q.
    reflectors: list[tuple[int, np.ndarray]] = []

    for j in range(min(m - 1, n)):
        x = R[j:, j]
        norm_x = np.linalg.norm(x)
        if norm_x < np.finfo(float).tiny:
            continue

        # Choosing the sign this way avoids cancellation when x[0] ~ -||x||.
        alpha = -norm_x if x[0] >= 0 else norm_x
        v = x.copy()
        v[0] -= alpha
        v_norm = np.linalg.norm(v)
        if v_norm < np.finfo(float).tiny:
            continue
        v /= v_norm

        # R[j:, j:] <- (I - 2 v v^T) R[j:, j:]
        R[j:, j:] -= 2.0 * np.outer(v, v @ R[j:, j:])
        reflectors.append((j, v))

    # Build Q by applying the reflectors in reverse order to the identity.
    q_cols = m if not reduced else min(m, n)
    Q = np.eye(m, q_cols)
    for j, v in reversed(reflectors):
        Q[j:, :] -= 2.0 * np.outer(v, v @ Q[j:, :])

    if reduced:
        R = R[: min(m, n), :]

    # Force exact zeros below the diagonal; the reflectors leave rounding dust
    # there and callers reasonably expect R to be upper triangular.
    R = np.triu(R)
    return Q, R


def modified_gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute a thin ``A = Q R`` using modified Gram-Schmidt.

    Differs from the classical algorithm by projecting each remaining column
    against ``q_j`` immediately after it is produced, rather than projecting
    the original column against every previous ``q_i``. Mathematically
    identical, numerically much better behaved.

    Rank-deficient columns yield a zero column in ``Q`` and a zero row in ``R``.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {A.shape}")

    m, n = A.shape
    V = A.copy()
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    # Threshold relative to the matrix scale, so this behaves the same whether
    # the data is measured in dB or in kilobels.
    scale = np.linalg.norm(A)
    tol = max(m, n) * np.finfo(float).eps * (scale if scale > 0 else 1.0)

    for j in range(n):
        R[j, j] = np.linalg.norm(V[:, j])
        if R[j, j] <= tol:
            R[j, j] = 0.0
            continue
        Q[:, j] = V[:, j] / R[j, j]
        if j + 1 < n:
            R[j, j + 1 :] = Q[:, j] @ V[:, j + 1 :]
            V[:, j + 1 :] -= np.outer(Q[:, j], R[j, j + 1 :])

    return Q, R


def qr(A: np.ndarray, method: str = "householder") -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to a QR implementation by name."""
    if method == "householder":
        return householder_qr(A, reduced=True)
    if method in ("mgs", "gram-schmidt", "modified_gram_schmidt"):
        return modified_gram_schmidt(A)
    raise ValueError(f"unknown QR method {method!r}; expected 'householder' or 'mgs'")
