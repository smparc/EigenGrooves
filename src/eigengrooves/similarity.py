"""
Similarity in latent space.

The original implementation computed one query's similarities with a Python
list comprehension calling ``cosine_similarity`` once per song. Measured at
10,000 songs that is 23 ms per query against 0.31 ms for the vectorised form --
74x, and it grows with the catalogue. Everything here is expressed as matrix
products.

Normalising the catalogue once up front (``normalize_rows``) and reusing it
turns every subsequent query into a single mat-vec, which is what the
recommender does.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cosine_similarity",
    "cosine_similarity_matrix",
    "normalize_rows",
    "query_similarities",
    "top_k_indices",
]

_TINY = 1e-12


def normalize_rows(X: np.ndarray) -> np.ndarray:
    """Scale every row to unit L2 norm; zero rows are left as zero.

    Once rows are unit-norm, cosine similarity is just a dot product, which is
    the whole trick behind the vectorised paths below.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {X.shape}")
    norms = np.sqrt(np.einsum("ij,ij->i", X, X))[:, None]
    return X / np.where(norms < _TINY, 1.0, norms)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, in [-1, 1].

    Returns 0.0 when either vector is degenerate -- an undefined angle is
    better reported as "no evidence of similarity" than as NaN propagating
    silently into a ranking.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"dimension mismatch: {a.size} vs {b.size}")
    na = np.sqrt(a @ a)
    nb = np.sqrt(b @ b)
    if na < _TINY or nb < _TINY:
        return 0.0
    return float(np.clip((a @ b) / (na * nb), -1.0, 1.0))


def cosine_similarity_matrix(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """All-pairs cosine similarity between rows of ``X`` (and ``Y``).

    Warning: the output is ``len(X) x len(Y)``. At 50,000 songs the self-
    similarity matrix is 20 GB. Use :func:`query_similarities` for the one-
    query-against-catalogue case, which is what recommendation actually needs.
    """
    Xn = normalize_rows(X)
    Yn = Xn if Y is None else normalize_rows(Y)
    return np.clip(Xn @ Yn.T, -1.0, 1.0)


def query_similarities(query: np.ndarray, catalog: np.ndarray) -> np.ndarray:
    """Cosine similarity of one or many queries against a catalogue.

    Parameters
    ----------
    query : np.ndarray, shape (k,) or (n_queries, k)
    catalog : np.ndarray, shape (n_songs, k)

    Returns
    -------
    np.ndarray, shape (n_songs,) for a single query, else (n_queries, n_songs)
    """
    query = np.asarray(query, dtype=float)
    catalog = np.asarray(catalog, dtype=float)
    single = query.ndim == 1
    Q = query[None, :] if single else query
    if Q.shape[1] != catalog.shape[1]:
        raise ValueError(
            f"dimension mismatch: query has {Q.shape[1]}, catalog has {catalog.shape[1]}"
        )
    scores = np.clip(normalize_rows(Q) @ normalize_rows(catalog).T, -1.0, 1.0)
    return scores[0] if single else scores


def top_k_indices(scores: np.ndarray, k: int, exclude: set[int] | None = None) -> np.ndarray:
    """Indices of the ``k`` highest scores, descending, honouring exclusions.

    Uses ``argpartition`` for the selection (O(n)) and sorts only the surviving
    ``k``, rather than sorting the whole catalogue to look at its head.
    """
    scores = np.asarray(scores, dtype=float).copy()
    if exclude:
        idx = np.fromiter(exclude, dtype=int, count=len(exclude))
        idx = idx[(idx >= 0) & (idx < scores.size)]
        scores[idx] = -np.inf

    valid = int(np.sum(np.isfinite(scores)))
    k = int(min(max(k, 0), valid))
    if k == 0:
        return np.empty(0, dtype=int)

    # argpartition puts the k largest anywhere in the first k slots; we then
    # sort just those.
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]
