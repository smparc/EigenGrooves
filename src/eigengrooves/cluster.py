"""
Clustering the latent space, and comparing the result to genre.

Why this module exists
----------------------
The project's stated research question is whether classifying music by audio
features reproduces the genre taxonomy or replaces it with something else.
Recommendation quality does not answer that. Clustering does: partition the
catalogue in latent space, then measure how much the partition agrees with the
genre labels.

The measurements are deliberately chosen to be label-permutation invariant.
Cluster 3 has no intrinsic relationship to "jazz", so any metric that depends
on which integer got assigned to which group is meaningless here. Adjusted
Rand Index and Normalised Mutual Information both compare *partitions*, not
labels, and both correct for the agreement you would get by chance.

Reading the numbers
-------------------
- ARI / NMI near 0: the latent partition is unrelated to genre -- audio
  features have found some other organising principle.
- Near 1: the latent partition reconstructs genre -- the taxonomy is
  recoverable from sound alone.
- In between (the realistic case): partial agreement, and the interesting
  work is in *which* genres merge and which split.

Everything here is implemented from scratch, consistent with the rest of the
project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ClusterResult",
    "GenreAgreement",
    "adjusted_rand_index",
    "compare_to_labels",
    "confusion_table",
    "kmeans",
    "normalized_mutual_information",
    "purity",
    "silhouette_score",
]


@dataclass(frozen=True)
class ClusterResult:
    """A fitted k-means partition."""

    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    n_iter: int
    converged: bool

    @property
    def k(self) -> int:
        return int(self.centroids.shape[0])


def _kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding.

    Picks the first centre uniformly, then each subsequent centre with
    probability proportional to its squared distance from the nearest existing
    centre. This is what stops Lloyd's algorithm from converging to a bad local
    optimum, and it comes with an O(log k) approximation guarantee that uniform
    seeding does not have.
    """
    n = X.shape[0]
    centres = np.empty((k, X.shape[1]))
    centres[0] = X[rng.integers(n)]

    closest_sq = np.sum((X - centres[0]) ** 2, axis=1)
    for i in range(1, k):
        total = closest_sq.sum()
        if total <= 0:
            # Every remaining point coincides with a centre; fill arbitrarily.
            centres[i] = X[rng.integers(n)]
        else:
            centres[i] = X[rng.choice(n, p=closest_sq / total)]
        closest_sq = np.minimum(closest_sq, np.sum((X - centres[i]) ** 2, axis=1))
    return centres


def _assign(X: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, float]:
    """Assign each point to its nearest centroid; return labels and inertia.

    Uses the expansion ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 and drops the
    ||x||^2 term, which is constant per point and cannot change the argmin.
    """
    cross = X @ centroids.T
    centroid_sq = np.einsum("ij,ij->i", centroids, centroids)
    distances = centroid_sq[None, :] - 2.0 * cross
    labels = np.argmin(distances, axis=1)

    point_sq = np.einsum("ij,ij->i", X, X)
    inertia = float(np.sum(point_sq + distances[np.arange(X.shape[0]), labels]))
    return labels, max(inertia, 0.0)


def kmeans(
    X: np.ndarray,
    k: int,
    n_restarts: int = 10,
    max_iter: int = 300,
    tol: float = 1e-6,
    random_state: int | np.random.Generator | None = 0,
) -> ClusterResult:
    """Partition ``X`` into ``k`` clusters by Lloyd's algorithm with k-means++.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    k : int
    n_restarts : int
        Independent runs; the lowest-inertia partition wins. Lloyd's algorithm
        only finds a local optimum, so a single run is a coin flip.
    max_iter : int
    tol : float
        Relative inertia improvement below which the run is considered
        converged.
    random_state : int | np.random.Generator | None

    Returns
    -------
    ClusterResult
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {X.shape}")
    n = X.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"k must be in 1..{n}, got {k}")
    if n_restarts < 1:
        raise ValueError(f"n_restarts must be >= 1, got {n_restarts}")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    best: ClusterResult | None = None
    for _ in range(n_restarts):
        centroids = _kmeans_plus_plus(X, k, rng)
        previous = np.inf
        converged = False
        iteration = 0

        for iteration in range(1, max_iter + 1):
            labels, inertia = _assign(X, centroids)

            for j in range(k):
                members = X[labels == j]
                if members.size:
                    centroids[j] = members.mean(axis=0)
                else:
                    # Empty cluster: reseed onto the point furthest from its
                    # own centroid, which is where an extra cluster helps most.
                    distances = np.sum((X - centroids[labels]) ** 2, axis=1)
                    centroids[j] = X[int(np.argmax(distances))]

            if previous - inertia <= tol * max(previous, 1.0):
                converged = True
                break
            previous = inertia

        labels, inertia = _assign(X, centroids)
        if best is None or inertia < best.inertia:
            best = ClusterResult(
                labels=labels,
                centroids=centroids.copy(),
                inertia=inertia,
                n_iter=iteration,
                converged=converged,
            )

    assert best is not None
    return best


def silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 2000,
    random_state: int | None = 0,
) -> float:
    """Mean silhouette coefficient, in [-1, 1].

    For each point, ``s = (b - a) / max(a, b)`` where ``a`` is its mean
    distance to its own cluster and ``b`` the mean distance to the nearest
    other cluster. Near 1 means tight, well-separated clusters; near 0 means
    the clusters overlap; negative means points are closer to a neighbouring
    cluster than their own.

    The computation is O(n^2) in memory, so it subsamples above
    ``max_samples``. Subsampling changes the estimate slightly but not its
    interpretation.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels).ravel()
    if X.shape[0] != labels.size:
        raise ValueError(f"{X.shape[0]} samples but {labels.size} labels")

    unique = np.unique(labels)
    if unique.size < 2:
        return 0.0

    if X.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        keep = rng.choice(X.shape[0], size=max_samples, replace=False)
        X, labels = X[keep], labels[keep]
        unique = np.unique(labels)
        if unique.size < 2:
            return 0.0

    # Pairwise Euclidean distances via the squared-norm expansion.
    sq = np.einsum("ij,ij->i", X, X)
    distances = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0))

    scores = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        own = labels == labels[i]
        own_count = own.sum() - 1
        if own_count <= 0:
            scores[i] = 0.0
            continue
        a = distances[i, own].sum() / own_count
        b = min(
            distances[i, labels == other].mean()
            for other in unique
            if other != labels[i]
        )
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(scores.mean())


def _contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Contingency table of two labellings."""
    a_values, a_idx = np.unique(a, return_inverse=True)
    b_values, b_idx = np.unique(b, return_inverse=True)
    table = np.zeros((a_values.size, b_values.size), dtype=np.int64)
    np.add.at(table, (a_idx, b_idx), 1)
    return table


def adjusted_rand_index(labels_a, labels_b) -> float:
    """Rand index corrected for chance, in [-1, 1] (0 = random agreement).

    Counts pairs of points that two partitions agree about, then subtracts the
    agreement expected from partitions of the same shape drawn at random. The
    correction is what makes the number comparable across different numbers of
    clusters -- the raw Rand index climbs toward 1 as ``k`` grows regardless of
    whether anything real is being captured.
    """
    a = np.asarray(labels_a).ravel()
    b = np.asarray(labels_b).ravel()
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")
    if a.size == 0:
        return 0.0

    table = _contingency(a, b)
    n = a.size

    def choose2(x):
        return x * (x - 1) / 2.0

    sum_ij = choose2(table.astype(float)).sum()
    sum_i = choose2(table.sum(axis=1).astype(float)).sum()
    sum_j = choose2(table.sum(axis=0).astype(float)).sum()
    total = choose2(float(n))

    expected = sum_i * sum_j / total if total > 0 else 0.0
    maximum = 0.5 * (sum_i + sum_j)
    if maximum == expected:
        return 0.0
    return float((sum_ij - expected) / (maximum - expected))


def normalized_mutual_information(labels_a, labels_b) -> float:
    """Mutual information normalised by the mean entropy, in [0, 1].

    Answers "how much does knowing the cluster tell you about the genre",
    scaled so that 1 means perfect correspondence and 0 means independence.
    """
    a = np.asarray(labels_a).ravel()
    b = np.asarray(labels_b).ravel()
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")
    if a.size == 0:
        return 0.0

    table = _contingency(a, b).astype(float)
    n = table.sum()
    if n == 0:
        return 0.0

    joint = table / n
    marginal_a = joint.sum(axis=1)
    marginal_b = joint.sum(axis=0)

    nonzero = joint > 0
    outer = np.outer(marginal_a, marginal_b)
    mutual = float(np.sum(joint[nonzero] * np.log(joint[nonzero] / outer[nonzero])))

    def entropy(p):
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    denominator = 0.5 * (entropy(marginal_a) + entropy(marginal_b))
    if denominator <= 0:
        return 0.0
    return float(np.clip(mutual / denominator, 0.0, 1.0))


def purity(cluster_labels, class_labels) -> float:
    """Fraction of points in the majority class of their own cluster.

    Intuitive but *not* chance-corrected: it rises toward 1 as the number of
    clusters approaches the number of points. Report it beside ARI and NMI,
    never instead of them.
    """
    table = _contingency(np.asarray(cluster_labels).ravel(), np.asarray(class_labels).ravel())
    n = table.sum()
    if n == 0:
        return 0.0
    return float(table.max(axis=1).sum() / n)


@dataclass(frozen=True)
class GenreAgreement:
    """How closely a latent partition reproduces a reference taxonomy."""

    k: int
    adjusted_rand_index: float
    normalized_mutual_information: float
    purity: float
    silhouette: float
    n_reference_classes: int

    def verdict(self) -> str:
        """A one-line reading of the agreement, for prose and captions."""
        ari = self.adjusted_rand_index
        if ari < 0.05:
            return "essentially unrelated to the reference taxonomy"
        if ari < 0.25:
            return "weakly related to the reference taxonomy"
        if ari < 0.55:
            return "partially reproduces the reference taxonomy"
        return "closely reproduces the reference taxonomy"


def compare_to_labels(
    latent: np.ndarray,
    reference_labels,
    k: int | None = None,
    n_restarts: int = 10,
    random_state: int | None = 0,
) -> tuple[ClusterResult, GenreAgreement]:
    """Cluster the latent space and measure agreement with reference labels.

    Parameters
    ----------
    latent : np.ndarray, shape (n_songs, k_latent)
    reference_labels : sequence
        Ground-truth labels, typically genre.
    k : int, optional
        Number of clusters. Defaults to the number of distinct reference
        classes, which is the fairest comparison -- giving the clustering more
        clusters than the taxonomy has classes inflates purity for free.
    n_restarts : int
    random_state : int, optional

    Returns
    -------
    (ClusterResult, GenreAgreement)
    """
    latent = np.asarray(latent, dtype=float)
    reference = np.asarray(list(reference_labels), dtype=object).ravel()
    if latent.shape[0] != reference.size:
        raise ValueError(f"{latent.shape[0]} songs but {reference.size} labels")

    n_classes = int(np.unique(reference).size)
    if k is None:
        k = n_classes
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    result = kmeans(latent, k, n_restarts=n_restarts, random_state=random_state)

    agreement = GenreAgreement(
        k=k,
        adjusted_rand_index=adjusted_rand_index(result.labels, reference),
        normalized_mutual_information=normalized_mutual_information(result.labels, reference),
        purity=purity(result.labels, reference),
        silhouette=silhouette_score(latent, result.labels, random_state=random_state),
        n_reference_classes=n_classes,
    )
    return result, agreement


def confusion_table(cluster_labels, class_labels) -> tuple[np.ndarray, list, list]:
    """Cluster-by-class counts, plus the row and column keys.

    This is where the interesting detail lives: which genres a cluster merges,
    and which genres get split across clusters. The scalar agreement metrics
    summarise it; this shows the structure.
    """
    clusters = np.asarray(cluster_labels).ravel()
    classes = np.asarray(list(class_labels), dtype=object).ravel()
    cluster_keys = sorted(np.unique(clusters).tolist())
    class_keys = sorted(np.unique(classes).tolist())
    return _contingency(clusters, classes), cluster_keys, class_keys
