"""
Ranking metrics, implemented from scratch.

Accuracy metrics answer "did we retrieve the right things". Beyond-accuracy
metrics answer "is the result list any good as a *list*" -- a recommender that
returns the ten most popular songs every time can score respectably on recall
while being completely useless, and only diversity, coverage and novelty
expose that.

All accuracy metrics take a ranked list of candidate indices and a set of
relevant ones, and treat relevance as binary.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "average_precision_at_k",
    "catalog_coverage",
    "hit_rate_at_k",
    "intra_list_diversity",
    "ndcg_at_k",
    "novelty",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "shannon_entropy",
]


def _prefix(ranked: np.ndarray | list[int], k: int) -> list[int]:
    return [int(i) for i in list(ranked)[: max(int(k), 0)]]


def hit_rate_at_k(ranked, relevant: set[int], k: int) -> float:
    """1.0 if any relevant item appears in the top ``k``, else 0.0."""
    if not relevant:
        return 0.0
    return float(any(i in relevant for i in _prefix(ranked, k)))


def precision_at_k(ranked, relevant: set[int], k: int) -> float:
    """Fraction of the top ``k`` that is relevant."""
    if k <= 0:
        return 0.0
    top = _prefix(ranked, k)
    if not top:
        return 0.0
    return sum(1 for i in top if i in relevant) / float(k)


def recall_at_k(ranked, relevant: set[int], k: int) -> float:
    """Fraction of the relevant set retrieved in the top ``k``."""
    if not relevant:
        return 0.0
    top = _prefix(ranked, k)
    return sum(1 for i in top if i in relevant) / float(len(relevant))


def reciprocal_rank(ranked, relevant: set[int]) -> float:
    """1 / rank of the first relevant item; 0 if none is retrieved."""
    if not relevant:
        return 0.0
    for position, index in enumerate(ranked, start=1):
        if int(index) in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked, relevant: set[int], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    Discount is ``1 / log2(rank + 1)``, and the ideal ranking places every
    relevant item first -- so the result is in [0, 1] and rewards putting hits
    near the top rather than merely inside the window.
    """
    if not relevant or k <= 0:
        return 0.0
    top = _prefix(ranked, k)
    gains = np.array([1.0 if i in relevant else 0.0 for i in top])
    if gains.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())

    n_ideal = min(len(relevant), k)
    ideal = float((np.ones(n_ideal) * (1.0 / np.log2(np.arange(2, n_ideal + 2)))).sum())
    return dcg / ideal if ideal > 0 else 0.0


def average_precision_at_k(ranked, relevant: set[int], k: int) -> float:
    """Mean of precision@i taken at each relevant hit within the top ``k``."""
    if not relevant or k <= 0:
        return 0.0
    hits = 0
    total = 0.0
    for position, index in enumerate(_prefix(ranked, k), start=1):
        if int(index) in relevant:
            hits += 1
            total += hits / position
    denominator = min(len(relevant), k)
    return total / denominator if denominator else 0.0


def intra_list_diversity(ranked, embeddings: np.ndarray) -> float:
    """Mean pairwise cosine *distance* within a result list, in [0, 2].

    Computed in a fixed shared embedding (the scaled raw features), not in any
    one model's latent space -- otherwise each system would be graded in the
    geometry most flattering to itself.
    """
    indices = [int(i) for i in ranked]
    if len(indices) < 2:
        return 0.0
    vectors = np.asarray(embeddings, dtype=float)[indices]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms < 1e-12, 1.0, norms)
    similarity = np.clip(unit @ unit.T, -1.0, 1.0)
    upper = similarity[np.triu_indices(len(indices), k=1)]
    return float(1.0 - upper.mean())


def catalog_coverage(all_ranked: list, n_items: int) -> float:
    """Fraction of the catalogue that appears across every query.

    A recommender confined to a small popular core scores near zero here
    regardless of how good its accuracy looks.
    """
    if n_items <= 0:
        return 0.0
    seen: set[int] = set()
    for ranked in all_ranked:
        seen.update(int(i) for i in ranked)
    return len(seen) / float(n_items)


def novelty(ranked, popularity: np.ndarray) -> float:
    """Mean self-information ``-log2(p)`` of the recommended items.

    Popularity is rescaled into (0, 1] as a pseudo-probability of being known.
    Higher means the list leans further into the tail.
    """
    indices = [int(i) for i in ranked]
    if not indices:
        return 0.0
    popularity = np.asarray(popularity, dtype=float)
    span = popularity.max() - popularity.min()
    if span <= 0:
        return 0.0
    scaled = (popularity - popularity.min()) / span
    # Floor keeps the logarithm finite for the least-popular track.
    probability = np.clip(scaled, 1e-3, 1.0)
    return float(np.mean(-np.log2(probability[indices])))


def shannon_entropy(labels: list) -> float:
    """Entropy of a label distribution, in bits.

    Used for artist and genre spread within a result list: a list of ten tracks
    by one artist has entropy 0, ten distinct artists gives log2(10).
    """
    if not labels:
        return 0.0
    _, counts = np.unique(np.asarray(labels, dtype=object), return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())
