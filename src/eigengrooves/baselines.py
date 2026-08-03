"""
Baselines the SVD recommender has to beat.

The original README asserted that the system "surfaced cross-genre matches
unified by shared latent sonic characteristics." That may well be true, but
nothing in the repository could distinguish it from the system surfacing
whatever happened to be nearby in a nine-dimensional space, or simply
surfacing popular songs.

These are the reference points that make the claim falsifiable:

``RandomRanker``
    The floor. Any metric a real system cannot beat here is measuring noise.

``PopularityRanker``
    Ignores the query entirely and returns the most popular tracks. Deceptively
    strong on accuracy metrics, which is exactly why it belongs here -- if the
    latent model cannot beat it, the latent model is not doing anything.

``RawFeatureRanker``
    Cosine similarity on the scaled features with no decomposition at all.
    **This is the one that matters.** SVD's entire justification is that
    projecting to a latent subspace beats using the features directly. That is
    a hypothesis, and this baseline tests it.

``LatentRanker``
    Wraps a fitted :class:`~eigengrooves.model.LatentModel`, so different ``k``,
    whitening and aggregation settings compete under identical conditions.

Every ranker exposes the same ``rank(seeds, n, exclude)`` interface, so the
evaluation harness cannot accidentally give one an advantage.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .catalog import Catalog
from .model import LatentModel
from .similarity import normalize_rows, top_k_indices

__all__ = [
    "Ranker",
    "RandomRanker",
    "PopularityRanker",
    "RawFeatureRanker",
    "LatentRanker",
    "build_standard_rankers",
]


@runtime_checkable
class Ranker(Protocol):
    """Anything that can order the catalogue against a seed playlist."""

    name: str

    def rank(self, seed_indices: list[int], n: int, exclude: set[int]) -> np.ndarray:
        """Return up to ``n`` catalogue indices, best first."""
        ...


class RandomRanker:
    """Uniformly random selection. The floor."""

    def __init__(self, n_items: int, random_state: int | None = 0):
        self.name = "random"
        self.n_items = n_items
        self._rng = np.random.default_rng(random_state)

    def rank(self, seed_indices: list[int], n: int, exclude: set[int]) -> np.ndarray:
        allowed = np.setdiff1d(
            np.arange(self.n_items), np.fromiter(exclude, dtype=int, count=len(exclude))
        )
        if allowed.size == 0:
            return np.empty(0, dtype=int)
        size = min(n, allowed.size)
        return self._rng.choice(allowed, size=size, replace=False)


class PopularityRanker:
    """Most popular first, regardless of the query."""

    def __init__(self, popularity: np.ndarray):
        self.name = "popularity"
        self._scores = np.asarray(popularity, dtype=float)

    def rank(self, seed_indices: list[int], n: int, exclude: set[int]) -> np.ndarray:
        return top_k_indices(self._scores, n, exclude=exclude)


class RawFeatureRanker:
    """Cosine similarity on scaled features, no dimensionality reduction.

    The honest control for the entire project.
    """

    def __init__(self, scaled_features: np.ndarray, aggregation: str = "max"):
        self.name = "raw_cosine"
        self._unit = normalize_rows(scaled_features)
        self.aggregation = aggregation

    def rank(self, seed_indices: list[int], n: int, exclude: set[int]) -> np.ndarray:
        if not seed_indices:
            return np.empty(0, dtype=int)
        per_seed = np.clip(self._unit[list(seed_indices)] @ self._unit.T, -1.0, 1.0)
        scores = per_seed.mean(axis=0) if self.aggregation == "mean" else per_seed.max(axis=0)
        return top_k_indices(scores, n, exclude=exclude)


class LatentRanker:
    """Cosine similarity in a fitted latent space."""

    def __init__(
        self,
        model: LatentModel,
        catalog: Catalog,
        aggregation: str = "max",
        name: str | None = None,
    ):
        self.model = model
        self.aggregation = aggregation
        whiten_tag = "+whiten" if model.whiten else ""
        self.name = name or f"svd_k{model.k}{whiten_tag}_{aggregation}"
        self._unit = normalize_rows(model.transform(catalog.features))

    def rank(self, seed_indices: list[int], n: int, exclude: set[int]) -> np.ndarray:
        if not seed_indices:
            return np.empty(0, dtype=int)
        per_seed = np.clip(self._unit[list(seed_indices)] @ self._unit.T, -1.0, 1.0)
        if self.aggregation == "mean":
            scores = per_seed.mean(axis=0)
        elif self.aggregation == "borda":
            n_items = per_seed.shape[1]
            scores = np.zeros(n_items)
            for row in per_seed:
                scores[np.argsort(-row, kind="stable")] += np.arange(n_items, 0, -1)
        else:
            scores = per_seed.max(axis=0)
        return top_k_indices(scores, n, exclude=exclude)


def build_standard_rankers(
    catalog: Catalog,
    scaled_features: np.ndarray,
    models: dict[str, LatentModel],
    random_state: int | None = 0,
) -> list[Ranker]:
    """Assemble the standard comparison set.

    Parameters
    ----------
    catalog : Catalog
    scaled_features : np.ndarray
        Scaled features shared by every system, so the comparison is fair.
    models : dict[str, LatentModel]
        Named latent models to include.
    random_state : int, optional

    Returns
    -------
    list[Ranker]
    """
    rankers: list[Ranker] = [
        RandomRanker(len(catalog), random_state=random_state),
        PopularityRanker(catalog.popularity()),
        RawFeatureRanker(scaled_features),
    ]
    rankers.extend(
        LatentRanker(model, catalog, name=name) for name, model in models.items()
    )
    return rankers
