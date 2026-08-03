"""
Recommendation strategies over the latent space.

What changed from the original two strategies
---------------------------------------------
``one_per_song`` and ``overall_top`` are both still here, and both still mean
what they meant. Three things are different:

*Exclusion is by track, not by row.* Previously a playlist entry excluded only
the single dataframe row it matched. With a chart dataset carrying one row per
track per week, the same track's other rows stayed eligible and were returned
at similarity 1.000000 -- the system recommended your own song back to you.
Deduplication in :mod:`eigengrooves.catalog` removes the duplicate rows; this
module additionally excludes every seed by index and, optionally, caps how many
tracks a single artist may contribute.

*Aggregation is a choice.* ``overall_top`` hardcoded elementwise ``max`` over
the seeds, which means one outlier seed can own the entire result list. ``max``
is still available and still the default, but ``mean`` (coherence with the
playlist as a whole) and ``borda`` (rank-based, robust to scale differences
between seeds) are now selectable.

*Diversity is principled.* ``one_per_song`` approximates diversity by
construction -- one result per seed. ``mmr`` does it properly, trading relevance
against redundancy with an explicit knob, which is the standard formulation
(Carbonell & Goldstein, 1998).

Beyond that: popularity de-biasing so the output is not simply the chart
re-ordered, Rocchio-style negative feedback, and per-result explanations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .catalog import Catalog
from .explain import Explanation, explain_match
from .model import LatentModel
from .similarity import normalize_rows, top_k_indices

__all__ = [
    "Recommendation",
    "RecommendationResult",
    "Recommender",
    "STRATEGIES",
    "AGGREGATIONS",
]

STRATEGIES = ("overall_top", "one_per_song", "mmr", "centroid")
AGGREGATIONS = ("max", "mean", "borda")


@dataclass(frozen=True)
class Recommendation:
    """One recommended track."""

    index: int
    title: str
    artist: str
    score: float
    seed: str | None = None
    explanation: Explanation | None = None
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        payload = {
            "index": self.index,
            "title": self.title,
            "artist": self.artist,
            "score": self.score,
            "seed": self.seed,
        }
        if self.explanation is not None:
            payload["explanation"] = self.explanation.as_dict()
        return payload


@dataclass(frozen=True)
class RecommendationResult:
    """A ranked list plus the settings that produced it."""

    items: tuple[Recommendation, ...]
    strategy: str
    aggregation: str
    seed_indices: tuple[int, ...]
    n_candidates: int

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "aggregation": self.aggregation,
            "n_candidates": self.n_candidates,
            "seeds": list(self.seed_indices),
            "recommendations": [item.as_dict() for item in self.items],
        }


class Recommender:
    """Ranks catalogue tracks against a seed playlist.

    The catalogue's latent representation is computed once and row-normalised
    once, so each query is a single mat-vec rather than a Python loop over
    songs -- the original per-query loop cost 23 ms at 10,000 songs against
    0.31 ms vectorised.
    """

    def __init__(self, model: LatentModel, catalog: Catalog):
        if tuple(model.feature_names) != tuple(catalog.feature_names):
            raise ValueError(
                "model and catalog disagree about features:\n"
                f"  model:   {model.feature_names}\n"
                f"  catalog: {catalog.feature_names}"
            )
        self.model = model
        self.catalog = catalog
        self.latent = model.transform(catalog.features)
        self._unit = normalize_rows(self.latent)
        self._popularity = catalog.popularity()

    # -- core scoring -------------------------------------------------------

    def similarities(self, seed_indices: list[int]) -> np.ndarray:
        """Per-seed cosine similarity against the catalogue.

        Returns
        -------
        np.ndarray, shape (n_seeds, n_songs)
        """
        if not seed_indices:
            return np.zeros((0, len(self.catalog)))
        return np.clip(self._unit[list(seed_indices)] @ self._unit.T, -1.0, 1.0)

    def _aggregate(self, per_seed: np.ndarray, method: str) -> np.ndarray:
        """Collapse per-seed scores into one score per candidate."""
        if per_seed.shape[0] == 0:
            return np.zeros(len(self.catalog))
        if method == "max":
            return per_seed.max(axis=0)
        if method == "mean":
            return per_seed.mean(axis=0)
        if method == "borda":
            # Rank-based aggregation: each seed ranks every candidate, and we
            # sum the ranks. Immune to one seed having systematically higher
            # cosine values than another, which is exactly the failure mode of
            # `max` on a playlist with one unusual track.
            n = per_seed.shape[1]
            points = np.zeros(n)
            for row in per_seed:
                order = np.argsort(-row, kind="stable")
                points[order] += np.arange(n, 0, -1)
            return points / (per_seed.shape[0] * n)
        raise ValueError(f"unknown aggregation {method!r}; expected one of {AGGREGATIONS}")

    def _build_scores(
        self,
        seed_indices: list[int],
        aggregation: str,
        negative_indices: list[int] | None,
        negative_weight: float,
        novelty_weight: float,
    ) -> np.ndarray:
        per_seed = self.similarities(seed_indices)
        scores = self._aggregate(per_seed, aggregation)

        if negative_indices:
            # Rocchio: push the query away from tracks the listener rejected.
            penalty = self._aggregate(self.similarities(list(negative_indices)), aggregation)
            scores = scores - negative_weight * penalty

        if novelty_weight:
            # De-bias toward the tail of the catalogue. Without this the
            # recommender's favourite songs are simply the popular ones, since
            # popular tracks are over-represented near every centroid.
            scores = scores - novelty_weight * self._popularity

        return scores

    def _excluded(self, seed_indices: list[int], extra: set[int] | None) -> set[int]:
        excluded = set(int(i) for i in seed_indices)
        if extra:
            excluded |= {int(i) for i in extra}
        return excluded

    def _make_item(
        self,
        index: int,
        score: float,
        seed_index: int | None,
        explain: bool,
    ) -> Recommendation:
        explanation = None
        if explain and seed_index is not None:
            explanation = explain_match(
                self.latent[seed_index], self.latent[index], self.model
            )
        return Recommendation(
            index=int(index),
            title=self.catalog.titles[index],
            artist=self.catalog.artists[index],
            score=float(score),
            seed=self.catalog.titles[seed_index] if seed_index is not None else None,
            explanation=explanation,
        )

    def _apply_artist_cap(
        self, ranked: np.ndarray, scores: np.ndarray, n: int, max_per_artist: int | None
    ) -> list[int]:
        """Take the top ``n`` from ``ranked``, honouring a per-artist cap."""
        if max_per_artist is None or max_per_artist <= 0:
            return [int(i) for i in ranked[:n]]
        chosen: list[int] = []
        counts: dict[str, int] = {}
        for idx in ranked:
            idx = int(idx)
            artist = self.catalog.artists[idx].lower().strip()
            if counts.get(artist, 0) >= max_per_artist:
                continue
            counts[artist] = counts.get(artist, 0) + 1
            chosen.append(idx)
            if len(chosen) >= n:
                break
        return chosen

    # -- strategies ---------------------------------------------------------

    def recommend(
        self,
        seed_indices: list[int],
        n: int = 10,
        strategy: str = "overall_top",
        aggregation: str = "max",
        max_per_artist: int | None = 2,
        novelty_weight: float = 0.0,
        negative_indices: list[int] | None = None,
        negative_weight: float = 0.5,
        mmr_lambda: float = 0.7,
        exclude: set[int] | None = None,
        explain: bool = False,
    ) -> RecommendationResult:
        """Produce ``n`` recommendations for a seed playlist.

        Parameters
        ----------
        seed_indices : list[int]
            Catalogue indices of the input playlist.
        n : int
        strategy : {"overall_top", "one_per_song", "mmr", "centroid"}
        aggregation : {"max", "mean", "borda"}
            How per-seed similarities collapse to one score. Ignored by
            ``one_per_song`` (which is inherently per-seed) and ``centroid``.
        max_per_artist : int, optional
            Cap on tracks from any one artist. ``None`` disables.
        novelty_weight : float
            Subtracts ``weight * popularity`` from each score. 0 disables;
            0.05-0.2 is a useful range.
        negative_indices : list[int], optional
            Tracks to steer away from.
        negative_weight : float
        mmr_lambda : float
            ``mmr`` only. 1.0 is pure relevance, 0.0 pure diversity.
        exclude : set[int], optional
            Additional indices to withhold.
        explain : bool
            Attach a per-result :class:`~eigengrooves.explain.Explanation`.

        Returns
        -------
        RecommendationResult
        """
        seed_indices = [int(i) for i in seed_indices]
        for idx in seed_indices:
            if not 0 <= idx < len(self.catalog):
                raise IndexError(f"seed index {idx} out of range for {len(self.catalog)} songs")
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")

        if not seed_indices:
            return RecommendationResult((), strategy, aggregation, (), len(self.catalog))

        excluded = self._excluded(seed_indices, exclude)

        if strategy == "one_per_song":
            items = self._one_per_song(
                seed_indices, n, excluded, novelty_weight, negative_indices,
                negative_weight, max_per_artist, explain,
            )
        elif strategy == "mmr":
            items = self._mmr(
                seed_indices, n, aggregation, excluded, mmr_lambda, novelty_weight,
                negative_indices, negative_weight, max_per_artist, explain,
            )
        elif strategy == "centroid":
            items = self._centroid(
                seed_indices, n, excluded, novelty_weight, negative_indices,
                negative_weight, max_per_artist, explain,
            )
        else:
            items = self._overall_top(
                seed_indices, n, aggregation, excluded, novelty_weight,
                negative_indices, negative_weight, max_per_artist, explain,
            )

        return RecommendationResult(
            items=tuple(items),
            strategy=strategy,
            aggregation=aggregation,
            seed_indices=tuple(seed_indices),
            n_candidates=len(self.catalog) - len(excluded),
        )

    def _overall_top(
        self, seeds, n, aggregation, excluded, novelty_weight,
        negative_indices, negative_weight, max_per_artist, explain,
    ) -> list[Recommendation]:
        scores = self._build_scores(
            seeds, aggregation, negative_indices, negative_weight, novelty_weight
        )
        # Over-fetch so the artist cap has candidates to fall back on.
        pool = n * 8 if max_per_artist else n
        ranked = top_k_indices(scores, min(pool, len(self.catalog)), exclude=excluded)
        chosen = self._apply_artist_cap(ranked, scores, n, max_per_artist)

        per_seed = self.similarities(seeds)
        items = []
        for idx in chosen:
            # Attribute the result to whichever seed matched it most strongly.
            best_seed = seeds[int(np.argmax(per_seed[:, idx]))] if per_seed.size else None
            items.append(self._make_item(idx, scores[idx], best_seed, explain))
        return items

    def _one_per_song(
        self, seeds, n, excluded, novelty_weight, negative_indices,
        negative_weight, max_per_artist, explain,
    ) -> list[Recommendation]:
        per_seed = self.similarities(seeds)
        if novelty_weight:
            per_seed = per_seed - novelty_weight * self._popularity[None, :]
        if negative_indices:
            penalty = self.similarities(list(negative_indices)).max(axis=0)
            per_seed = per_seed - negative_weight * penalty[None, :]

        taken: set[int] = set()
        artist_counts: dict[str, int] = {}
        items: list[Recommendation] = []

        # Cycle through seeds so that with n > len(seeds) every seed gets a
        # second pick before any seed gets a third.
        seed_order = [seeds[i % len(seeds)] for i in range(max(n, len(seeds)))]
        for position, seed in enumerate(seed_order):
            if len(items) >= n:
                break
            row = per_seed[seeds.index(seed)]
            blocked = excluded | taken
            for candidate in top_k_indices(row, min(64, len(self.catalog)), exclude=blocked):
                candidate = int(candidate)
                artist = self.catalog.artists[candidate].lower().strip()
                if max_per_artist and artist_counts.get(artist, 0) >= max_per_artist:
                    continue
                taken.add(candidate)
                artist_counts[artist] = artist_counts.get(artist, 0) + 1
                items.append(self._make_item(candidate, row[candidate], seed, explain))
                break
        return items[:n]

    def _centroid(
        self, seeds, n, excluded, novelty_weight, negative_indices,
        negative_weight, max_per_artist, explain,
    ) -> list[Recommendation]:
        """Query by the playlist's mean latent vector -- its "taste vector".

        Different from ``mean`` aggregation: this averages the *vectors* and
        then measures similarity once, where ``mean`` aggregation measures
        similarity per seed and then averages the scores. Averaging vectors
        first is more sensitive to a playlist that is coherent in direction,
        and less to one seed that happens to be an outlier.
        """
        centroid = self.latent[seeds].mean(axis=0)
        if negative_indices:
            centroid = centroid - negative_weight * self.latent[list(negative_indices)].mean(axis=0)

        norm = np.linalg.norm(centroid)
        if norm < 1e-12:
            return []
        scores = np.clip(self._unit @ (centroid / norm), -1.0, 1.0)
        if novelty_weight:
            scores = scores - novelty_weight * self._popularity

        pool = n * 8 if max_per_artist else n
        ranked = top_k_indices(scores, min(pool, len(self.catalog)), exclude=excluded)
        chosen = self._apply_artist_cap(ranked, scores, n, max_per_artist)

        per_seed = self.similarities(seeds)
        return [
            self._make_item(
                idx, scores[idx], seeds[int(np.argmax(per_seed[:, idx]))], explain
            )
            for idx in chosen
        ]

    def _mmr(
        self, seeds, n, aggregation, excluded, lam, novelty_weight,
        negative_indices, negative_weight, max_per_artist, explain,
    ) -> list[Recommendation]:
        """Maximal Marginal Relevance.

        Greedily selects the candidate maximising::

            lambda * relevance(d) - (1 - lambda) * max_{s in selected} sim(d, s)

        so each pick must be both close to the playlist and unlike what has
        already been chosen. Only the top slice of the catalogue by relevance
        is considered -- MMR is quadratic in the candidate pool and the tail
        is never going to win.
        """
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"mmr_lambda must be in [0, 1], got {lam}")

        relevance = self._build_scores(
            seeds, aggregation, negative_indices, negative_weight, novelty_weight
        )
        pool_size = min(max(n * 20, 100), len(self.catalog))
        pool = top_k_indices(relevance, pool_size, exclude=excluded)
        if pool.size == 0:
            return []

        pool_unit = self._unit[pool]
        pool_relevance = relevance[pool]
        redundancy = np.full(pool.size, -np.inf)
        available = np.ones(pool.size, dtype=bool)

        selected: list[int] = []
        artist_counts: dict[str, int] = {}

        while len(selected) < n and available.any():
            if not selected:
                mmr_scores = pool_relevance.copy()
            else:
                mmr_scores = lam * pool_relevance - (1.0 - lam) * redundancy
            mmr_scores = np.where(available, mmr_scores, -np.inf)

            pick = int(np.argmax(mmr_scores))
            if not np.isfinite(mmr_scores[pick]):
                break
            available[pick] = False

            catalog_idx = int(pool[pick])
            artist = self.catalog.artists[catalog_idx].lower().strip()
            if max_per_artist and artist_counts.get(artist, 0) >= max_per_artist:
                continue
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
            selected.append(catalog_idx)

            # Update each remaining candidate's similarity to the selected set.
            sims = np.clip(pool_unit @ pool_unit[pick], -1.0, 1.0)
            redundancy = np.maximum(redundancy, sims)

        per_seed = self.similarities(seeds)
        return [
            self._make_item(
                idx, relevance[idx], seeds[int(np.argmax(per_seed[:, idx]))], explain
            )
            for idx in selected
        ]
