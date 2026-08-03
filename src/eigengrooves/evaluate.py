"""
Offline evaluation.

The protocol
------------
Recommendation has no labelled ground truth here -- nobody recorded which songs
a listener would have enjoyed. So we borrow one: tracks that share an *artist*
(or a genre, when the dataset carries genre labels) are treated as mutually
relevant.

For each evaluation group we sample ``seed_size`` tracks as the query playlist
and treat the group's remaining tracks as the relevant set. A system is scored
on how much of that held-out remainder it recovers.

What this protocol is and is not
--------------------------------
It measures whether a system recovers *known stylistic neighbours* from audio
features alone. That is a real and checkable question, and it is the one the
project's premise depends on.

It is **not** a measure of whether a human would enjoy the recommendations, and
it is biased against the very thing the README celebrates -- cross-genre
discovery is penalised here, because a good cross-genre match is by
construction outside the ground-truth group. Read the numbers as "does the
latent space encode real structure", not as "is this a good product". Both
grouping keys are reported so the bias is at least visible from two angles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .baselines import Ranker
from .catalog import Catalog
from .metrics import (
    catalog_coverage,
    hit_rate_at_k,
    intra_list_diversity,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    shannon_entropy,
)

__all__ = ["EvaluationGroup", "SystemScores", "build_groups", "compare_rankers", "evaluate_ranker"]


@dataclass(frozen=True)
class EvaluationGroup:
    """One query: seed tracks plus the relevant set held out from them."""

    key: str
    seeds: tuple[int, ...]
    relevant: frozenset[int]


@dataclass
class SystemScores:
    """Averaged metrics for one ranking system."""

    name: str
    k: int
    n_queries: int
    metrics: dict[str, float] = field(default_factory=dict)

    def as_row(self, columns: list[str]) -> list[str]:
        return [self.name] + [f"{self.metrics.get(c, float('nan')):.4f}" for c in columns]


def build_groups(
    catalog: Catalog,
    group_by: str = "artist",
    seed_size: int = 3,
    min_group_size: int = 6,
    max_groups: int | None = 400,
    random_state: int | None = 0,
) -> list[EvaluationGroup]:
    """Construct evaluation queries by holding out members of a group.

    Parameters
    ----------
    catalog : Catalog
    group_by : {"artist", "genre"}
        ``genre`` requires a ``genre`` column; groups are much larger and the
        task correspondingly easier, so scores are not comparable across keys.
    seed_size : int
        Tracks used as the query playlist.
    min_group_size : int
        Groups smaller than this are skipped -- with too few members the
        relevant set is a rounding error.
    max_groups : int, optional
        Cap for runtime. Groups are sampled without replacement.
    random_state : int, optional

    Returns
    -------
    list[EvaluationGroup]
    """
    if group_by == "artist":
        labels = [a.lower().strip() for a in catalog.artists]
    elif group_by == "genre":
        if "genre" not in catalog.frame.columns:
            raise ValueError("catalog has no 'genre' column; use group_by='artist'")
        labels = catalog.frame["genre"].astype(str).str.lower().str.strip().tolist()
    else:
        raise ValueError(f"unknown group_by {group_by!r}; expected 'artist' or 'genre'")

    buckets: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        if label and label not in ("unknown", "nan"):
            buckets.setdefault(label, []).append(index)

    eligible = {k: v for k, v in buckets.items() if len(v) >= min_group_size}
    if not eligible:
        raise ValueError(
            f"no {group_by} group has at least {min_group_size} tracks; "
            "lower min_group_size or use a larger catalogue"
        )

    rng = np.random.default_rng(random_state)
    keys = sorted(eligible)
    if max_groups is not None and len(keys) > max_groups:
        keys = [keys[i] for i in rng.choice(len(keys), size=max_groups, replace=False)]

    groups: list[EvaluationGroup] = []
    for key in keys:
        members = np.array(eligible[key])
        picked = rng.choice(members, size=min(seed_size, len(members) - 1), replace=False)
        seeds = tuple(int(i) for i in picked)
        relevant = frozenset(int(i) for i in members if int(i) not in set(seeds))
        if relevant:
            groups.append(EvaluationGroup(key=key, seeds=seeds, relevant=relevant))
    return groups


def evaluate_ranker(
    ranker: Ranker,
    groups: list[EvaluationGroup],
    catalog: Catalog,
    shared_embedding: np.ndarray,
    k: int = 10,
) -> SystemScores:
    """Score one ranking system across every evaluation group.

    Parameters
    ----------
    ranker : Ranker
    groups : list[EvaluationGroup]
    catalog : Catalog
    shared_embedding : np.ndarray
        System-independent embedding used for the diversity metric.
    k : int
        Cutoff for the @k metrics.

    Returns
    -------
    SystemScores
    """
    if not groups:
        raise ValueError("no evaluation groups supplied")

    popularity = catalog.popularity()
    accumulators: dict[str, list[float]] = {
        name: []
        for name in (
            "hit_rate", "precision", "recall", "mrr", "ndcg",
            "diversity", "novelty", "artist_entropy",
        )
    }
    all_ranked: list[np.ndarray] = []

    for group in groups:
        # Seeds are withheld so a system cannot score by returning the query.
        ranked = ranker.rank(list(group.seeds), k, set(group.seeds))
        all_ranked.append(ranked)
        relevant = set(group.relevant)

        accumulators["hit_rate"].append(hit_rate_at_k(ranked, relevant, k))
        accumulators["precision"].append(precision_at_k(ranked, relevant, k))
        accumulators["recall"].append(recall_at_k(ranked, relevant, k))
        accumulators["mrr"].append(reciprocal_rank(ranked, relevant))
        accumulators["ndcg"].append(ndcg_at_k(ranked, relevant, k))
        accumulators["diversity"].append(intra_list_diversity(ranked, shared_embedding))
        accumulators["novelty"].append(novelty(ranked, popularity))
        accumulators["artist_entropy"].append(
            shannon_entropy([catalog.artists[int(i)] for i in ranked])
        )

    metrics = {name: float(np.mean(values)) for name, values in accumulators.items()}
    metrics["coverage"] = catalog_coverage(all_ranked, len(catalog))

    return SystemScores(
        name=ranker.name, k=k, n_queries=len(groups), metrics=metrics
    )


def compare_rankers(
    rankers: list[Ranker],
    groups: list[EvaluationGroup],
    catalog: Catalog,
    shared_embedding: np.ndarray,
    k: int = 10,
) -> list[SystemScores]:
    """Score several systems on identical queries, sorted by NDCG."""
    results = [
        evaluate_ranker(r, groups, catalog, shared_embedding, k=k) for r in rankers
    ]
    return sorted(results, key=lambda s: s.metrics.get("ndcg", 0.0), reverse=True)


def format_comparison(results: list[SystemScores], k: int = 10) -> str:
    """Render a comparison as a fixed-width table."""
    columns = [
        "hit_rate", "recall", "precision", "mrr", "ndcg",
        "diversity", "novelty", "coverage",
    ]
    headers = ["system"] + [f"{c}@{k}" if c in ("hit_rate", "recall", "precision", "ndcg") else c
                            for c in columns]

    rows = [r.as_row(columns) for r in results]
    widths = [
        max(len(headers[i]), max((len(row[i]) for row in rows), default=0))
        for i in range(len(headers))
    ]

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    out = [line(headers), "  ".join("-" * w for w in widths)]
    out.extend(line(row) for row in rows)
    return "\n".join(out)
