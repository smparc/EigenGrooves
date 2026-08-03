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

__all__ = [
    "ComparisonTest",
    "EvaluationGroup",
    "SystemScores",
    "bootstrap_ci",
    "build_groups",
    "compare_rankers",
    "evaluate_ranker",
    "format_comparison",
    "paired_bootstrap_test",
]


@dataclass(frozen=True)
class EvaluationGroup:
    """One query: seed tracks plus the relevant set held out from them."""

    key: str
    seeds: tuple[int, ...]
    relevant: frozenset[int]


@dataclass
class SystemScores:
    """Averaged metrics for one ranking system, plus the raw per-query values.

    Keeping ``per_query`` is what makes honest comparison possible. Two systems
    scoring 0.0371 and 0.0340 on 219 queries may or may not actually differ;
    the mean alone cannot say, and the queries are shared between systems, so
    the comparison must be *paired*. See :func:`paired_bootstrap_test`.
    """

    name: str
    k: int
    n_queries: int
    metrics: dict[str, float] = field(default_factory=dict)
    per_query: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def as_row(self, columns: list[str]) -> list[str]:
        return [self.name] + [f"{self.metrics.get(c, float('nan')):.4f}" for c in columns]

    def ci(self, metric: str, confidence: float = 0.95) -> tuple[float, float]:
        """Bootstrap confidence interval for one metric's mean."""
        values = self.per_query.get(metric)
        if values is None or values.size == 0:
            return (float("nan"), float("nan"))
        return bootstrap_ci(values, confidence=confidence)


@dataclass(frozen=True)
class ComparisonTest:
    """Result of a paired comparison between two systems on one metric."""

    metric: str
    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    n_queries: int

    @property
    def significant(self) -> bool:
        """True when the confidence interval for the difference excludes zero."""
        return not (self.ci_low <= 0.0 <= self.ci_high)

    def summary(self) -> str:
        verdict = "significant" if self.significant else "NOT significant"
        direction = ">" if self.difference > 0 else "<"
        return (
            f"{self.name_a} ({self.mean_a:.4f}) {direction} {self.name_b} ({self.mean_b:.4f}) "
            f"on {self.metric}: Δ={self.difference:+.4f} "
            f"[95% CI {self.ci_low:+.4f}, {self.ci_high:+.4f}], p={self.p_value:.3f} — {verdict}"
        )


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    random_state: int | None = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Resamples the per-query values with replacement and reads the empirical
    quantiles. Makes no distributional assumption, which matters here: ranking
    metrics are bounded, heavily zero-inflated and nothing like Gaussian, so a
    textbook t-interval would be misleading.
    """
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        return (float(values[0]), float(values[0]))

    rng = np.random.default_rng(random_state)
    draws = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def paired_bootstrap_test(
    a: SystemScores,
    b: SystemScores,
    metric: str = "ndcg",
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    random_state: int | None = 0,
) -> ComparisonTest:
    """Paired bootstrap test of whether ``a`` beats ``b`` on ``metric``.

    Both systems are evaluated on identical queries, so the per-query
    differences are paired and the resampling must draw *query indices*, not
    each system independently. Ignoring the pairing throws away the
    query-difficulty variance that both systems share and badly overstates the
    uncertainty.

    The p-value is two-sided, computed as the fraction of resampled mean
    differences falling on the opposite side of zero from the observed
    difference (doubled, and floored at 1/n_resamples since a bootstrap cannot
    resolve p below its own resolution).
    """
    values_a = a.per_query.get(metric)
    values_b = b.per_query.get(metric)
    if values_a is None or values_b is None:
        raise ValueError(f"metric {metric!r} not recorded for both systems")
    if values_a.size != values_b.size:
        raise ValueError(
            f"systems were evaluated on different query counts: "
            f"{values_a.size} vs {values_b.size}"
        )
    if values_a.size == 0:
        raise ValueError("no queries to compare")

    differences = values_a - values_b
    observed = float(differences.mean())

    rng = np.random.default_rng(random_state)
    draws = rng.integers(0, differences.size, size=(n_resamples, differences.size))
    resampled = differences[draws].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.quantile(resampled, alpha))
    ci_high = float(np.quantile(resampled, 1.0 - alpha))

    # Centre the resamples on zero to approximate the null distribution.
    centred = resampled - observed
    tail = float(np.mean(np.abs(centred) >= abs(observed)))
    p_value = max(tail, 1.0 / n_resamples)

    return ComparisonTest(
        metric=metric,
        name_a=a.name,
        name_b=b.name,
        mean_a=float(values_a.mean()),
        mean_b=float(values_b.mean()),
        difference=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=min(p_value, 1.0),
        n_queries=int(differences.size),
    )


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

    per_query = {name: np.asarray(values, dtype=float) for name, values in accumulators.items()}
    metrics = {name: float(values.mean()) for name, values in per_query.items()}
    # Coverage is a property of the whole run, not of any single query, so it
    # has no per-query counterpart and cannot be bootstrapped this way.
    metrics["coverage"] = catalog_coverage(all_ranked, len(catalog))

    return SystemScores(
        name=ranker.name,
        k=k,
        n_queries=len(groups),
        metrics=metrics,
        per_query=per_query,
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
