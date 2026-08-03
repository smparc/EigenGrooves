"""Tests for similarity, recommendation strategies, explanations and metrics."""

from __future__ import annotations

import numpy as np
import pytest

from eigengrooves import Recommender, build_groups, compare_rankers, fit_latent_model
from eigengrooves.baselines import (
    LatentRanker,
    PopularityRanker,
    RandomRanker,
    RawFeatureRanker,
)
from eigengrooves.evaluate import evaluate_ranker, format_comparison
from eigengrooves.explain import explain_match
from eigengrooves.metrics import (
    average_precision_at_k,
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
from eigengrooves.similarity import (
    cosine_similarity,
    cosine_similarity_matrix,
    normalize_rows,
    query_similarities,
    top_k_indices,
)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_known_values():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_similarity_is_scale_invariant():
    assert cosine_similarity([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_cosine_similarity_of_zero_vector_is_zero():
    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


def test_cosine_similarity_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity([1, 2], [1, 2, 3])


def test_vectorized_similarity_matches_the_scalar_definition(rng):
    """The optimisation must not change the answer."""
    catalog = rng.normal(size=(200, 5))
    query = rng.normal(size=5)
    vectorized = query_similarities(query, catalog)
    scalar = np.array([cosine_similarity(query, row) for row in catalog])
    assert np.allclose(vectorized, scalar, atol=1e-12)


def test_similarity_matrix_is_symmetric_with_unit_diagonal(rng):
    S = cosine_similarity_matrix(rng.normal(size=(40, 6)))
    assert np.allclose(S, S.T)
    assert np.allclose(np.diag(S), 1.0)
    assert S.min() >= -1.0 and S.max() <= 1.0


def test_normalize_rows_leaves_zero_rows_alone():
    X = np.array([[3.0, 4.0], [0.0, 0.0]])
    normalized = normalize_rows(X)
    assert np.allclose(normalized[0], [0.6, 0.8])
    assert np.allclose(normalized[1], 0.0)


def test_top_k_indices_orders_descending_and_honours_exclusions():
    scores = np.array([0.1, 0.9, 0.5, 0.7, 0.3])
    assert list(top_k_indices(scores, 3)) == [1, 3, 2]
    assert list(top_k_indices(scores, 3, exclude={1, 3})) == [2, 4, 0]


def test_top_k_indices_handles_k_larger_than_catalogue():
    assert len(top_k_indices(np.array([0.5, 0.2]), 10)) == 2


def test_top_k_indices_with_everything_excluded():
    assert len(top_k_indices(np.array([0.5, 0.2]), 5, exclude={0, 1})) == 0


# ---------------------------------------------------------------------------
# Recommendation strategies
# ---------------------------------------------------------------------------


STRATEGIES = ["overall_top", "one_per_song", "mmr", "centroid"]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_strategies_return_the_requested_count(recommender, strategy):
    result = recommender.recommend([1, 2, 3], n=8, strategy=strategy)
    assert len(result) == 8


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_strategies_return_unique_indices(recommender, strategy):
    result = recommender.recommend([1, 2, 3], n=12, strategy=strategy)
    indices = [item.index for item in result]
    assert len(indices) == len(set(indices))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_empty_playlist_returns_nothing(recommender, strategy):
    assert len(recommender.recommend([], n=5, strategy=strategy)) == 0


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_zero_recommendations_requested(recommender, strategy):
    assert len(recommender.recommend([1, 2], n=0, strategy=strategy)) == 0


def test_overall_top_is_ordered_by_score(recommender):
    result = recommender.recommend([1, 2, 3], n=10, strategy="overall_top",
                                   max_per_artist=None)
    scores = [item.score for item in result]
    assert scores == sorted(scores, reverse=True)


def test_one_per_song_attributes_each_result_to_a_seed(recommender, small_catalog):
    seeds = [1, 2, 3]
    result = recommender.recommend(seeds, n=3, strategy="one_per_song")
    seed_titles = {small_catalog.titles[i] for i in seeds}
    assert len(result) == 3
    assert {item.seed for item in result} == seed_titles


def test_one_per_song_can_exceed_seed_count(recommender):
    """Requesting more than one per seed should cycle rather than truncate."""
    result = recommender.recommend([1, 2], n=6, strategy="one_per_song")
    assert len(result) == 6


def test_mmr_lambda_trades_relevance_for_diversity(recommender, scaled_features):
    seeds = [1, 2, 3]
    relevance_only = recommender.recommend(
        seeds, n=10, strategy="mmr", mmr_lambda=1.0, max_per_artist=None
    )
    diversity_heavy = recommender.recommend(
        seeds, n=10, strategy="mmr", mmr_lambda=0.15, max_per_artist=None
    )

    diverse_spread = intra_list_diversity(
        [i.index for i in diversity_heavy], scaled_features
    )
    focused_spread = intra_list_diversity(
        [i.index for i in relevance_only], scaled_features
    )
    assert diverse_spread > focused_spread

    mean_relevance_focused = np.mean([i.score for i in relevance_only])
    mean_relevance_diverse = np.mean([i.score for i in diversity_heavy])
    assert mean_relevance_focused >= mean_relevance_diverse


def test_mmr_rejects_out_of_range_lambda(recommender):
    with pytest.raises(ValueError, match="mmr_lambda"):
        recommender.recommend([1, 2], n=5, strategy="mmr", mmr_lambda=1.5)


def test_artist_cap_is_respected(recommender, small_catalog):
    result = recommender.recommend([1, 2, 3], n=20, strategy="overall_top",
                                   max_per_artist=2)
    counts: dict[str, int] = {}
    for item in result:
        key = item.artist.lower().strip()
        counts[key] = counts.get(key, 0) + 1
    assert max(counts.values()) <= 2


def test_disabling_the_artist_cap_allows_repeats(recommender):
    capped = recommender.recommend([1], n=20, strategy="overall_top", max_per_artist=1)
    uncapped = recommender.recommend([1], n=20, strategy="overall_top", max_per_artist=None)
    capped_artists = [i.artist for i in capped]
    assert len(set(capped_artists)) == len(capped_artists)
    # The uncapped list is free to repeat; it need not, but it must not be
    # constrained to unique artists by construction.
    assert len(uncapped) == 20


def test_novelty_weight_shifts_results_toward_the_tail(recommender, small_catalog):
    popularity = small_catalog.popularity()
    plain = recommender.recommend([1, 2, 3], n=15, strategy="overall_top",
                                  max_per_artist=None)
    novel = recommender.recommend([1, 2, 3], n=15, strategy="overall_top",
                                  novelty_weight=0.5, max_per_artist=None)
    assert np.mean(popularity[[i.index for i in novel]]) < np.mean(
        popularity[[i.index for i in plain]]
    )


def test_negative_feedback_pushes_away_from_avoided_tracks(recommender):
    seeds = [1, 2, 3]
    plain = recommender.recommend(seeds, n=10, strategy="overall_top", max_per_artist=None)
    avoided = [item.index for item in plain[:3]]
    steered = recommender.recommend(
        seeds, n=10, strategy="overall_top", negative_indices=avoided,
        negative_weight=1.0, max_per_artist=None,
    )
    overlap = set(avoided) & {item.index for item in steered}
    assert len(overlap) < len(avoided)


@pytest.mark.parametrize("aggregation", ["max", "mean", "borda"])
def test_aggregations_all_produce_valid_results(recommender, aggregation):
    result = recommender.recommend(
        [1, 2, 3], n=10, strategy="overall_top", aggregation=aggregation
    )
    assert len(result) == 10
    assert all(np.isfinite(item.score) for item in result)


def test_unknown_strategy_is_rejected(recommender):
    with pytest.raises(ValueError, match="unknown strategy"):
        recommender.recommend([1], n=5, strategy="telepathy")


def test_unknown_aggregation_is_rejected(recommender):
    with pytest.raises(ValueError, match="unknown aggregation"):
        recommender.recommend([1], n=5, aggregation="vibes")


def test_out_of_range_seed_is_rejected(recommender):
    with pytest.raises(IndexError, match="out of range"):
        recommender.recommend([10**9], n=5)


def test_negative_n_is_rejected(recommender):
    with pytest.raises(ValueError, match="non-negative"):
        recommender.recommend([1], n=-1)


def test_recommender_rejects_mismatched_model_and_catalog(small_catalog):
    model = fit_latent_model(
        small_catalog.features[:, :4], small_catalog.feature_names[:4], k=2
    )
    with pytest.raises(ValueError, match="disagree about features"):
        Recommender(model, small_catalog)


def test_result_serialises_to_json_friendly_dict(recommender):
    payload = recommender.recommend([1, 2], n=3, explain=True).as_dict()
    import json

    json.dumps(payload)  # must not raise
    assert payload["strategy"]
    assert len(payload["recommendations"]) == 3
    assert "explanation" in payload["recommendations"][0]


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------


def test_explanation_contributions_sum_to_the_score(model, rng):
    a = rng.normal(size=model.k)
    b = rng.normal(size=model.k)
    explanation = explain_match(a, b, model, top_n=model.k)
    total = sum(c.contribution for c in explanation.contributions)
    assert total == pytest.approx(explanation.score, abs=1e-10)
    assert explanation.score == pytest.approx(cosine_similarity(a, b), abs=1e-10)


def test_explanation_of_identical_vectors_is_perfect(model, rng):
    v = rng.normal(size=model.k)
    assert explain_match(v, v, model).score == pytest.approx(1.0)


def test_explanation_handles_zero_vectors(model):
    explanation = explain_match(np.zeros(model.k), np.ones(model.k), model)
    assert explanation.score == 0.0
    assert explanation.contributions == ()


def test_explanation_summary_is_human_readable(recommender):
    result = recommender.recommend([1, 2], n=3, explain=True)
    for item in result:
        assert isinstance(item.explanation.summary(), str)
        assert item.explanation.summary()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_on_a_perfect_ranking():
    ranked = [1, 2, 3, 4, 5]
    relevant = {1, 2, 3}
    assert hit_rate_at_k(ranked, relevant, 5) == 1.0
    assert recall_at_k(ranked, relevant, 5) == pytest.approx(1.0)
    assert precision_at_k(ranked, relevant, 3) == pytest.approx(1.0)
    assert reciprocal_rank(ranked, relevant) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, relevant, 5) == pytest.approx(1.0)


def test_metrics_on_a_ranking_with_no_hits():
    ranked = [7, 8, 9]
    relevant = {1, 2}
    assert hit_rate_at_k(ranked, relevant, 3) == 0.0
    assert recall_at_k(ranked, relevant, 3) == 0.0
    assert reciprocal_rank(ranked, relevant) == 0.0
    assert ndcg_at_k(ranked, relevant, 3) == 0.0


def test_reciprocal_rank_uses_the_first_hit():
    assert reciprocal_rank([9, 9, 1], {1}) == pytest.approx(1 / 3)


def test_ndcg_rewards_earlier_hits():
    relevant = {1, 2}
    early = ndcg_at_k([1, 2, 8, 9], relevant, 4)
    late = ndcg_at_k([8, 9, 1, 2], relevant, 4)
    assert early > late
    assert 0.0 <= late <= early <= 1.0


def test_average_precision_is_bounded():
    assert 0.0 <= average_precision_at_k([1, 5, 2], {1, 2}, 3) <= 1.0


def test_metrics_with_empty_relevant_set():
    for fn in (hit_rate_at_k, recall_at_k, precision_at_k, ndcg_at_k):
        assert fn([1, 2, 3], set(), 3) == 0.0
    assert reciprocal_rank([1, 2, 3], set()) == 0.0


def test_intra_list_diversity_extremes(rng):
    identical = np.tile(rng.normal(size=(1, 5)), (4, 1))
    assert intra_list_diversity([0, 1, 2, 3], identical) == pytest.approx(0.0, abs=1e-9)

    orthogonal = np.eye(4)
    assert intra_list_diversity([0, 1, 2, 3], orthogonal) == pytest.approx(1.0)


def test_intra_list_diversity_of_a_single_item_is_zero(rng):
    assert intra_list_diversity([0], rng.normal(size=(5, 3))) == 0.0


def test_catalog_coverage():
    assert catalog_coverage([[0, 1], [1, 2]], 10) == pytest.approx(0.3)
    assert catalog_coverage([], 10) == 0.0


def test_novelty_prefers_unpopular_items():
    popularity = np.array([1.0, 0.5, 0.0])
    assert novelty([2], popularity) > novelty([0], popularity)


def test_shannon_entropy_bounds():
    assert shannon_entropy(["a", "a", "a"]) == pytest.approx(0.0)
    assert shannon_entropy(["a", "b", "c", "d"]) == pytest.approx(2.0)
    assert shannon_entropy([]) == 0.0


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------


def test_build_groups_holds_out_relevant_tracks(small_catalog):
    groups = build_groups(small_catalog, group_by="artist", seed_size=2,
                          min_group_size=5, random_state=0)
    assert groups
    for group in groups:
        assert len(group.seeds) <= 2
        assert group.relevant
        assert not (set(group.seeds) & set(group.relevant))


def test_build_groups_by_genre(small_catalog):
    groups = build_groups(small_catalog, group_by="genre", seed_size=3,
                          min_group_size=10, random_state=0)
    assert groups


def test_build_groups_rejects_unknown_key(small_catalog):
    with pytest.raises(ValueError, match="unknown group_by"):
        build_groups(small_catalog, group_by="mood")


def test_build_groups_errors_when_nothing_qualifies(small_catalog):
    with pytest.raises(ValueError, match="at least"):
        build_groups(small_catalog, min_group_size=10**6)


def test_evaluation_ranks_real_systems_above_random(small_catalog, scaled_features, model):
    """The floor test: any working system must beat uniform random selection."""
    groups = build_groups(small_catalog, group_by="artist", seed_size=3,
                          min_group_size=6, max_groups=80, random_state=0)

    latent = LatentRanker(model, small_catalog)
    raw = RawFeatureRanker(scaled_features)
    chance = RandomRanker(len(small_catalog), random_state=0)

    scores = {
        r.name: evaluate_ranker(r, groups, small_catalog, scaled_features, k=10)
        for r in (latent, raw, chance)
    }
    assert scores[latent.name].metrics["ndcg"] > scores[chance.name].metrics["ndcg"]
    assert scores[raw.name].metrics["ndcg"] > scores[chance.name].metrics["ndcg"]


def test_popularity_ranker_ignores_the_query(small_catalog):
    ranker = PopularityRanker(small_catalog.popularity())
    first = ranker.rank([1, 2], 10, set())
    second = ranker.rank([50, 60], 10, set())
    assert np.array_equal(first, second)


def test_rankers_never_return_excluded_indices(small_catalog, scaled_features, model):
    exclude = {0, 1, 2, 3, 4}
    for ranker in (
        RandomRanker(len(small_catalog), random_state=0),
        PopularityRanker(small_catalog.popularity()),
        RawFeatureRanker(scaled_features),
        LatentRanker(model, small_catalog),
    ):
        ranked = ranker.rank([10, 11], 20, exclude)
        assert not (set(int(i) for i in ranked) & exclude), ranker.name


def test_compare_rankers_sorts_by_ndcg(small_catalog, scaled_features, model):
    groups = build_groups(small_catalog, seed_size=3, min_group_size=6,
                          max_groups=40, random_state=0)
    results = compare_rankers(
        [RandomRanker(len(small_catalog), 0), LatentRanker(model, small_catalog)],
        groups, small_catalog, scaled_features, k=10,
    )
    ndcgs = [r.metrics["ndcg"] for r in results]
    assert ndcgs == sorted(ndcgs, reverse=True)

    table = format_comparison(results, k=10)
    assert "ndcg@10" in table
    assert len(table.splitlines()) == len(results) + 2


def test_evaluate_requires_groups(small_catalog, scaled_features, model):
    with pytest.raises(ValueError, match="no evaluation groups"):
        evaluate_ranker(LatentRanker(model, small_catalog), [], small_catalog, scaled_features)
