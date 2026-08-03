"""Tests for clustering, taxonomy agreement, and evaluation statistics."""

from __future__ import annotations

import numpy as np
import pytest

from eigengrooves import fit_latent_model
from eigengrooves.baselines import LatentRanker, RandomRanker, RawFeatureRanker
from eigengrooves.cluster import (
    adjusted_rand_index,
    compare_to_labels,
    confusion_table,
    kmeans,
    normalized_mutual_information,
    purity,
    silhouette_score,
)
from eigengrooves.evaluate import (
    bootstrap_ci,
    build_groups,
    evaluate_ranker,
    paired_bootstrap_test,
)


# ---------------------------------------------------------------------------
# k-means
# ---------------------------------------------------------------------------


def _blobs(rng, n_per=60, separation=12.0):
    """Three well-separated Gaussian blobs with known membership."""
    centres = np.array([[0.0, 0.0], [separation, 0.0], [0.0, separation]])
    points = np.vstack([c + rng.normal(scale=0.6, size=(n_per, 2)) for c in centres])
    truth = np.repeat([0, 1, 2], n_per)
    return points, truth


def test_kmeans_recovers_well_separated_blobs(rng):
    X, truth = _blobs(rng)
    result = kmeans(X, k=3, random_state=0)
    # Cluster ids are arbitrary, so compare partitions, not labels.
    assert adjusted_rand_index(result.labels, truth) == pytest.approx(1.0)


def test_kmeans_centroids_land_on_blob_centres(rng):
    X, truth = _blobs(rng)
    result = kmeans(X, k=3, random_state=0)
    expected = np.array([X[truth == i].mean(axis=0) for i in range(3)])
    for centre in expected:
        assert np.min(np.linalg.norm(result.centroids - centre, axis=1)) < 0.5


def test_kmeans_inertia_decreases_with_k(rng):
    X, _ = _blobs(rng)
    inertias = [kmeans(X, k=k, random_state=0).inertia for k in (1, 2, 3, 5)]
    assert inertias == sorted(inertias, reverse=True)


def test_kmeans_is_deterministic_given_a_seed(rng):
    X, _ = _blobs(rng)
    first = kmeans(X, k=3, random_state=42)
    second = kmeans(X, k=3, random_state=42)
    assert np.array_equal(first.labels, second.labels)
    assert first.inertia == pytest.approx(second.inertia)


def test_kmeans_k_equals_n_gives_zero_inertia(rng):
    X = rng.normal(size=(12, 3))
    assert kmeans(X, k=12, random_state=0).inertia == pytest.approx(0.0, abs=1e-9)


def test_kmeans_k_of_one_centres_on_the_mean(rng):
    X = rng.normal(size=(50, 4))
    result = kmeans(X, k=1, random_state=0)
    assert np.allclose(result.centroids[0], X.mean(axis=0))


def test_kmeans_handles_duplicate_points():
    """Degenerate input must not divide by zero during k-means++ seeding."""
    X = np.ones((20, 3))
    result = kmeans(X, k=3, random_state=0)
    assert result.inertia == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.isfinite(result.centroids))


@pytest.mark.parametrize("bad_k", [0, -1, 999])
def test_kmeans_rejects_invalid_k(rng, bad_k):
    with pytest.raises(ValueError, match="k must be"):
        kmeans(rng.normal(size=(10, 2)), k=bad_k)


def test_kmeans_rejects_zero_restarts(rng):
    with pytest.raises(ValueError, match="n_restarts"):
        kmeans(rng.normal(size=(10, 2)), k=2, n_restarts=0)


# ---------------------------------------------------------------------------
# Silhouette
# ---------------------------------------------------------------------------


def test_silhouette_is_high_for_separated_clusters(rng):
    X, truth = _blobs(rng, separation=25.0)
    assert silhouette_score(X, truth) > 0.9


def test_silhouette_is_low_for_random_labels(rng):
    X = rng.normal(size=(300, 4))
    labels = rng.integers(0, 3, size=300)
    assert abs(silhouette_score(X, labels)) < 0.15


def test_silhouette_of_a_single_cluster_is_zero(rng):
    X = rng.normal(size=(50, 3))
    assert silhouette_score(X, np.zeros(50, dtype=int)) == 0.0


def test_silhouette_subsamples_large_inputs(rng):
    X = rng.normal(size=(5000, 3))
    labels = rng.integers(0, 4, size=5000)
    value = silhouette_score(X, labels, max_samples=500)
    assert np.isfinite(value)


def test_silhouette_rejects_mismatched_lengths(rng):
    with pytest.raises(ValueError, match="labels"):
        silhouette_score(rng.normal(size=(10, 2)), np.zeros(5))


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------


def test_ari_of_identical_partitions_is_one():
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert adjusted_rand_index(labels, labels) == pytest.approx(1.0)


def test_ari_is_invariant_to_label_renaming():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([7, 7, 3, 3, 9, 9])
    assert adjusted_rand_index(a, b) == pytest.approx(1.0)


def test_ari_of_independent_partitions_is_near_zero(rng):
    a = rng.integers(0, 5, size=3000)
    b = rng.integers(0, 5, size=3000)
    assert abs(adjusted_rand_index(a, b)) < 0.02


def test_ari_beats_purity_at_resisting_cluster_inflation(rng):
    """Purity can be gamed by using more clusters; ARI cannot.

    This is why the module reports both and warns about the difference.
    """
    truth = rng.integers(0, 4, size=600)
    singletons = np.arange(600)  # every point its own cluster
    assert purity(singletons, truth) == pytest.approx(1.0)
    assert adjusted_rand_index(singletons, truth) < 0.01


def test_nmi_of_identical_partitions_is_one():
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert normalized_mutual_information(labels, labels) == pytest.approx(1.0)


def test_nmi_of_independent_partitions_is_near_zero(rng):
    a = rng.integers(0, 4, size=4000)
    b = rng.integers(0, 4, size=4000)
    assert normalized_mutual_information(a, b) < 0.02


def test_nmi_is_bounded():
    a = np.array([0, 1, 0, 1, 2, 2])
    b = np.array([0, 0, 1, 1, 1, 2])
    assert 0.0 <= normalized_mutual_information(a, b) <= 1.0


def test_agreement_metrics_reject_mismatched_lengths():
    for fn in (adjusted_rand_index, normalized_mutual_information):
        with pytest.raises(ValueError, match="length mismatch"):
            fn(np.array([0, 1]), np.array([0, 1, 2]))


def test_purity_of_perfect_clustering_is_one():
    assert purity(np.array([0, 0, 1, 1]), np.array(["a", "a", "b", "b"])) == pytest.approx(1.0)


def test_confusion_table_shape_and_totals():
    clusters = np.array([0, 0, 1, 1, 1])
    classes = ["x", "y", "y", "y", "z"]
    table, cluster_keys, class_keys = confusion_table(clusters, classes)
    assert table.shape == (2, 3)
    assert table.sum() == 5
    assert cluster_keys == [0, 1]
    assert class_keys == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# Taxonomy comparison end to end
# ---------------------------------------------------------------------------


def test_compare_to_labels_detects_real_structure(small_catalog):
    """The latent space should agree with genre better than chance."""
    model = fit_latent_model(small_catalog.features, small_catalog.feature_names, k=7)
    latent = model.transform(small_catalog.features)
    genres = small_catalog.frame["genre"].astype(str).tolist()

    _, agreement = compare_to_labels(latent, genres, random_state=0)
    assert agreement.adjusted_rand_index > 0.1
    assert agreement.normalized_mutual_information > 0.2
    assert agreement.n_reference_classes == len(set(genres))


def test_compare_to_labels_finds_nothing_in_noise(rng):
    """The control: random latent coordinates must not agree with labels."""
    latent = rng.normal(size=(600, 5))
    labels = rng.choice(["a", "b", "c", "d"], size=600).tolist()
    _, agreement = compare_to_labels(latent, labels, random_state=0)
    assert abs(agreement.adjusted_rand_index) < 0.05


def test_compare_to_labels_defaults_k_to_class_count(small_catalog):
    model = fit_latent_model(small_catalog.features, small_catalog.feature_names, k=5)
    genres = small_catalog.frame["genre"].astype(str).tolist()
    _, agreement = compare_to_labels(
        model.transform(small_catalog.features), genres, random_state=0
    )
    assert agreement.k == agreement.n_reference_classes


def test_compare_to_labels_rejects_length_mismatch(rng):
    with pytest.raises(ValueError, match="labels"):
        compare_to_labels(rng.normal(size=(10, 3)), ["a", "b"])


def test_verdict_is_a_readable_sentence(small_catalog):
    model = fit_latent_model(small_catalog.features, small_catalog.feature_names, k=7)
    _, agreement = compare_to_labels(
        model.transform(small_catalog.features),
        small_catalog.frame["genre"].astype(str).tolist(),
        random_state=0,
    )
    assert isinstance(agreement.verdict(), str)
    assert "taxonomy" in agreement.verdict()


# ---------------------------------------------------------------------------
# Bootstrap statistics
# ---------------------------------------------------------------------------


def test_bootstrap_ci_brackets_the_mean(rng):
    values = rng.normal(loc=0.5, scale=0.1, size=500)
    low, high = bootstrap_ci(values, random_state=0)
    assert low < values.mean() < high


def test_bootstrap_ci_narrows_with_more_data(rng):
    small = bootstrap_ci(rng.normal(size=30), random_state=0)
    large = bootstrap_ci(rng.normal(size=3000), random_state=0)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_bootstrap_ci_of_a_constant_is_degenerate():
    low, high = bootstrap_ci(np.full(50, 0.25), random_state=0)
    assert low == pytest.approx(0.25)
    assert high == pytest.approx(0.25)


def test_bootstrap_ci_handles_empty_and_single():
    assert np.isnan(bootstrap_ci(np.array([]))).all()
    assert bootstrap_ci(np.array([3.0])) == (3.0, 3.0)


def test_paired_test_finds_no_difference_between_identical_systems(
    small_catalog, scaled_features, model
):
    groups = build_groups(small_catalog, seed_size=3, min_group_size=6,
                          max_groups=60, random_state=0)
    ranker = LatentRanker(model, small_catalog)
    a = evaluate_ranker(ranker, groups, small_catalog, scaled_features, k=10)
    b = evaluate_ranker(ranker, groups, small_catalog, scaled_features, k=10)

    test = paired_bootstrap_test(a, b, metric="ndcg", random_state=0)
    assert test.difference == pytest.approx(0.0)
    assert not test.significant
    assert test.p_value > 0.5


def test_paired_test_detects_a_real_difference(small_catalog, scaled_features, model):
    """A working system beating random must register as significant."""
    groups = build_groups(small_catalog, seed_size=3, min_group_size=6,
                          max_groups=120, random_state=0)
    good = evaluate_ranker(
        LatentRanker(model, small_catalog), groups, small_catalog, scaled_features, k=10
    )
    chance = evaluate_ranker(
        RandomRanker(len(small_catalog), random_state=0),
        groups, small_catalog, scaled_features, k=10,
    )

    test = paired_bootstrap_test(good, chance, metric="ndcg", random_state=0)
    assert test.difference > 0
    assert test.significant
    assert test.p_value < 0.05
    assert test.ci_low > 0


def test_paired_test_reports_query_count(small_catalog, scaled_features, model):
    groups = build_groups(small_catalog, seed_size=3, min_group_size=6,
                          max_groups=50, random_state=0)
    a = evaluate_ranker(LatentRanker(model, small_catalog), groups,
                        small_catalog, scaled_features, k=10)
    b = evaluate_ranker(RawFeatureRanker(scaled_features), groups,
                        small_catalog, scaled_features, k=10)
    test = paired_bootstrap_test(a, b, random_state=0)
    assert test.n_queries == len(groups)
    assert isinstance(test.summary(), str)


def test_paired_test_rejects_unknown_metric(small_catalog, scaled_features, model):
    groups = build_groups(small_catalog, seed_size=3, min_group_size=6,
                          max_groups=30, random_state=0)
    scores = evaluate_ranker(LatentRanker(model, small_catalog), groups,
                             small_catalog, scaled_features, k=10)
    with pytest.raises(ValueError, match="not recorded"):
        paired_bootstrap_test(scores, scores, metric="nonsense")


def test_per_query_values_are_recorded(small_catalog, scaled_features, model):
    groups = build_groups(small_catalog, seed_size=3, min_group_size=6,
                          max_groups=40, random_state=0)
    scores = evaluate_ranker(LatentRanker(model, small_catalog), groups,
                             small_catalog, scaled_features, k=10)
    assert scores.per_query["ndcg"].size == len(groups)
    assert scores.metrics["ndcg"] == pytest.approx(scores.per_query["ndcg"].mean())

    low, high = scores.ci("ndcg")
    assert low <= scores.metrics["ndcg"] <= high
