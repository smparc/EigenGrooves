"""Tests for catalogue, scaling, rank selection, model and matching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eigengrooves import Catalog, fit_latent_model, make_synthetic_catalog, make_synthetic_frame
from eigengrooves.catalog import CatalogError
from eigengrooves.matching import (
    levenshtein,
    normalize_title,
    similarity_ratio,
    token_set_ratio,
)
from eigengrooves.normalization import fit_scaler
from eigengrooves.rank import (
    rank_by_elbow,
    rank_by_gavish_donoho,
    rank_by_variance,
    select_rank,
)

# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def test_zscore_produces_zero_mean_unit_variance(rng):
    X = rng.normal(loc=[5.0, -100.0, 0.2], scale=[2.0, 30.0, 0.05], size=(500, 3))
    scaled, _ = fit_scaler(X, method="zscore")
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-12)


def test_scaler_roundtrips(rng):
    X = rng.normal(size=(200, 6)) * 17 + 3
    scaled, scaler = fit_scaler(X)
    assert np.allclose(scaler.inverse_transform(scaled), X, atol=1e-10)


def test_scaler_applies_training_statistics_to_new_points(rng):
    X = rng.normal(size=(200, 4)) * 5
    _, scaler = fit_scaler(X)
    new = np.full(4, 5.0)
    assert np.allclose(scaler.transform(new), (new - scaler.center) / scaler.scale)


def test_constant_feature_does_not_divide_by_zero():
    X = np.column_stack([np.ones(50), np.arange(50.0)])
    scaled, _ = fit_scaler(X)
    assert np.all(np.isfinite(scaled))
    assert np.allclose(scaled[:, 0], 0.0)


def test_robust_scaling_resists_outliers(rng):
    X = rng.normal(size=(500, 1))
    X[0, 0] = 1e6  # single extreme outlier
    z_scaled, _ = fit_scaler(X, method="zscore")
    robust_scaled, _ = fit_scaler(X, method="robust")
    # Under z-scoring one outlier compresses everything else toward zero.
    assert robust_scaled[1:].std() > z_scaled[1:].std() * 10


def test_scaler_rejects_non_finite(rng):
    X = rng.normal(size=(10, 3))
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        fit_scaler(X)


# ---------------------------------------------------------------------------
# Rank selection
# ---------------------------------------------------------------------------


def test_variance_threshold_picks_the_smallest_sufficient_rank():
    sigma = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
    energy = sigma**2
    cumulative = np.cumsum(energy) / energy.sum()
    k = rank_by_variance(sigma, 0.90)
    assert cumulative[k - 1] >= 0.90
    assert k == 1 or cumulative[k - 2] < 0.90


def test_variance_threshold_of_one_keeps_everything():
    sigma = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert rank_by_variance(sigma, 1.0) == 5


def test_elbow_finds_an_obvious_knee():
    # Three dominant components, then a flat tail.
    sigma = np.array([100.0, 90.0, 80.0, 4.0, 3.5, 3.0, 2.5, 2.0])
    assert rank_by_elbow(sigma) == 3


def test_gavish_donoho_recovers_a_planted_rank(rng):
    """The threshold should find low-rank signal buried in Gaussian noise."""
    m, n, true_rank = 600, 30, 5
    signal = rng.normal(size=(m, true_rank)) @ rng.normal(size=(true_rank, n)) * 6.0
    noise = rng.normal(size=(m, n))
    sigma = np.linalg.svd(signal + noise, compute_uv=False)

    assert rank_by_gavish_donoho(sigma, m, n) == true_rank


def test_gavish_donoho_with_known_noise(rng):
    m, n, true_rank = 500, 20, 4
    signal = rng.normal(size=(m, true_rank)) @ rng.normal(size=(true_rank, n)) * 8.0
    sigma = np.linalg.svd(signal + rng.normal(size=(m, n)), compute_uv=False)
    assert rank_by_gavish_donoho(sigma, m, n, noise_sigma=1.0) == true_rank


def test_select_rank_accepts_an_explicit_integer():
    sigma = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    selection = select_rank(sigma, strategy=3)
    assert selection.k == 3
    assert selection.strategy == "explicit"


def test_select_rank_clamps_an_oversized_request():
    selection = select_rank(np.array([5.0, 4.0]), strategy=99)
    assert selection.k == 2
    assert "clamped" in selection.detail


def test_select_rank_reports_cumulative_variance():
    sigma = np.array([4.0, 3.0])
    selection = select_rank(sigma, strategy=1)
    assert selection.cumulative_variance == pytest.approx(16.0 / 25.0)


def test_select_rank_rejects_ascending_spectrum():
    with pytest.raises(ValueError, match="descending"):
        select_rank(np.array([1.0, 5.0]), strategy="variance")


def test_select_rank_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="unknown strategy"):
        select_rank(np.array([2.0, 1.0]), strategy="vibes")


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_synthetic_catalog_is_deduplicated():
    catalog = make_synthetic_catalog(n_songs=400, n_artists=40, random_state=1)
    keys = list(zip(catalog.titles, catalog.artists))
    assert len(keys) == len(set(keys))


def test_synthetic_frame_contains_duplicates_before_dedup():
    """The generator must exercise the dedup path, or it tests nothing."""
    frame = make_synthetic_frame(n_songs=400, n_artists=40, duplicate_rate=0.5, random_state=1)
    assert frame.duplicated(subset=["track_name", "artist_names"]).any()


def test_catalog_index_alignment(small_catalog):
    """Positional and label indexing must agree, or every lookup is a bug."""
    assert list(small_catalog.frame.index) == list(range(len(small_catalog)))
    assert small_catalog.features.shape[0] == len(small_catalog)
    for i in (0, len(small_catalog) // 2, len(small_catalog) - 1):
        assert small_catalog.metadata(i)["track_name"] == small_catalog.titles[i]


def test_catalog_rejects_missing_feature_columns():
    frame = pd.DataFrame({"track_name": ["a"], "artist_names": ["b"], "energy": [0.5]})
    with pytest.raises(CatalogError, match="missing required audio features"):
        Catalog.from_frame(frame)


def test_catalog_rejects_missing_title_column():
    frame = pd.DataFrame({"whatever": ["a"]})
    with pytest.raises(CatalogError, match="no track title column"):
        Catalog.from_frame(frame)


def test_catalog_accepts_column_aliases():
    base = make_synthetic_frame(n_songs=40, n_artists=8, duplicate_rate=0, random_state=2)
    renamed = base.rename(columns={"track_name": "Song", "artist_names": "Artist"})
    catalog = Catalog.from_frame(renamed)
    assert len(catalog) == 40


def test_catalog_drops_rows_with_unparseable_features():
    frame = make_synthetic_frame(n_songs=50, n_artists=10, duplicate_rate=0, random_state=2)
    # Cast first: pandas refuses to store a string in a float column in place.
    frame["energy"] = frame["energy"].astype(object)
    frame.loc[0, "energy"] = "not a number"
    catalog = Catalog.from_frame(frame)
    assert len(catalog) == 49
    assert np.all(np.isfinite(catalog.features))


def test_catalog_missing_file_message_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="--synthetic"):
        Catalog.from_csv(tmp_path / "nope.csv")


def test_popularity_is_normalised(small_catalog):
    popularity = small_catalog.popularity()
    assert popularity.shape == (len(small_catalog),)
    assert popularity.min() >= 0.0
    assert popularity.max() <= 1.0


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [("", "", 0), ("abc", "abc", 0), ("abc", "", 3), ("kitten", "sitting", 3),
     ("flaw", "lawn", 2), ("a", "b", 1)],
)
def test_levenshtein_known_distances(a, b, expected):
    assert levenshtein(a, b) == expected


def test_levenshtein_is_symmetric():
    assert levenshtein("saturn", "satin") == levenshtein("satin", "saturn")


def test_levenshtein_early_exit_is_consistent():
    a, b = "kill bill", "chill grill"
    true_distance = levenshtein(a, b)
    capped = levenshtein(a, b, max_distance=2)
    assert capped > 2
    assert true_distance > 2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Kill Bill", "kill bill"),
        ("  KILL BILL  ", "kill bill"),
        ("Kill Bill (feat. Doja Cat)", "kill bill"),
        ("Kill Bill - Remastered 2023", "kill bill"),
        ("Café Naïve", "cafe naive"),
        ("Don't Stop!", "don t stop"),
    ],
)
def test_normalize_title(raw, expected):
    assert normalize_title(raw) == expected


def test_token_set_ratio_rewards_containment():
    assert token_set_ratio("saturn", "saturn deluxe edition") == pytest.approx(1.0)


def test_similarity_ratio_bounds():
    assert similarity_ratio("abc", "abc") == pytest.approx(1.0)
    assert 0.0 <= similarity_ratio("abc", "xyz") <= 1.0


def test_catalog_find_handles_typos_and_decoration(small_catalog):
    title = small_catalog.titles[10]
    artist = small_catalog.artists[10]

    assert small_catalog.find(title)[0].index == 10
    assert small_catalog.find(f"  {title.upper()}  ")[0].index == 10
    assert small_catalog.find(f"{title} (feat. Nobody)")[0].index == 10
    assert small_catalog.find(f"{title} - {artist}")[0].index == 10


def test_exact_matching_rejects_typos(small_catalog):
    title = small_catalog.titles[3]
    assert small_catalog.find(title, fuzzy=False)
    assert not small_catalog.find(title + "zzz", fuzzy=False)


def test_resolve_playlist_reports_unresolved(small_catalog):
    queries = [small_catalog.titles[0], "definitely not a real song xyzzy"]
    indices, resolved, unresolved = small_catalog.resolve_playlist(queries)
    assert indices == [0]
    assert len(resolved) == 1
    assert unresolved == ["definitely not a real song xyzzy"]


def test_resolve_playlist_deduplicates_seeds(small_catalog):
    title = small_catalog.titles[0]
    indices, _, _ = small_catalog.resolve_playlist([title, title, title])
    assert indices == [0]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def test_model_transform_shapes(model, small_catalog):
    latent = model.transform(small_catalog.features)
    assert latent.shape == (len(small_catalog), model.k)
    single = model.transform(small_catalog.features[0])
    assert single.shape == (model.k,)
    assert np.allclose(single, latent[0])


def test_whitening_equalises_latent_axes(small_catalog):
    """Without whitening LF1 dominates; with it, axes carry comparable scale."""
    plain = fit_latent_model(small_catalog.features, small_catalog.feature_names, k=5)
    whitened = plain.with_whiten(True)

    plain_variance = plain.transform(small_catalog.features).var(axis=0)
    whitened_variance = whitened.transform(small_catalog.features).var(axis=0)

    plain_spread = plain_variance.max() / plain_variance.min()
    whitened_spread = whitened_variance.max() / whitened_variance.min()

    # Whitening divides each axis by its singular value, so the retained axes
    # end up with essentially identical variance.
    assert whitened_spread == pytest.approx(1.0, abs=1e-6)
    assert plain_spread > 2.0
    assert whitened_spread < plain_spread


def test_inverse_transform_recovers_full_rank_input(small_catalog):
    model = fit_latent_model(small_catalog.features, small_catalog.feature_names, k=9)
    recovered = model.inverse_transform(model.transform(small_catalog.features))
    assert np.allclose(recovered, small_catalog.features, atol=1e-8)


@pytest.mark.parametrize("whiten", [False, True])
def test_inverse_transform_roundtrips_under_whitening(small_catalog, whiten):
    model = fit_latent_model(
        small_catalog.features, small_catalog.feature_names, k=9, whiten=whiten
    )
    recovered = model.inverse_transform(model.transform(small_catalog.features))
    assert np.allclose(recovered, small_catalog.features, atol=1e-8)


def test_loadings_are_sorted_by_magnitude(model):
    weights = [abs(w) for _, w in model.loadings(1)]
    assert weights == sorted(weights, reverse=True)


def test_loadings_reject_out_of_range_component(model):
    with pytest.raises(IndexError):
        model.loadings(0)
    with pytest.raises(IndexError):
        model.loadings(model.k + 1)


def test_model_save_load_roundtrip(model, small_catalog, tmp_path):
    path = tmp_path / "model.npz"
    model.save(path)
    from eigengrooves.model import LatentModel

    loaded = LatentModel.load(path)
    assert loaded.k == model.k
    assert loaded.whiten == model.whiten
    assert loaded.feature_names == model.feature_names
    assert np.allclose(loaded.components, model.components)
    assert np.allclose(
        loaded.transform(small_catalog.features), model.transform(small_catalog.features)
    )


def test_model_requires_matching_feature_count(small_catalog):
    with pytest.raises(ValueError, match="feature columns"):
        fit_latent_model(small_catalog.features, ("too", "few"))


def test_model_rejects_tiny_catalogue():
    with pytest.raises(ValueError, match="at least 2 songs"):
        fit_latent_model(np.zeros((1, 9)), tuple("abcdefghi"))


@pytest.mark.parametrize("strategy", ["variance", "elbow", "gavish_donoho"])
def test_rank_strategies_all_produce_usable_models(small_catalog, strategy):
    model = fit_latent_model(
        small_catalog.features, small_catalog.feature_names, k=strategy
    )
    assert 1 <= model.k <= 9
    assert model.rank_selection.strategy == strategy
