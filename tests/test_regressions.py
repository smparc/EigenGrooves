"""
Regression tests for the specific defects found in v1.

Each test here corresponds to a bug that was reproduced before it was fixed.
They are gathered in one file deliberately: this is the list of things that
were once wrong, and the suite exists so they cannot quietly become wrong
again.
"""

from __future__ import annotations

import io
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from eigengrooves import Catalog, Recommender, fit_latent_model
from eigengrooves.console import Console, glyph, supports_unicode
from eigengrooves.linalg import explained_variance_ratio, svd

# ---------------------------------------------------------------------------
# 1. Windows console crash
# ---------------------------------------------------------------------------


def test_glyphs_degrade_to_ascii_when_encoding_cannot_represent_them(monkeypatch):
    """v1 printed U+2713 unconditionally and died on a cp1252 console."""
    monkeypatch.setenv("EIGENGROOVES_ASCII", "1")
    supports_unicode.cache_clear()
    try:
        for char in ("✓", "✗", "─", "σ", "−", "×"):
            rendered = glyph(char)
            assert rendered.encode("ascii")  # must not raise
    finally:
        supports_unicode.cache_clear()


def test_console_survives_unencodable_song_titles():
    """A track title with characters the terminal cannot encode must not crash.

    This is more common than the glyph problem: catalogue data is full of
    accented and CJK titles, and cp1252 cannot represent most of them.
    """
    buffer = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    console = Console(stream=buffer)
    console.print("Bohemian Rhapsody")
    console.print("君の名は — 前前前世")   # unrepresentable in cp1252
    console.print("Épater — Naïve Café")
    buffer.flush()  # must not raise


def test_cli_runs_under_a_legacy_code_page():
    """End-to-end: the entry point must survive a cp1252 stdout.

    This is the exact configuration that made ``python main.py`` unrunnable.
    """
    result = subprocess.run(
        [sys.executable, "-m", "eigengrooves.cli", "recommend",
         "--synthetic", "--synthetic-songs", "300", "-n", "3"],
        capture_output=True,
        env={"PYTHONIOENCODING": "cp1252", "PATH": "", "SYSTEMROOT": ""},
        timeout=300,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in result.stderr


# ---------------------------------------------------------------------------
# 2. Self-recommendation and duplicate flooding
# ---------------------------------------------------------------------------


def _chart_style_frame(n_unique: int = 120, seed: int = 3) -> pd.DataFrame:
    """A catalogue in weekly-chart form: one row per track per chart week."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_unique, 9))
    columns = [
        "danceability", "energy", "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "loudness", "tempo",
    ]
    rows = []
    for i in range(n_unique):
        for _ in range(int(rng.integers(1, 12))):  # 1-11 chart weeks
            row = dict(zip(columns, features[i] + 1e-9 * rng.normal(size=9)))
            row["track_name"] = f"Song {i}"
            row["artist_names"] = f"Artist {i % 20}"
            rows.append(row)
    return pd.DataFrame(rows)


def test_deduplication_collapses_chart_repeats():
    frame = _chart_style_frame()
    catalog = Catalog.from_frame(frame)
    assert len(catalog) == 120
    assert catalog.n_duplicates_removed == len(frame) - 120


def test_deduplication_records_chart_appearances_as_popularity():
    """Collapsing duplicates should create a popularity signal, not lose one."""
    catalog = Catalog.from_frame(_chart_style_frame())
    assert "chart_appearances" in catalog.frame.columns
    popularity = catalog.popularity()
    assert popularity.shape == (len(catalog),)
    assert popularity.min() >= 0.0 and popularity.max() <= 1.0
    assert popularity.std() > 0  # genuinely varies


def test_a_song_is_never_recommended_to_itself():
    """v1 returned the query track back at similarity 1.000000.

    The exclusion set held the single matched row index while the same track's
    other chart rows remained eligible.
    """
    catalog = Catalog.from_frame(_chart_style_frame())
    model = fit_latent_model(catalog.features, catalog.feature_names, k=5)
    recommender = Recommender(model, catalog)

    for seed_index in range(0, len(catalog), 17):
        seed_title = catalog.titles[seed_index]
        for strategy in ("overall_top", "one_per_song", "mmr", "centroid"):
            result = recommender.recommend([seed_index], n=5, strategy=strategy)
            titles = [item.title for item in result]
            assert seed_title not in titles, f"{strategy} recommended the seed back"
            assert seed_index not in [item.index for item in result]


def test_recommendations_contain_no_duplicate_tracks():
    """v1 returned 10 recommendations containing 4 unique titles."""
    catalog = Catalog.from_frame(_chart_style_frame())
    model = fit_latent_model(catalog.features, catalog.feature_names, k=5)
    recommender = Recommender(model, catalog)

    seeds = [0, 1, 2, 3, 4]
    for strategy in ("overall_top", "one_per_song", "mmr", "centroid"):
        result = recommender.recommend(seeds, n=10, strategy=strategy, max_per_artist=None)
        titles = [item.title for item in result]
        assert len(titles) == len(set(titles)), f"{strategy} returned duplicates: {titles}"


def test_seeds_are_excluded_across_all_strategies(recommender):
    seeds = [5, 17, 42, 88]
    for strategy in ("overall_top", "one_per_song", "mmr", "centroid"):
        result = recommender.recommend(seeds, n=15, strategy=strategy)
        assert not (set(item.index for item in result) & set(seeds))


# ---------------------------------------------------------------------------
# 3. Explained variance always reporting 100%
# ---------------------------------------------------------------------------


def test_model_reports_variance_against_the_full_spectrum(small_catalog):
    """v1 printed 'total variance explained: 100.0%' for every configuration."""
    model = fit_latent_model(small_catalog.features, small_catalog.feature_names, k=5)
    retained = model.explained_variance().sum()

    assert model.k == 5
    assert len(model.full_spectrum) == 9
    assert 0.0 < retained < 1.0, "truncated model cannot explain 100% of variance"

    expected = np.sum(model.singular_values**2) / np.sum(model.full_spectrum**2)
    assert retained == pytest.approx(expected)


def test_full_rank_model_does_explain_everything(small_catalog):
    model = fit_latent_model(small_catalog.features, small_catalog.feature_names, k=9)
    assert model.explained_variance().sum() == pytest.approx(1.0)


def test_explained_variance_ratio_denominator_is_explicit():
    sigma = np.array([4.0, 3.0, 2.0, 1.0])
    truncated = sigma[:2]
    assert explained_variance_ratio(truncated).sum() == pytest.approx(1.0)
    assert explained_variance_ratio(truncated, sigma).sum() == pytest.approx(25.0 / 30.0)


# ---------------------------------------------------------------------------
# 4. Broken package imports
# ---------------------------------------------------------------------------


def test_package_imports_cleanly():
    """v1's ``src/__init__.py`` mixed relative and absolute imports.

    ``import src`` raised ``ModuleNotFoundError: No module named 'svd'``, so the
    public API the package advertised could not be imported at all.
    """
    import eigengrooves

    for name in eigengrooves.__all__:
        assert hasattr(eigengrooves, name), f"{name} is exported but missing"


def test_submodules_import_without_path_manipulation():
    for module in (
        "eigengrooves.linalg", "eigengrooves.catalog", "eigengrooves.model",
        "eigengrooves.recommend", "eigengrooves.evaluate", "eigengrooves.cli",
        "eigengrooves.datasets", "eigengrooves.synthetic",
    ):
        __import__(module)


# ---------------------------------------------------------------------------
# 5. Scale-dependent convergence tolerance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_rows", [50, 500, 5000])
def test_decomposition_quality_is_independent_of_catalogue_size(rng, n_rows):
    """v1's fixed 1e-10 off-diagonal budget was unreachable at scale."""
    A = rng.normal(size=(n_rows, 9))
    U, sigma, Vt = svd(A)
    relative_error = np.linalg.norm(A - U @ np.diag(sigma) @ Vt) / np.linalg.norm(A)
    assert relative_error < 1e-13
