"""
EigenGrooves -- music recommendation by latent audio-feature analysis.

Decomposes a catalogue of songs into latent sonic dimensions with a
from-scratch SVD, then recommends by similarity in that reduced space rather
than by genre tags or listening history.

Quick start
-----------
    from eigengrooves import make_synthetic_catalog, fit_latent_model, Recommender

    catalog = make_synthetic_catalog()
    model = fit_latent_model(catalog.features, catalog.feature_names, k="variance")
    recommender = Recommender(model, catalog)

    seeds, _, _ = catalog.resolve_playlist(["Neon Fever", "Velvet Signal"])
    for item in recommender.recommend(seeds, n=10, strategy="mmr", explain=True):
        print(item.title, "-", item.artist, item.explanation.summary())

Everything numerical -- QR, eigendecomposition, SVD, cosine similarity, edit
distance, ranking metrics -- is implemented directly from the mathematics.
NumPy provides array storage and elementwise operations; its ``linalg``
decompositions appear only in the test suite, as ground truth.
"""

from __future__ import annotations

__version__ = "2.0.0"

from .baselines import (
    LatentRanker,
    PopularityRanker,
    RandomRanker,
    RawFeatureRanker,
    build_standard_rankers,
)
from .catalog import DEFAULT_FEATURES, Catalog, CatalogError
from .evaluate import build_groups, compare_rankers, evaluate_ranker, format_comparison
from .explain import Explanation, explain_match
from .linalg import explained_variance_ratio, svd
from .model import LatentModel, fit_latent_model
from .normalization import Scaler, fit_scaler
from .rank import RankSelection, select_rank
from .recommend import Recommendation, RecommendationResult, Recommender
from .similarity import cosine_similarity, cosine_similarity_matrix, query_similarities
from .synthetic import make_synthetic_catalog, make_synthetic_frame

__all__ = [
    "Catalog",
    "CatalogError",
    "DEFAULT_FEATURES",
    "Explanation",
    "LatentModel",
    "LatentRanker",
    "PopularityRanker",
    "RandomRanker",
    "RankSelection",
    "RawFeatureRanker",
    "Recommendation",
    "RecommendationResult",
    "Recommender",
    "Scaler",
    "__version__",
    "build_groups",
    "build_standard_rankers",
    "compare_rankers",
    "cosine_similarity",
    "cosine_similarity_matrix",
    "evaluate_ranker",
    "explain_match",
    "explained_variance_ratio",
    "fit_latent_model",
    "fit_scaler",
    "format_comparison",
    "make_synthetic_catalog",
    "make_synthetic_frame",
    "query_similarities",
    "select_rank",
    "svd",
]
