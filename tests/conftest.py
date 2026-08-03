"""Shared fixtures.

Everything here is built from the synthetic generator, so the suite never
depends on a dataset file that cannot be redistributed.
"""

from __future__ import annotations

import numpy as np
import pytest

from eigengrooves import Catalog, Recommender, fit_latent_model, make_synthetic_catalog
from eigengrooves.normalization import fit_scaler


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture(scope="session")
def small_catalog() -> Catalog:
    """A modest catalogue; fast enough to use in most tests."""
    return make_synthetic_catalog(n_songs=600, n_artists=60, random_state=7)


@pytest.fixture(scope="session")
def model(small_catalog: Catalog):
    return fit_latent_model(
        small_catalog.features, small_catalog.feature_names, k=5, random_state=0
    )


@pytest.fixture(scope="session")
def recommender(model, small_catalog: Catalog) -> Recommender:
    return Recommender(model, small_catalog)


@pytest.fixture(scope="session")
def scaled_features(small_catalog: Catalog) -> np.ndarray:
    scaled, _ = fit_scaler(small_catalog.features, method="zscore")
    return scaled


def well_conditioned(rng: np.random.Generator, m: int = 120, n: int = 9) -> np.ndarray:
    """A generic full-rank matrix."""
    return rng.normal(size=(m, n))


def ill_conditioned(rng: np.random.Generator, m: int = 400, n: int = 9) -> np.ndarray:
    """Strongly correlated columns plus a tiny independent component.

    Mirrors the structure of real audio features, where energy, loudness and
    acousticness are near-collinear. This is the matrix that separates a
    Jacobi SVD from one that goes through ``A^T A``.
    """
    base = rng.normal(size=(m, 4))
    mixing = rng.normal(size=(4, n))
    return base @ mixing + 1e-6 * rng.normal(size=(m, n))
