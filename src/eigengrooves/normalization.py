"""
Feature scaling.

Audio features arrive on wildly different scales -- ``danceability`` lives in
[0, 1], ``loudness`` is negative decibels, ``tempo`` is in the hundreds. Without
scaling, tempo alone would account for essentially all the variance and the SVD
would return "loud fast songs" as its entire theory of music.

Two scalers are provided. Z-score is the default and matches the original
project. Robust scaling (median / IQR) is available because several audio
features are heavily skewed with real outliers -- ``speechiness`` and
``instrumentalness`` in particular are near-zero for most tracks with a long
tail -- and the mean/std of such a distribution describes almost none of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["METHODS", "Scaler", "fit_scaler"]

METHODS = ("zscore", "robust", "none")


@dataclass(frozen=True)
class Scaler:
    """A fitted, reusable scaling transform.

    Holding the fitted statistics as an object (rather than returning bare
    ``mean``/``std`` arrays) is what makes it possible to project a song the
    model has never seen into the same space -- the statistics have to come
    from the training catalogue, not from the new point.
    """

    center: np.ndarray
    scale: np.ndarray
    method: str
    feature_names: tuple[str, ...] = ()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted transform to a matrix or a single row."""
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        if single:
            X = X[None, :]
        if X.shape[1] != self.center.size:
            raise ValueError(
                f"expected {self.center.size} features, got {X.shape[1]}"
            )
        out = (X - self.center) / self.scale
        return out[0] if single else out

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map back to the original feature units."""
        Z = np.asarray(Z, dtype=float)
        single = Z.ndim == 1
        if single:
            Z = Z[None, :]
        out = Z * self.scale + self.center
        return out[0] if single else out


def fit_scaler(
    X: np.ndarray, method: str = "zscore", feature_names: tuple[str, ...] = ()
) -> tuple[np.ndarray, Scaler]:
    """Fit a scaler and return the transformed matrix alongside it.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    method : {"zscore", "robust", "none"}
    feature_names : tuple[str, ...]
        Carried along for reporting; not used in the arithmetic.

    Returns
    -------
    X_scaled : np.ndarray
    scaler : Scaler
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("feature matrix contains NaN or inf")
    if X.shape[0] == 0:
        raise ValueError("cannot fit a scaler on zero samples")

    if method == "zscore":
        center = X.mean(axis=0)
        scale = X.std(axis=0)
    elif method == "robust":
        center = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        scale = q75 - q25
    elif method == "none":
        center = np.zeros(X.shape[1])
        scale = np.ones(X.shape[1])
    else:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")

    # A zero scale means the feature is constant across the catalogue: it
    # carries no information, so map it to a constant zero rather than dividing
    # by zero and poisoning the whole matrix with NaN.
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)

    scaler = Scaler(
        center=center, scale=scale, method=method, feature_names=tuple(feature_names)
    )
    return scaler.transform(X), scaler
