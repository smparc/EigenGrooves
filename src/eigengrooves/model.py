"""
The latent-space model: scale, decompose, project, persist.

This is the piece the original project spread across ``main.py`` and the
notebook as a sequence of loose calls, which meant the notebook and the CLI
could -- and did -- disagree about what the pipeline was. Bundling it into a
fitted object with explicit state makes the pipeline one thing, and makes
"project a song the model has never seen" expressible at all.

Whitening
---------
``B_k = B V_k`` leaves each latent axis scaled by its singular value, so LF1
carries far more magnitude than LF5. Under cosine similarity that means LF1
dominates the ranking and the trailing components are close to decorative: on
representative data LF1 alone held 45.7% of the latent variance.

Setting ``whiten=True`` divides each axis by its singular value, giving every
retained component equal say. Which is better is an empirical question, not a
matter of taste -- so it is a flag, and ``eigengrooves evaluate`` scores both.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .linalg import explained_variance_ratio, svd
from .normalization import Scaler, fit_scaler
from .rank import RankSelection, select_rank

__all__ = ["LatentModel", "fit_latent_model"]

_FORMAT_VERSION = 2


@dataclass(frozen=True)
class LatentModel:
    """A fitted SVD projection of a feature matrix.

    Attributes
    ----------
    components : np.ndarray, shape (k, n_features)
        ``V_k^T`` -- the latent basis. Row ``i`` is latent feature ``i + 1``
        expressed as a weighting of the original audio features.
    singular_values : np.ndarray, shape (k,)
    full_spectrum : np.ndarray
        Every singular value, retained so explained-variance figures have an
        honest denominator.
    scaler : Scaler
    feature_names : tuple[str, ...]
    whiten : bool
    rank_selection : RankSelection
    backend : str
    """

    components: np.ndarray
    singular_values: np.ndarray
    full_spectrum: np.ndarray
    scaler: Scaler
    feature_names: tuple[str, ...]
    whiten: bool
    rank_selection: RankSelection
    backend: str

    @property
    def k(self) -> int:
        return int(self.components.shape[0])

    @property
    def basis(self) -> np.ndarray:
        """``V_k``, shape (n_features, k) -- the projection matrix."""
        return self.components.T

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        """Project raw (unscaled) feature vectors into latent space.

        Applies the *fitted* scaler, so a new song is placed using the
        catalogue's statistics rather than its own.
        """
        X_raw = np.asarray(X_raw, dtype=float)
        single = X_raw.ndim == 1
        Z = self.scaler.transform(X_raw)
        if single:
            Z = Z[None, :]
        latent = Z @ self.basis
        if self.whiten:
            latent = latent / np.where(
                self.singular_values == 0, 1.0, self.singular_values
            )
        return latent[0] if single else latent

    def transform_scaled(self, Z: np.ndarray) -> np.ndarray:
        """Project already-scaled features into latent space."""
        Z = np.asarray(Z, dtype=float)
        single = Z.ndim == 1
        if single:
            Z = Z[None, :]
        latent = Z @ self.basis
        if self.whiten:
            latent = latent / np.where(
                self.singular_values == 0, 1.0, self.singular_values
            )
        return latent[0] if single else latent

    def inverse_transform(self, latent: np.ndarray) -> np.ndarray:
        """Map latent coordinates back to raw feature units.

        Lossy by construction -- the discarded components are gone -- but it is
        what lets you ask "what does the centre of this playlist sound like?"
        in decibels and BPM instead of in arbitrary latent units.
        """
        latent = np.asarray(latent, dtype=float)
        single = latent.ndim == 1
        if single:
            latent = latent[None, :]
        if self.whiten:
            latent = latent * self.singular_values
        return self.scaler.inverse_transform(latent @ self.components)[0 if single else slice(None)]

    def explained_variance(self) -> np.ndarray:
        """Per-component share of the *total* variance.

        Denominated against the full spectrum, so these sum to less than 1
        whenever components have been discarded. The original implementation
        normalised the truncated spectrum against itself and therefore reported
        100% for every configuration; the true figure for k=5 of 9 was 58.7%.
        """
        return explained_variance_ratio(self.singular_values, self.full_spectrum)

    def loadings(self, component: int) -> list[tuple[str, float]]:
        """Feature weights for one latent component, strongest first.

        ``component`` is 1-based, matching how the components are displayed.
        """
        if not 1 <= component <= self.k:
            raise IndexError(f"component must be in 1..{self.k}, got {component}")
        weights = self.components[component - 1]
        pairs = list(zip(self.feature_names, (float(w) for w in weights)))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        return pairs

    def describe_component(self, component: int, top_n: int = 3) -> str:
        """A short human-readable gloss of a latent component."""
        pairs = self.loadings(component)[:top_n]
        parts = [f"{'+' if w >= 0 else '-'}{abs(w):.2f} {name}" for name, w in pairs]
        return " ".join(parts)

    def with_whiten(self, whiten: bool) -> "LatentModel":
        """A copy with whitening toggled -- no refit needed."""
        return replace(self, whiten=whiten)

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist to ``.npz``.

        Caching the decomposition matters more than it looks: the analysis
        notebook, the CLI and the evaluation harness all otherwise refit the
        same model on the same data.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            format_version=_FORMAT_VERSION,
            components=self.components,
            singular_values=self.singular_values,
            full_spectrum=self.full_spectrum,
            scaler_center=self.scaler.center,
            scaler_scale=self.scaler.scale,
            scaler_method=self.scaler.method,
            feature_names=np.array(self.feature_names, dtype=object),
            whiten=self.whiten,
            backend=self.backend,
            rank_k=self.rank_selection.k,
            rank_strategy=self.rank_selection.strategy,
            rank_detail=self.rank_selection.detail,
            rank_cumvar=self.rank_selection.cumulative_variance,
            allow_pickle=True,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LatentModel":
        """Load a model saved by :meth:`save`."""
        with np.load(Path(path), allow_pickle=True) as data:
            version = int(data["format_version"])
            if version != _FORMAT_VERSION:
                raise ValueError(
                    f"model file has format version {version}, expected {_FORMAT_VERSION}; refit it"
                )
            feature_names = tuple(str(x) for x in data["feature_names"])
            scaler = Scaler(
                center=data["scaler_center"],
                scale=data["scaler_scale"],
                method=str(data["scaler_method"]),
                feature_names=feature_names,
            )
            return cls(
                components=data["components"],
                singular_values=data["singular_values"],
                full_spectrum=data["full_spectrum"],
                scaler=scaler,
                feature_names=feature_names,
                whiten=bool(data["whiten"]),
                rank_selection=RankSelection(
                    k=int(data["rank_k"]),
                    strategy=str(data["rank_strategy"]),
                    detail=str(data["rank_detail"]),
                    cumulative_variance=float(data["rank_cumvar"]),
                ),
                backend=str(data["backend"]),
            )


def fit_latent_model(
    features: np.ndarray,
    feature_names: tuple[str, ...],
    k: int | str = "variance",
    scaling: str = "zscore",
    whiten: bool = False,
    backend: str = "jacobi",
    variance_threshold: float = 0.90,
    random_state: int | None = None,
) -> LatentModel:
    """Fit the scaling + SVD pipeline.

    Parameters
    ----------
    features : np.ndarray, shape (n_songs, n_features)
        Raw, unscaled.
    feature_names : tuple[str, ...]
    k : int | {"variance", "elbow", "gavish_donoho"}
        An explicit rank, or a strategy from :mod:`eigengrooves.rank`.
    scaling : {"zscore", "robust", "none"}
    whiten : bool
    backend : {"jacobi", "eigh", "randomized"}
    variance_threshold : float
        Used when ``k == "variance"``.
    random_state : int, optional
        Only relevant to the randomized backend.

    Returns
    -------
    LatentModel
    """
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError(f"expected a 2-D feature matrix, got shape {features.shape}")
    if features.shape[0] < 2:
        raise ValueError("need at least 2 songs to fit a latent model")
    if features.shape[1] != len(feature_names):
        raise ValueError(
            f"{features.shape[1]} feature columns but {len(feature_names)} names"
        )

    scaled, scaler = fit_scaler(features, method=scaling, feature_names=feature_names)

    # Always compute the full spectrum first. It costs nothing at this width
    # and it is the only way to report explained variance honestly or to let a
    # rank strategy see the shape of the whole scree curve.
    reference_backend = "jacobi" if backend == "randomized" else backend
    _, full_spectrum, _ = svd(scaled, backend=reference_backend)

    selection = select_rank(
        full_spectrum,
        strategy=k,
        n_rows=scaled.shape[0],
        n_cols=scaled.shape[1],
        variance_threshold=variance_threshold,
    )
    if selection.k < 1:
        raise ValueError(
            "rank selection produced k=0; the feature matrix appears to carry no variance"
        )

    _, sigma, Vt = svd(
        scaled, k=selection.k, backend=backend, random_state=random_state
    )

    return LatentModel(
        components=Vt,
        singular_values=sigma,
        full_spectrum=full_spectrum,
        scaler=scaler,
        feature_names=tuple(feature_names),
        whiten=whiten,
        rank_selection=selection,
        backend=backend,
    )
