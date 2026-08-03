"""
The song catalogue: loading, validation, and -- critically -- deduplication.

Why deduplication is not a nicety
---------------------------------
The upstream dataset is a *chart* dataset: one row per track per chart week. A
track that spent eleven weeks on the Top 200 appears eleven times, with eleven
near-identical feature vectors.

The original pipeline fed those rows straight into the recommender, and the
consequences were not subtle. Measured on a simulated weekly-chart catalogue:

* ``overall_top`` returned 10 recommendations containing **4 unique titles**.
* ``one_per_song`` returned 5 containing **3 unique titles**.
* Querying a track returned **that same track** at similarity 1.000000,
  because the exclusion set held the one row index that happened to match while
  the track's other ten rows stayed eligible.

Collapsing to one row per ``(title, artist)`` fixes all three. It also *creates*
information rather than destroying it: the number of collapsed rows is the
number of chart appearances, which is a genuine popularity signal, and the
recommender uses it for novelty weighting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .matching import Match, best_matches, normalize_title

__all__ = ["Catalog", "DEFAULT_FEATURES", "CatalogError"]

DEFAULT_FEATURES: tuple[str, ...] = (
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "loudness",
    "tempo",
)

# Real-world exports disagree about column naming; map the common variants onto
# our canonical names so a plain `from_csv` works on more than one dataset.
_COLUMN_ALIASES: dict[str, str] = {
    "track": "track_name",
    "song": "track_name",
    "name": "track_name",
    "title": "track_name",
    "track_title": "track_name",
    "artist": "artist_names",
    "artists": "artist_names",
    "artist_name": "artist_names",
    "performer": "artist_names",
    "album": "album_name",
    "streams": "popularity_raw",
    "weeks_on_chart": "chart_weeks",
    "peak_rank": "peak_rank",
}


class CatalogError(ValueError):
    """Raised when a dataset cannot be interpreted as a song catalogue."""


@dataclass
class Catalog:
    """A deduplicated set of songs with an aligned numeric feature matrix.

    Attributes
    ----------
    frame : pd.DataFrame
        One row per unique track, index reset to ``0..n-1`` so that positional
        indices into ``features`` and label indices into ``frame`` agree. Every
        other component indexes songs positionally, and silent divergence
        between the two is a whole category of bug this guarantees away.
    features : np.ndarray, shape (n_songs, n_features)
        Raw (unscaled) feature matrix, column order matching ``feature_names``.
    feature_names : tuple[str, ...]
    n_duplicates_removed : int
    """

    frame: pd.DataFrame
    features: np.ndarray
    feature_names: tuple[str, ...] = DEFAULT_FEATURES
    n_duplicates_removed: int = 0
    _title_cache: list[str] = field(default_factory=list, repr=False)
    _artist_cache: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if len(self.frame) != self.features.shape[0]:
            raise CatalogError(
                f"frame has {len(self.frame)} rows but features has "
                f"{self.features.shape[0]}"
            )
        self._title_cache = self.frame["track_name"].astype(str).tolist()
        self._artist_cache = (
            self.frame["artist_names"].astype(str).tolist()
            if "artist_names" in self.frame.columns
            else [""] * len(self.frame)
        )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        feature_names: tuple[str, ...] = DEFAULT_FEATURES,
        deduplicate: bool = True,
    ) -> "Catalog":
        """Build a catalogue from an in-memory frame."""
        df = df.copy()
        df.columns = (
            df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
        )
        df = df.rename(columns={k: v for k, v in _COLUMN_ALIASES.items() if k in df.columns})

        if "track_name" not in df.columns:
            raise CatalogError(
                "dataset has no track title column; expected one of "
                f"'track_name', {sorted(k for k, v in _COLUMN_ALIASES.items() if v == 'track_name')}"
            )
        if "artist_names" not in df.columns:
            df["artist_names"] = "Unknown"

        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise CatalogError(
                f"dataset is missing required audio features: {missing}\n"
                f"available columns: {sorted(df.columns)}"
            )

        # Drop rows with unusable features rather than letting NaN propagate
        # into the decomposition, where it turns the entire matrix to NaN.
        before = len(df)
        df[list(feature_names)] = df[list(feature_names)].apply(
            pd.to_numeric, errors="coerce"
        )
        df = df.dropna(subset=list(feature_names))
        dropped_nan = before - len(df)

        if df.empty:
            raise CatalogError("no rows remain after dropping missing audio features")

        n_before = len(df)
        if deduplicate:
            df = cls._deduplicate(df, feature_names)
        n_removed = n_before - len(df)

        df = df.reset_index(drop=True)
        features = df[list(feature_names)].to_numpy(dtype=float)

        catalog = cls(
            frame=df,
            features=features,
            feature_names=tuple(feature_names),
            n_duplicates_removed=n_removed + dropped_nan,
        )
        return catalog

    @classmethod
    def from_csv(
        cls,
        path: str | os.PathLike,
        feature_names: tuple[str, ...] = DEFAULT_FEATURES,
        deduplicate: bool = True,
    ) -> "Catalog":
        """Load a catalogue from CSV."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{path}'.\n\n"
                "Either fetch the real dataset:\n"
                "    python scripts/fetch_data.py\n"
                "or run against the built-in synthetic catalogue:\n"
                "    eigengrooves recommend --synthetic\n"
            )
        return cls.from_frame(pd.read_csv(path), feature_names, deduplicate)

    @staticmethod
    def _deduplicate(df: pd.DataFrame, feature_names: tuple[str, ...]) -> pd.DataFrame:
        """Collapse to one row per (normalised title, normalised artist).

        Feature values are aggregated by *median* rather than mean: chart
        datasets occasionally carry a mis-joined row for a track, and the median
        ignores it where the mean would not.
        """
        key_title = df["track_name"].map(normalize_title)
        key_artist = df["artist_names"].astype(str).str.lower().str.strip()
        df = df.assign(_key_title=key_title, _key_artist=key_artist)

        # Chart appearances are the natural popularity proxy this dataset
        # offers, and deduplication is the only moment we can count them.
        counts = df.groupby(["_key_title", "_key_artist"], sort=False).size()

        aggregations: dict[str, object] = {c: "median" for c in feature_names}
        for col in df.columns:
            if col in feature_names or col.startswith("_key"):
                continue
            # Keep the first observed value for metadata; take the best rank.
            aggregations[col] = "min" if col == "peak_rank" else "first"

        grouped = (
            df.groupby(["_key_title", "_key_artist"], sort=False)
            .agg(aggregations)
            .reset_index()
        )
        grouped["chart_appearances"] = counts.reset_index(drop=True).to_numpy()
        return grouped.drop(columns=["_key_title", "_key_artist"])

    # -- accessors ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def titles(self) -> list[str]:
        return self._title_cache

    @property
    def artists(self) -> list[str]:
        return self._artist_cache

    def popularity(self) -> np.ndarray:
        """Per-song popularity in [0, 1], or uniform if the data has none.

        Prefers an explicit ``popularity`` column, then raw stream counts, then
        the chart-appearance count recovered during deduplication. Stream
        counts are log-compressed before scaling because their distribution
        spans several orders of magnitude and the raw values would make every
        song but the top few indistinguishable from zero.
        """
        for column, log_scale in (
            ("popularity", False),
            ("popularity_raw", True),
            ("chart_appearances", True),
        ):
            if column in self.frame.columns:
                values = pd.to_numeric(self.frame[column], errors="coerce").to_numpy(dtype=float)
                if np.all(np.isnan(values)):
                    continue
                values = np.nan_to_num(values, nan=0.0)
                if log_scale:
                    values = np.log1p(np.clip(values, 0, None))
                span = values.max() - values.min()
                if span <= 0:
                    continue
                return (values - values.min()) / span
        return np.ones(len(self), dtype=float)

    def metadata(self, index: int) -> dict:
        """Metadata for one song as a plain dict."""
        return self.frame.iloc[int(index)].to_dict()

    def describe(self, index: int) -> str:
        """``'Title - Artist'`` for display."""
        return f"{self.titles[index]} - {self.artists[index]}"

    # -- lookup -------------------------------------------------------------

    def find(
        self,
        query: str,
        artist_hint: str | None = None,
        fuzzy: bool = True,
        limit: int = 5,
    ) -> list[Match]:
        """Look up a track by title, optionally disambiguated by artist.

        Accepts ``"Title - Artist"`` as shorthand when ``artist_hint`` is not
        given separately.
        """
        if artist_hint is None and " - " in query:
            query, artist_hint = query.rsplit(" - ", 1)

        if not fuzzy:
            target = normalize_title(query)
            hits = [
                Match(index=i, title=self.titles[i], artist=self.artists[i], score=1.0, exact=True)
                for i, t in enumerate(self.titles)
                if normalize_title(t) == target
            ]
            if artist_hint:
                hint = artist_hint.lower().strip()
                narrowed = [m for m in hits if hint in m.artist.lower()]
                hits = narrowed or hits
            return hits[:limit]

        return best_matches(
            query,
            self.titles,
            self.artists,
            artist_hint=artist_hint,
            limit=limit,
        )

    def resolve_playlist(
        self, queries: list[str], fuzzy: bool = True
    ) -> tuple[list[int], list[tuple[str, Match]], list[str]]:
        """Resolve titles to catalogue indices.

        Returns
        -------
        indices : list[int]
            Deduplicated, order-preserving.
        resolved : list[tuple[str, Match]]
            Each input query paired with the entry it matched.
        unresolved : list[str]
            Queries with no acceptable match.
        """
        indices: list[int] = []
        resolved: list[tuple[str, Match]] = []
        unresolved: list[str] = []
        seen: set[int] = set()

        for query in queries:
            matches = self.find(query, fuzzy=fuzzy, limit=1)
            if not matches:
                unresolved.append(query)
                continue
            match = matches[0]
            resolved.append((query, match))
            if match.index not in seen:
                seen.add(match.index)
                indices.append(match.index)

        return indices, resolved, unresolved
