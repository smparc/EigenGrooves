"""
A synthetic song catalogue.

The original repository could not be run after cloning: the dataset is
gitignored, there was no fetch script, and every entry point assumed
``data/spotify_songs.csv`` existed. A reader's first experience was a traceback.

This module generates a catalogue with the same shape and statistical character
as the real thing, so ``eigengrooves recommend --synthetic`` works on a fresh
clone with no network access. It is also what the test suite runs against,
which means tests do not depend on a file nobody can redistribute.

The generator is not noise. Songs are drawn from latent *genre* clusters with
hand-set audio-feature profiles, so the data has genuine low-rank structure for
the SVD to find, realistic inter-feature correlations (energy with loudness,
acousticness against both), and a ground-truth genre label the evaluation
harness can score against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .catalog import Catalog

__all__ = ["GenreProfile", "GENRE_PROFILES", "make_synthetic_frame", "make_synthetic_catalog"]


@dataclass(frozen=True)
class GenreProfile:
    """Centre and spread of each audio feature for one genre.

    Values are in the natural units of the Spotify audio features: most are in
    [0, 1], ``loudness`` is dB (roughly [-60, 0]) and ``tempo`` is BPM.
    """

    name: str
    danceability: tuple[float, float]
    energy: tuple[float, float]
    speechiness: tuple[float, float]
    acousticness: tuple[float, float]
    instrumentalness: tuple[float, float]
    liveness: tuple[float, float]
    valence: tuple[float, float]
    loudness: tuple[float, float]
    tempo: tuple[float, float]


GENRE_PROFILES: tuple[GenreProfile, ...] = (
    GenreProfile("edm",        (0.75, 0.10), (0.88, 0.08), (0.07, 0.04), (0.05, 0.05),
                 (0.35, 0.30), (0.20, 0.12), (0.55, 0.20), (-4.5, 1.8), (128, 6)),
    GenreProfile("hip-hop",    (0.80, 0.09), (0.66, 0.12), (0.28, 0.14), (0.15, 0.13),
                 (0.01, 0.03), (0.17, 0.11), (0.52, 0.21), (-6.0, 2.2), (92, 14)),
    GenreProfile("rnb",        (0.66, 0.11), (0.52, 0.13), (0.10, 0.07), (0.30, 0.20),
                 (0.02, 0.05), (0.14, 0.09), (0.44, 0.20), (-7.5, 2.4), (98, 16)),
    GenreProfile("pop",        (0.68, 0.12), (0.70, 0.13), (0.06, 0.04), (0.20, 0.17),
                 (0.01, 0.03), (0.16, 0.11), (0.62, 0.20), (-5.5, 2.0), (118, 15)),
    GenreProfile("rock",       (0.48, 0.13), (0.79, 0.12), (0.06, 0.04), (0.09, 0.10),
                 (0.06, 0.15), (0.25, 0.18), (0.51, 0.22), (-5.8, 2.3), (126, 20)),
    GenreProfile("folk",       (0.46, 0.13), (0.34, 0.15), (0.04, 0.03), (0.72, 0.18),
                 (0.06, 0.14), (0.15, 0.10), (0.42, 0.21), (-11.0, 3.0), (108, 20)),
    GenreProfile("jazz",       (0.53, 0.14), (0.38, 0.16), (0.06, 0.05), (0.62, 0.22),
                 (0.45, 0.34), (0.16, 0.11), (0.48, 0.22), (-12.5, 3.4), (112, 26)),
    GenreProfile("classical",  (0.28, 0.12), (0.18, 0.13), (0.04, 0.02), (0.92, 0.08),
                 (0.86, 0.20), (0.13, 0.09), (0.24, 0.17), (-18.0, 4.5), (104, 28)),
    GenreProfile("ambient",    (0.32, 0.13), (0.24, 0.13), (0.04, 0.02), (0.78, 0.16),
                 (0.80, 0.22), (0.12, 0.08), (0.28, 0.16), (-16.0, 3.8), (96, 22)),
    GenreProfile("live-set",   (0.60, 0.14), (0.76, 0.12), (0.09, 0.06), (0.16, 0.14),
                 (0.10, 0.20), (0.78, 0.14), (0.55, 0.20), (-6.5, 2.6), (122, 18)),
)

_FEATURE_ORDER = (
    "danceability", "energy", "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "loudness", "tempo",
)

# Features bounded to [0, 1] by the Spotify API; loudness and tempo are not.
_UNIT_INTERVAL = {
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence",
}

_ADJECTIVES = (
    "Silver", "Midnight", "Paper", "Velvet", "Neon", "Golden", "Hollow", "Bitter",
    "Electric", "Quiet", "Crimson", "Glass", "Wild", "Slow", "Broken", "Feral",
    "Lunar", "Salt", "Amber", "Static", "Marble", "Copper", "Restless", "Distant",
    "Pale", "Solar", "Iron", "Tender", "Vacant", "Sunken", "Reckless", "Northern",
)
_NOUNS = (
    "Hours", "Machine", "Lantern", "Weather", "Cathedral", "Riverbed", "Signal",
    "Orchard", "Parade", "Mercy", "Kingdom", "Fever", "Anthem", "Ghost", "Harbour",
    "Summer", "Distance", "Gravity", "Echo", "Chorus", "Alibi", "Window", "Tide",
    "Chapel", "Circuit", "Meridian", "Halo", "Vessel", "Threshold", "Lullaby",
)
_ARTIST_TEMPLATES = (
    "{adj} {noun}",
    "The {noun}",
    "{adj} {noun} Club",
    "{noun} & {noun2}",
    "The {adj} {noun}",
)
_TITLE_TEMPLATES = (
    "{adj} {noun}",
    "{noun}",
    "{adj} {noun} (Reprise)",
    "All the {noun}",
    "{noun} in {adj} Light",
    "No {noun}",
    "{adj} {noun} Blues",
)


def _unique_names(templates, rng, count, used):
    """Draw ``count`` distinct names from the template space.

    Falls back to a numeric suffix only after the combinatorial space is
    genuinely exhausted, which keeps names readable at realistic sizes instead
    of degenerating into 'Bitter Riverbed 1550'.
    """
    names = []
    for _ in range(count):
        for _attempt in range(64):
            template = templates[rng.integers(len(templates))]
            candidate = template.format(
                adj=_ADJECTIVES[rng.integers(len(_ADJECTIVES))],
                noun=_NOUNS[rng.integers(len(_NOUNS))],
                noun2=_NOUNS[rng.integers(len(_NOUNS))],
            )
            if candidate not in used:
                break
        else:
            candidate = f"{candidate} {len(used)}"
        used.add(candidate)
        names.append(candidate)
    return names


def make_synthetic_frame(
    n_songs: int = 3000,
    n_artists: int = 220,
    duplicate_rate: float = 0.45,
    random_state: int | np.random.Generator | None = 20240,
) -> pd.DataFrame:
    """Generate a raw, *undeduplicated* catalogue frame.

    Parameters
    ----------
    n_songs : int
        Number of unique tracks.
    n_artists : int
        Number of artists. Each artist is anchored to one genre, so artist
        identity carries real signal -- which is what the evaluation harness
        uses to build ground-truth playlists.
    duplicate_rate : float
        Fraction of tracks that get extra chart rows, reproducing the
        weekly-chart duplication of the real dataset. Set to 0 for a clean
        catalogue. Kept non-zero by default so that the deduplication path is
        exercised by anything using this generator.
    random_state : int | np.random.Generator | None

    Returns
    -------
    pd.DataFrame
        Columns: track_name, artist_names, genre, popularity, plus the nine
        audio features. Row count exceeds ``n_songs`` when duplicates are on.
    """
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    if n_songs < 1:
        raise ValueError(f"n_songs must be >= 1, got {n_songs}")
    n_artists = max(1, min(n_artists, n_songs))

    # Each artist belongs to a genre and has a persistent stylistic offset, so
    # their tracks cluster tighter than the genre as a whole.
    artist_genre = rng.integers(0, len(GENRE_PROFILES), size=n_artists)
    artist_bias = rng.normal(0.0, 0.35, size=(n_artists, len(_FEATURE_ORDER)))
    used_names: set[str] = set()
    artist_names = _unique_names(_ARTIST_TEMPLATES, rng, n_artists, used_names)

    song_artist = rng.integers(0, n_artists, size=n_songs)
    used_titles: set[str] = set()
    titles = _unique_names(_TITLE_TEMPLATES, rng, n_songs, used_titles)

    rows = []
    for s in range(n_songs):
        a = int(song_artist[s])
        profile = GENRE_PROFILES[int(artist_genre[a])]

        record: dict[str, object] = {}
        for f_idx, feature in enumerate(_FEATURE_ORDER):
            centre, spread = getattr(profile, feature)
            value = rng.normal(centre + artist_bias[a, f_idx] * spread, spread)
            if feature in _UNIT_INTERVAL:
                value = float(np.clip(value, 0.0, 1.0))
            elif feature == "loudness":
                value = float(np.clip(value, -60.0, 0.0))
            elif feature == "tempo":
                value = float(np.clip(value, 40.0, 220.0))
            record[feature] = value

        record["track_name"] = titles[s]
        record["artist_names"] = artist_names[a]
        record["genre"] = profile.name
        # Popularity is long-tailed, as it is in reality.
        record["popularity"] = float(np.clip(rng.beta(1.6, 6.0) * 100, 0, 100))
        rows.append(record)

    frame = pd.DataFrame(rows)

    if duplicate_rate > 0:
        n_dup = int(len(frame) * duplicate_rate)
        if n_dup:
            picks = rng.choice(len(frame), size=n_dup, replace=True)
            extra = frame.iloc[picks].copy()
            # Chart re-entries carry slightly different measured features.
            for feature in _FEATURE_ORDER:
                jitter = rng.normal(0.0, 1e-4, size=len(extra))
                extra[feature] = extra[feature].to_numpy() + jitter
            frame = pd.concat([frame, extra], ignore_index=True)
            frame = frame.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(
                drop=True
            )

    columns = ["track_name", "artist_names", "genre", "popularity", *_FEATURE_ORDER]
    return frame[columns]


def make_synthetic_catalog(
    n_songs: int = 3000,
    n_artists: int = 220,
    duplicate_rate: float = 0.45,
    random_state: int | np.random.Generator | None = 20240,
) -> Catalog:
    """Generate a synthetic frame and wrap it in a deduplicated :class:`Catalog`."""
    frame = make_synthetic_frame(
        n_songs=n_songs,
        n_artists=n_artists,
        duplicate_rate=duplicate_rate,
        random_state=random_state,
    )
    return Catalog.from_frame(frame)
