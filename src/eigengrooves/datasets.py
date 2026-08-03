"""
Fetching the reference dataset.

The repository gitignores ``data/*.csv`` (correctly -- the dataset is not ours
to redistribute), but the original version shipped no way to obtain it, so a
fresh clone could not run at all. This module closes that gap.

If the download is unavailable for any reason, the fallback is not an error
message: it is :func:`eigengrooves.synthetic.make_synthetic_catalog`, which
produces a catalogue with the same shape and statistical structure. Every
entry point accepts ``--synthetic``.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

__all__ = ["DATASET_URL", "REQUIRED_COLUMNS", "fetch", "sha256_of"]

DATASET_URL = (
    "https://raw.githubusercontent.com/JulianoOrlandi/"
    "Spotify_Top_Songs_and_Audio_Features/main/spotify_top_songs_audio_features.csv"
)

REQUIRED_COLUMNS = (
    "track_name",
    "artist_names",
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


def sha256_of(path: Path) -> str:
    """SHA-256 of a file, streamed so large datasets do not land in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(path: Path) -> tuple[bool, str]:
    """Check that a downloaded file looks like the catalogue we expect."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            header = handle.readline().strip()
    except OSError as exc:
        return False, f"could not read {path}: {exc}"

    if not header:
        return False, "file is empty"

    columns = {
        c.strip().strip('"').lower().replace(" ", "_") for c in header.split(",")
    }
    # Accept the common aliases the Catalog loader also accepts.
    aliases = {"track": "track_name", "song": "track_name", "artist": "artist_names",
               "artists": "artist_names", "artist_name": "artist_names"}
    columns |= {aliases[c] for c in columns if c in aliases}

    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        return False, f"missing expected columns: {missing}"
    return True, "ok"


def fetch(destination: Path | str, url: str = DATASET_URL, force: bool = False) -> int:
    """Download the dataset to ``destination``.

    Downloads to a temporary path and only moves it into place after the header
    validates, so a failed or truncated download cannot leave a half-written
    file that later looks like a parsing bug.

    Returns
    -------
    int
        Process exit code: 0 on success, non-zero on failure.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        ok, reason = _validate(destination)
        if ok:
            print(f"Dataset already present: {destination}")
            print(f"  sha256: {sha256_of(destination)}")
            return 0
        print(f"Existing file at {destination} looks wrong ({reason}); re-downloading.")

    temporary = destination.with_suffix(destination.suffix + ".partial")
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(temporary, "wb") as out:
            shutil.copyfileobj(response, out)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        print(
            "\nThe dataset is optional. Everything runs against the built-in\n"
            "synthetic catalogue instead:\n"
            "    eigengrooves recommend --synthetic\n"
            "    eigengrooves evaluate  --synthetic\n",
            file=sys.stderr,
        )
        return 1

    ok, reason = _validate(temporary)
    if not ok:
        temporary.unlink(missing_ok=True)
        print(f"\nDownloaded file rejected: {reason}", file=sys.stderr)
        print("The upstream dataset layout may have changed.", file=sys.stderr)
        return 1

    temporary.replace(destination)
    print(f"Saved to {destination}")
    print(f"  size  : {destination.stat().st_size / 1e6:.1f} MB")
    print(f"  sha256: {sha256_of(destination)}")
    print("\nRecord that checksum if you need reproducible results -- the upstream")
    print("file is not versioned and can change without notice.")
    return 0
