"""
Obtaining a catalogue.

About the referenced dataset
----------------------------
The project README cites Orlandi's *Spotify Top Songs and Audio Features* as
its data source. That repository does not host a CSV. It hosts a Jupyter
notebook that *builds* one, and building it requires two things the reader does
not have: a folder of weekly chart exports produced by a separate scraper
project, and personal Spotify API credentials.

So there is no URL to download, and a fetch script that pretends otherwise
would just fail confusingly. This module therefore does three honest things:

1. ``fetch`` accepts an explicit ``--url`` for a CSV you have located yourself,
   validates its header before installing it, and reports a checksum.
2. ``describe_sources`` explains how to obtain or build a real catalogue.
3. Failing both, every entry point runs against
   :func:`eigengrooves.synthetic.make_synthetic_catalog`.

Any CSV works as long as it carries :data:`REQUIRED_COLUMNS`; the loader in
:mod:`eigengrooves.catalog` accepts several common column-naming variants.

One caveat worth knowing before you invest effort in rebuilding it: Spotify
restricted access to the ``audio-features`` endpoint for newly-created API
applications in late 2024. Older credentials may still work, but a fresh app
likely cannot retrieve the nine features this project is built on. Verify
current API access before starting.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

__all__ = [
    "REQUIRED_COLUMNS",
    "UPSTREAM_PROJECT",
    "describe_sources",
    "fetch",
    "sha256_of",
]

UPSTREAM_PROJECT = (
    "https://github.com/JulianoOrlandi/Spotify_Top_Songs_and_Audio_Features"
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


def describe_sources() -> str:
    """Human-readable guidance on getting a real catalogue."""
    return f"""
No catalogue CSV is bundled with this repository, and none can be downloaded
automatically -- the cited upstream project ships a *builder notebook*, not a
dataset.

Three ways forward:

  1. Run without one. Everything works against a generated catalogue:
         eigengrooves recommend --synthetic
         eigengrooves evaluate  --synthetic

  2. Supply your own CSV. Any file with these columns will load:
         {", ".join(REQUIRED_COLUMNS)}
     Common aliases (track/song/title, artist/artists) are accepted.
         eigengrooves recommend --data path/to/songs.csv

  3. Build the referenced dataset yourself, via
         {UPSTREAM_PROJECT}
     This needs weekly chart exports from its companion scraper project plus
     Spotify API credentials. Note that Spotify restricted the audio-features
     endpoint for new API applications in late 2024; confirm you still have
     access before committing to this path.
""".strip()


def sha256_of(path: Path) -> str:
    """SHA-256 of a file, streamed so large datasets do not land in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_csv(path: Path) -> tuple[bool, str]:
    """Check that a file's header looks like a song catalogue."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            header = handle.readline().strip()
    except OSError as exc:
        return False, f"could not read {path}: {exc}"

    if not header:
        return False, "file is empty"

    columns = {c.strip().strip('"').lower().replace(" ", "_") for c in header.split(",")}
    aliases = {
        "track": "track_name", "song": "track_name", "title": "track_name",
        "artist": "artist_names", "artists": "artist_names", "artist_name": "artist_names",
    }
    columns |= {aliases[c] for c in columns if c in aliases}

    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        return False, f"missing expected columns: {missing}"
    return True, "ok"


def fetch(
    destination: Path | str,
    url: str | None = None,
    force: bool = False,
) -> int:
    """Install a catalogue CSV at ``destination``.

    Downloads to a temporary path and only moves it into place after the header
    validates, so a failed or truncated transfer cannot leave a half-written
    file that later looks like a parsing bug.

    Parameters
    ----------
    destination : Path | str
    url : str, optional
        Direct link to a CSV. Required -- there is no default, because no
        canonical download exists (see the module docstring).
    force : bool
        Re-download even if a valid file is already present.

    Returns
    -------
    int
        Process exit code: 0 on success, non-zero on failure.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        ok, reason = validate_csv(destination)
        if ok:
            print(f"Dataset already present: {destination}")
            print(f"  sha256: {sha256_of(destination)}")
            return 0
        print(f"Existing file at {destination} looks wrong ({reason}).")

    if not url:
        print(describe_sources())
        return 1

    temporary = destination.with_suffix(destination.suffix + ".partial")
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(temporary, "wb") as out:
            shutil.copyfileobj(response, out)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        print(f"\nDownload failed: {exc}\n", file=sys.stderr)
        print(describe_sources(), file=sys.stderr)
        return 1

    ok, reason = validate_csv(temporary)
    if not ok:
        temporary.unlink(missing_ok=True)
        print(f"\nDownloaded file rejected: {reason}\n", file=sys.stderr)
        print(describe_sources(), file=sys.stderr)
        return 1

    temporary.replace(destination)
    print(f"Saved to {destination}")
    print(f"  size  : {destination.stat().st_size / 1e6:.1f} MB")
    print(f"  sha256: {sha256_of(destination)}")
    print("\nRecord that checksum if you need reproducible results.")
    return 0
