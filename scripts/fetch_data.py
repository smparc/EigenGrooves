#!/usr/bin/env python
"""
Install a song-catalogue CSV into ``data/``.

    python scripts/fetch_data.py                      # explains the options
    python scripts/fetch_data.py --url https://.../songs.csv
    python scripts/fetch_data.py --url ... --force

There is no default download URL, because the dataset this project references
is not published as a file -- the upstream repository ships a notebook that
builds one from the Spotify API. Run with no arguments for the full
explanation, or skip the dataset entirely:

    eigengrooves recommend --synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from eigengrooves.datasets import fetch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "spotify_songs.csv",
        help="destination path",
    )
    parser.add_argument("--url", default=None, help="direct link to a catalogue CSV")
    parser.add_argument("--force", action="store_true", help="re-download if present")
    args = parser.parse_args()
    return fetch(args.out, url=args.url, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
