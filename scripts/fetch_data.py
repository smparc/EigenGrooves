#!/usr/bin/env python
"""
Download the reference Spotify dataset into ``data/``.

    python scripts/fetch_data.py
    python scripts/fetch_data.py --force
    python scripts/fetch_data.py --out /tmp/songs.csv

The dataset is optional: every entry point accepts ``--synthetic`` and will run
against a generated catalogue instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from eigengrooves.datasets import DATASET_URL, fetch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "spotify_songs.csv",
        help="destination path",
    )
    parser.add_argument("--url", default=DATASET_URL, help="source URL")
    parser.add_argument("--force", action="store_true", help="re-download if present")
    args = parser.parse_args()
    return fetch(args.out, url=args.url, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
