#!/usr/bin/env python
"""
Entry point shim.

The real interface is the ``eigengrooves`` console script, installed by
``pip install -e .``. This file exists so that ``python main.py`` keeps working
for anyone who cloned the repository and expects it to, and so it works without
installing anything.

    python main.py recommend --synthetic --explain
    python main.py analyze --synthetic
    python main.py evaluate --synthetic

Running it with no arguments recommends against the synthetic catalogue, which
means a fresh clone produces output rather than a traceback.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Support running from a clone without `pip install -e .`.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from eigengrooves.cli import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        data_present = (Path(__file__).resolve().parent / "data" / "spotify_songs.csv").exists()
        argv = ["recommend", "--explain"]
        if not data_present:
            argv.append("--synthetic")
            print(
                "No dataset found at data/spotify_songs.csv - using the synthetic\n"
                "catalogue. Run `python scripts/fetch_data.py` for the real one.\n"
            )
    raise SystemExit(main(argv))
