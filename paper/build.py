#!/usr/bin/env python
"""
Build the paper: regenerate every figure, then compile the PDF.

    python paper/build.py
    python paper/build.py --data data/spotify_songs.csv
    python paper/build.py --skip-figures        # LaTeX only

Figures and quoted numbers are regenerated first, so the document can never
quote a stale result. LaTeX runs twice to settle cross-references.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
BUILD = PAPER / "build"


def run(command: list[str], cwd: Path) -> int:
    print(f"$ {' '.join(str(c) for c in command)}")
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        # LaTeX buries the actual error in a wall of font paths.
        for line in (result.stdout + result.stderr).splitlines():
            if line.startswith("!") or "Error" in line or "error:" in line:
                print(f"  {line}")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None,
                        help="catalogue CSV; omit for the synthetic catalogue")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--engine", default="pdflatex")
    args = parser.parse_args()

    if not args.skip_figures:
        command = [sys.executable, str(PAPER / "make_figures.py")]
        if args.data:
            command += ["--data", str(args.data)]
        if run(command, cwd=PAPER.parent) != 0:
            print("figure generation failed", file=sys.stderr)
            return 1

    if shutil.which(args.engine) is None:
        print(
            f"\n{args.engine} not found on PATH. Figures are up to date in\n"
            f"  {PAPER / 'figures'}\n"
            "Install a TeX distribution (MiKTeX, TeX Live) to build the PDF.",
            file=sys.stderr,
        )
        return 1

    BUILD.mkdir(exist_ok=True)
    # Twice: the first pass writes the .aux that resolves refs on the second.
    for pass_number in (1, 2):
        code = run(
            [args.engine, "-interaction=nonstopmode",
             f"-output-directory={BUILD}", "main.tex"],
            cwd=PAPER,
        )
        if code != 0 and pass_number == 1:
            print("LaTeX failed; see paper/build/main.log", file=sys.stderr)
            return 1

    log = (BUILD / "main.log").read_text(encoding="utf-8", errors="replace")
    warnings = [
        line for line in log.splitlines()
        if line.startswith("Overfull") or "Undefined" in line or "undefined" in line
    ]
    if warnings:
        print(f"\n{len(warnings)} layout/reference warning(s):")
        for line in warnings[:10]:
            print(f"  {line}")

    pdf = BUILD / "main.pdf"
    if pdf.exists():
        print(f"\nBuilt {pdf}  ({pdf.stat().st_size / 1e6:.1f} MB)")
        return 0
    print("no PDF produced", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
