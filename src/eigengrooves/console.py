"""
Terminal output that does not crash on Windows.

The original ``main.py`` printed U+2713 CHECK MARK, U+2500 BOX DRAWINGS LIGHT
HORIZONTAL, U+03C3 GREEK SMALL LETTER SIGMA and U+2212 MINUS SIGN. On a Windows
console running code page 1252 -- the default on the machine this project was
developed on -- the first of those raises::

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u2713'

so the program died before printing a single recommendation. That is not a
cosmetic bug; it made the entry point unrunnable.

The fix is to route every glyph through :func:`glyph`, which probes what the
active stdout encoding can actually represent and falls back to ASCII when it
cannot. We probe rather than sniff the platform, because the same Windows
machine will happily print UTF-8 under Windows Terminal, under
``PYTHONIOENCODING=utf-8``, or when output is piped somewhere else.
"""

from __future__ import annotations

import os
import sys
from functools import cache

__all__ = ["Console", "glyph", "rule", "supports_unicode"]

# Preferred glyph -> ASCII fallback.
_FALLBACKS = {
    "✓": "+",  # check mark
    "✗": "x",  # ballot x
    "─": "-",  # horizontal rule
    "│": "|",  # vertical bar
    "└": "`",  # up-and-right
    "├": "|",  # tee
    "→": "->",  # rightwards arrow
    "σ": "sigma",
    "−": "-",  # minus sign
    "×": "x",  # multiplication sign
    "█": "#",  # full block
    "░": ".",  # light shade
    "…": "...",
}


@cache
def supports_unicode(encoding: str | None = None) -> bool:
    """Can the active output encoding represent our preferred glyphs?

    ``EIGENGROOVES_ASCII=1`` forces the fallback, which is useful for tests and
    for terminals that accept the bytes but render tofu.
    """
    if os.environ.get("EIGENGROOVES_ASCII", "").strip().lower() in ("1", "true", "yes"):
        return False
    enc = encoding or getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "".join(_FALLBACKS).encode(enc)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def glyph(char: str) -> str:
    """Return ``char`` if the terminal can render it, else an ASCII stand-in."""
    if supports_unicode():
        return char
    return _FALLBACKS.get(char, "?")


def rule(width: int = 62, char: str = "─") -> str:
    """A horizontal rule, encoding-safe."""
    return glyph(char) * width


class Console:
    """Minimal encoding-safe writer.

    Every write goes through an encode/decode round-trip against the real
    stream encoding, so even an unexpected glyph in *data* -- a song title with
    a character cp1252 cannot represent, which is extremely common -- degrades
    to a replacement character instead of taking the process down.
    """

    def __init__(self, stream=None, quiet: bool = False):
        self.stream = stream if stream is not None else sys.stdout
        self.quiet = quiet

    def _sanitize(self, text: str) -> str:
        enc = getattr(self.stream, "encoding", None) or "utf-8"
        try:
            text.encode(enc)
            return text
        except (UnicodeEncodeError, LookupError):
            return text.encode(enc, errors="replace").decode(enc, errors="replace")

    def print(self, text: str = "") -> None:
        if self.quiet:
            return
        self.stream.write(self._sanitize(str(text)) + "\n")

    def header(self, title: str, width: int = 62) -> None:
        self.print("")
        self.print(rule(width))
        self.print(f"  {title}")
        self.print(rule(width))

    def section(self, title: str, width: int = 62) -> None:
        pad = max(width - len(title) - 4, 0)
        self.print("")
        self.print(f"{glyph('─') * 2} {title} {glyph('─') * pad}")

    def ok(self, text: str) -> None:
        self.print(f"  {glyph('✓')} {text}")

    def warn(self, text: str) -> None:
        self.print(f"  {glyph('✗')} {text}")
