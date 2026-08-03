"""
Fuzzy track lookup.

The original ``find_playlist_indices`` matched titles by exact lowercase
equality, so ``"Kill Bill"`` worked but ``"kill bill "``, ``"Kill Bill (feat.
Doja Cat)"`` and ``"Kil Bill"`` all silently vanished from the playlist. Since
a missing track just prints a warning and continues, a typo quietly changes
your results rather than failing.

Implemented from scratch, in keeping with the rest of the project: Levenshtein
distance with the two-row optimisation, plus a token-set score that handles the
"same song, extra parenthetical" case that edit distance handles badly.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = ["levenshtein", "similarity_ratio", "token_set_ratio", "normalize_title", "Match", "best_matches"]

# Strip the noise that streaming catalogues attach to titles.
_PAREN = re.compile(r"[\(\[].*?[\)\]]")
_FEAT = re.compile(r"\b(feat|ft|featuring|with)\b.*", re.IGNORECASE)
_SUFFIX = re.compile(
    r"\s*-\s*(remaster(ed)?|radio edit|single version|album version|live|"
    r"bonus track|deluxe|explicit|clean)\b.*",
    re.IGNORECASE,
)
_NONWORD = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize_title(text: str) -> str:
    """Fold a title to a comparable form.

    Lowercases, strips accents, removes parentheticals, ``feat.`` clauses and
    remaster/edit suffixes, drops punctuation and collapses whitespace.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    text = _FEAT.sub("", text)
    text = _SUFFIX.sub("", text)
    text = _PAREN.sub("", text)
    text = _NONWORD.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def levenshtein(a: str, b: str, max_distance: int | None = None) -> int:
    """Edit distance between two strings.

    Uses the two-row dynamic-programming formulation: the full ``len(a) x
    len(b)`` table is never materialised, since each row depends only on its
    predecessor. ``max_distance`` enables early exit once every cell in a row
    exceeds the cap, which matters when screening one query against thousands
    of candidates.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a

    if max_distance is not None and len(a) - len(b) > max_distance:
        return max_distance + 1

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,          # deletion
                    current[j - 1] + 1,       # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        if max_distance is not None and min(current) > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def similarity_ratio(a: str, b: str) -> float:
    """Edit-distance similarity in [0, 1]."""
    if not a and not b:
        return 1.0
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / longest


def token_set_ratio(a: str, b: str) -> float:
    """Jaccard overlap of word tokens.

    Complements edit distance: ``"saturn"`` vs ``"saturn deluxe edition"`` is a
    poor edit-distance match but a perfect subset match, and titles differing
    only by trailing junk are exactly the case we want to forgive.
    """
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    intersection = len(ta & tb)
    # Reward containment: a query that is a full subset of the candidate scores 1.
    return intersection / min(len(ta), len(tb))


@dataclass(frozen=True)
class Match:
    """One candidate result from a lookup."""

    index: int
    title: str
    artist: str
    score: float
    exact: bool


def best_matches(
    query: str,
    titles: list[str],
    artists: list[str] | None = None,
    artist_hint: str | None = None,
    limit: int = 5,
    threshold: float = 0.72,
) -> list[Match]:
    """Rank catalogue entries against a query title.

    Scoring blends edit-distance similarity with token overlap, taking the
    stronger of the two; an ``artist_hint`` (from ``"Title - Artist"`` syntax,
    say) adds a bonus so that covers and same-titled songs disambiguate.

    Returns
    -------
    list[Match]
        Descending by score. Empty if nothing clears ``threshold``.
    """
    q = normalize_title(query)
    if not q:
        return []

    hint = normalize_title(artist_hint) if artist_hint else None
    results: list[Match] = []

    for i, raw_title in enumerate(titles):
        cand = normalize_title(raw_title)
        if not cand:
            continue
        exact = cand == q
        if exact:
            score = 1.0
        else:
            # Cheap length prefilter before paying for the DP table.
            if abs(len(cand) - len(q)) > max(len(q), len(cand)) * 0.5:
                continue
            score = max(similarity_ratio(q, cand), token_set_ratio(q, cand) * 0.95)

        artist = artists[i] if artists is not None and i < len(artists) else ""
        if hint and artist:
            if hint in normalize_title(artist):
                score = min(1.0, score + 0.15)
            else:
                score -= 0.05

        if score >= threshold:
            results.append(
                Match(index=i, title=str(raw_title), artist=str(artist), score=score, exact=exact)
            )

    # Exact matches first, then by score. Stable so catalogue order breaks ties.
    results.sort(key=lambda m: (m.exact, m.score), reverse=True)
    return results[:limit]
