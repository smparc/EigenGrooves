"""
Why a song was recommended.

This is the part the original project was one step away from and never took.
It already computed the latent basis and printed the loadings; what it never
did was connect a *specific recommendation* back to them. So the output was a
list of song titles and a similarity score, which asks the reader to take the
mathematics on faith.

The decomposition is exact and cheap. For unit-normalised latent vectors
``q`` and ``r``, cosine similarity is just their dot product::

    cos(q, r) = sum_i  q_i * r_i

so the term ``q_i * r_i`` is precisely how much latent component ``i``
contributed to the match, and the terms sum to the score. Pair each component
with its strongest audio-feature loadings and the result is a sentence:
*"matched on the energy-loudness axis, which both tracks score high on."*
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import LatentModel

__all__ = ["ComponentContribution", "Explanation", "explain_match"]


@dataclass(frozen=True)
class ComponentContribution:
    """One latent component's share of a similarity score."""

    component: int
    contribution: float
    query_coordinate: float
    candidate_coordinate: float
    description: str

    @property
    def agrees(self) -> bool:
        """True when both tracks sit on the same side of this axis."""
        return self.contribution >= 0

    def phrase(self) -> str:
        direction = "high" if self.query_coordinate >= 0 else "low"
        if self.agrees:
            return f"both score {direction} on LF{self.component} ({self.description})"
        return f"they diverge on LF{self.component} ({self.description})"


@dataclass(frozen=True)
class Explanation:
    """A decomposition of one similarity score into component contributions."""

    score: float
    contributions: tuple[ComponentContribution, ...]

    def summary(self, top_n: int = 2) -> str:
        """A one-line natural-language gloss."""
        positive = [c for c in self.contributions if c.agrees][:top_n]
        if not positive:
            return "matched weakly; no single latent axis dominates"
        return "; ".join(c.phrase() for c in positive)

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "contributions": [
                {
                    "component": c.component,
                    "contribution": c.contribution,
                    "query_coordinate": c.query_coordinate,
                    "candidate_coordinate": c.candidate_coordinate,
                    "description": c.description,
                }
                for c in self.contributions
            ],
            "summary": self.summary(),
        }


def explain_match(
    query_latent: np.ndarray,
    candidate_latent: np.ndarray,
    model: LatentModel,
    top_n: int = 3,
) -> Explanation:
    """Decompose the cosine similarity between two latent vectors.

    Parameters
    ----------
    query_latent, candidate_latent : np.ndarray, shape (k,)
    model : LatentModel
        Supplies the loadings used to describe each component.
    top_n : int
        How many components to report, ranked by absolute contribution.

    Returns
    -------
    Explanation
        ``contributions`` sums to ``score`` across *all* components; the
        reported subset is the largest ``top_n`` by magnitude.
    """
    q = np.asarray(query_latent, dtype=float).ravel()
    r = np.asarray(candidate_latent, dtype=float).ravel()
    if q.size != r.size:
        raise ValueError(f"dimension mismatch: {q.size} vs {r.size}")

    nq = float(np.sqrt(q @ q))
    nr = float(np.sqrt(r @ r))
    if nq < 1e-12 or nr < 1e-12:
        return Explanation(score=0.0, contributions=())

    q_hat = q / nq
    r_hat = r / nr
    terms = q_hat * r_hat
    score = float(np.clip(terms.sum(), -1.0, 1.0))

    order = np.argsort(-np.abs(terms))[: max(top_n, 0)]
    contributions = tuple(
        ComponentContribution(
            component=int(i) + 1,
            contribution=float(terms[i]),
            query_coordinate=float(q_hat[i]),
            candidate_coordinate=float(r_hat[i]),
            description=model.describe_component(int(i) + 1, top_n=2),
        )
        for i in order
    )
    return Explanation(score=score, contributions=contributions)
