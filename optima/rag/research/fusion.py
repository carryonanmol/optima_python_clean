"""Reciprocal Rank Fusion with full per-source provenance.

RRF combines *ranks*, not scores, so dense cosine similarities, BM25 scores and
graph orderings (which live on unrelated scales) are never added directly:

    score(d) = sum over lists L containing d of  weight_L / (rrf_k + rank_L(d))

Ranks are 1-based.  Each input list is de-duplicated by stable function ID
(first occurrence wins).  Ties are broken by best (smallest) rank, then by the
position of the first list that contained the item, then by ID, so the output
is deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class RankedList:
    """One ranked candidate list from a single retriever."""

    source: str  # e.g. "dense:raw", "bm25:semantic"
    items: list[tuple[str, float]]  # (function_id, native score), best first
    weight: float = 1.0


@dataclass
class Contribution:
    source: str
    rank: int
    score: float  # native score of the source retriever (not comparable across sources)
    rrf: float  # this source's contribution to the fused score


@dataclass
class FusedItem:
    id: str
    score: float
    contributions: list[Contribution] = field(default_factory=list)

    @property
    def best_rank(self) -> int:
        return min(c.rank for c in self.contributions)


def rrf_fuse(
    lists: Sequence[RankedList], rrf_k: int = 60, limit: int | None = None
) -> list[FusedItem]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")
    fused: dict[str, FusedItem] = {}
    first_list: dict[str, int] = {}
    for list_index, ranked in enumerate(lists):
        if ranked.weight < 0:
            raise ValueError("RRF weights must be >= 0")
        seen = set()
        rank = 0
        for ident, native in ranked.items:
            if ident in seen:
                continue
            seen.add(ident)
            rank += 1
            contribution = ranked.weight / (rrf_k + rank)
            item = fused.get(ident)
            if item is None:
                item = fused[ident] = FusedItem(ident, 0.0)
                first_list[ident] = list_index
            item.score += contribution
            item.contributions.append(
                Contribution(ranked.source, rank, float(native), contribution)
            )
    ordered = sorted(
        fused.values(),
        key=lambda it: (-round(it.score, 12), it.best_rank, first_list[it.id], it.id),
    )
    return ordered[:limit] if limit is not None else ordered
