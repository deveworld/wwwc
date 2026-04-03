from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log

from .chunking import Chunk
from .config import CacheConfig


@dataclass(slots=True)
class CacheSpan:
    chunk_id: int
    start_token: int
    end_token: int
    score: float
    text: str


def propose_cache_spans(chunks: list[Chunk], config: CacheConfig) -> list[CacheSpan]:
    if not chunks:
        return []

    token_counts = Counter(
        token.lower()
        for chunk in chunks
        for token in chunk.tokens
    )
    spans: list[CacheSpan] = []
    for chunk in chunks:
        for start in range(0, len(chunk.tokens), config.span_size):
            end = min(start + config.span_size, len(chunk.tokens))
            if start == end:
                continue
            score = _span_rarity_score(chunk.tokens[start:end], token_counts, config.rarity_floor)
            spans.append(
                CacheSpan(
                    chunk_id=chunk.chunk_id,
                    start_token=chunk.start_token + start,
                    end_token=chunk.start_token + end,
                    score=score,
                    text=" ".join(chunk.tokens[start:end]),
                )
            )
    spans.sort(key=lambda item: item.score, reverse=True)
    return spans[: config.max_spans]


def _span_rarity_score(tokens: list[str], counts: Counter[str], floor: float) -> float:
    if not tokens:
        return 0.0
    total = sum(counts.values())
    score = 0.0
    for token in tokens:
        freq = counts[token.lower()] / max(1, total)
        score += -log(max(freq, floor))
    return score / len(tokens)
