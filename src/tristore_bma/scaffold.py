from __future__ import annotations

from dataclasses import dataclass
from math import exp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk, whitespace_tokenize
from .config import ScaffoldConfig


@dataclass(slots=True)
class ScaffoldSpan:
    chunk_id: int
    score: float
    text: str


def build_scaffold(query: str, chunks: list[Chunk], config: ScaffoldConfig) -> list[ScaffoldSpan]:
    if not chunks:
        return []

    query_terms = set(whitespace_tokenize(query.lower()))
    docs = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(lowercase=True)
    matrix = vectorizer.fit_transform([query, *docs])
    query_vec = matrix[0]
    chunk_vecs = matrix[1:]
    tfidf_scores = cosine_similarity(chunk_vecs, query_vec).ravel()

    selected: list[ScaffoldSpan] = []
    token_budget = 0
    covered_tokens: set[str] = set()

    for idx, chunk in enumerate(chunks):
        lexical_hits = sum(1 for token in chunk.tokens if token.lower() in query_terms)
        lexical_score = lexical_hits / max(1, len(chunk.tokens))
        coverage_penalty = _coverage_penalty(chunk.tokens, covered_tokens, config.coverage_decay)
        combined = (
            config.lexical_weight * lexical_score + config.tfidf_weight * float(tfidf_scores[idx])
        ) * coverage_penalty
        selected.append(ScaffoldSpan(chunk_id=chunk.chunk_id, score=combined, text=chunk.text))

    selected.sort(key=lambda item: item.score, reverse=True)
    pruned: list[ScaffoldSpan] = []
    for item in selected:
        item_tokens = whitespace_tokenize(item.text)
        if len(pruned) >= config.max_scaffold_spans:
            break
        if token_budget + len(item_tokens) > config.max_scaffold_tokens:
            continue
        pruned.append(item)
        token_budget += len(item_tokens)
        covered_tokens.update(token.lower() for token in item_tokens)

    return pruned


def _coverage_penalty(tokens: list[str], covered_tokens: set[str], decay: float) -> float:
    if not tokens:
        return 1.0
    overlap_ratio = sum(1 for token in tokens if token.lower() in covered_tokens) / len(tokens)
    return float(exp(-decay * overlap_ratio))
