from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk, whitespace_tokenize
from .config import PreselectorConfig


@dataclass(slots=True)
class RankedChunk:
    chunk: Chunk
    score: float


def shortlist_chunks(query: str, chunks: list[Chunk], config: PreselectorConfig) -> list[RankedChunk]:
    if not chunks:
        return []

    query_terms = set(whitespace_tokenize(query.lower()))
    vectorizer = TfidfVectorizer(lowercase=True)
    matrix = vectorizer.fit_transform([query, *(chunk.text for chunk in chunks)])
    query_vec = matrix[0]
    chunk_vecs = matrix[1:]
    tfidf_scores = cosine_similarity(chunk_vecs, query_vec).ravel()

    ranked: list[RankedChunk] = []
    for idx, chunk in enumerate(chunks):
        lexical_hits = sum(1 for token in chunk.tokens if token.lower() in query_terms)
        lexical_score = lexical_hits / max(1, len(chunk.tokens))
        combined = config.lexical_weight * lexical_score + config.tfidf_weight * float(tfidf_scores[idx])
        ranked.append(RankedChunk(chunk=chunk, score=combined))

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[: config.shortlist_k]
