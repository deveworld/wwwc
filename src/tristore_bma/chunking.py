from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Chunk:
    chunk_id: int
    start_token: int
    end_token: int
    text: str
    tokens: list[str]


def whitespace_tokenize(text: str) -> list[str]:
    return [token for token in text.split() if token]


def chunk_document(text: str, *, chunk_size: int, overlap: int) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = whitespace_tokenize(text)
    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for chunk_id, start in enumerate(range(0, len(tokens), step)):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            continue
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                start_token=start,
                end_token=end,
                text=" ".join(chunk_tokens),
                tokens=chunk_tokens,
            )
        )
        if end >= len(tokens):
            break
    return chunks
