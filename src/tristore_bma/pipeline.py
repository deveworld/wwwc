from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .allocator import AllocationDecision, allocate_interleaved_budget, build_write_candidates
from .cache import CacheSpan, propose_cache_spans
from .chunking import Chunk, chunk_document, whitespace_tokenize
from .config import CalibrationConfig
from .logging_utils import RunAccounting, build_accounting
from .preselector import RankedChunk, shortlist_chunks
from .scaffold import ScaffoldSpan, build_scaffold
from .simulation import SimulationResult, ToyTheoryInputs, simulate_expected_error


@dataclass(slots=True)
class CalibrationArtifacts:
    chunks: list[Chunk]
    scaffold: list[ScaffoldSpan]
    shortlist: list[RankedChunk]
    cache_spans: list[CacheSpan]
    allocation: list[AllocationDecision]
    simulation: SimulationResult
    accounting: RunAccounting


def run_cpu_calibration(config: CalibrationConfig, document_text: str) -> CalibrationArtifacts:
    chunks = chunk_document(
        document_text,
        chunk_size=config.scaffold.chunk_size,
        overlap=config.scaffold.chunk_overlap,
    )

    route_start = perf_counter()
    scaffold = build_scaffold(config.sample_query, chunks, config.scaffold)
    shortlist = shortlist_chunks(config.sample_query, chunks, config.preselector)
    cache_spans = propose_cache_spans([item.chunk for item in shortlist], config.cache)
    allocation = allocate_interleaved_budget(
        build_write_candidates(shortlist),
        cache_spans,
        config.allocation,
    )
    route_overhead_ms = (perf_counter() - route_start) * 1000.0

    scaffold_tokens = sum(len(whitespace_tokenize(span.text)) for span in scaffold)
    cached_tokens = sum(len(whitespace_tokenize(span.text)) for span in cache_spans)
    simulation = simulate_expected_error(
        allocation,
        ToyTheoryInputs(p_write=0.5, p_cache=0.5),
    )

    accounting = build_accounting(
        config,
        raw_document_length=len(whitespace_tokenize(document_text)),
        final_materialized_prompt_length=scaffold_tokens + cached_tokens + len(whitespace_tokenize(config.sample_query)),
        scaffold_token_count=scaffold_tokens,
        cached_spans=len(cache_spans),
        route_overhead_ms=route_overhead_ms,
        decode_overhead_ms=0.0,
        total_write_steps=sum(1 for item in allocation if item.target == "write"),
        thinking_enabled=False,
    )

    return CalibrationArtifacts(
        chunks=chunks,
        scaffold=scaffold,
        shortlist=shortlist,
        cache_spans=cache_spans,
        allocation=allocation,
        simulation=simulation,
        accounting=accounting,
    )
