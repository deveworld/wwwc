from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
import json

from .config import CalibrationConfig


@dataclass(slots=True)
class RunAccounting:
    gemma_variant: str
    raw_document_length: int
    final_materialized_prompt_length: int
    scaffold_token_count: int
    shortlist_k: int
    chunk_size: int
    cached_spans: int
    route_overhead_ms: float
    decode_overhead_ms: float
    total_write_steps: int
    thinking_enabled: bool
    budget_ratios: list[float]
    created_at: str


def build_accounting(
    config: CalibrationConfig,
    *,
    raw_document_length: int,
    final_materialized_prompt_length: int,
    scaffold_token_count: int,
    cached_spans: int,
    route_overhead_ms: float,
    decode_overhead_ms: float,
    total_write_steps: int,
    thinking_enabled: bool,
) -> RunAccounting:
    return RunAccounting(
        gemma_variant=config.gemma_variant.value,
        raw_document_length=raw_document_length,
        final_materialized_prompt_length=final_materialized_prompt_length,
        scaffold_token_count=scaffold_token_count,
        shortlist_k=config.preselector.shortlist_k,
        chunk_size=config.scaffold.chunk_size,
        cached_spans=cached_spans,
        route_overhead_ms=route_overhead_ms,
        decode_overhead_ms=decode_overhead_ms,
        total_write_steps=total_write_steps,
        thinking_enabled=thinking_enabled,
        budget_ratios=list(config.budget.ratios),
        created_at=datetime.now(UTC).isoformat(),
    )


def write_accounting(path: str | Path, accounting: RunAccounting) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(accounting), indent=2) + "\n")
