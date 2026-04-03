from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class ArtifactReport:
    run_count: int
    variants: list[str]
    average_route_overhead_ms: float
    average_prompt_tokens: float
    total_write_steps: int


def summarize_artifacts(root: str | Path) -> ArtifactReport:
    root_path = Path(root)
    accounting_files = sorted(root_path.glob("**/accounting.json"))
    payloads = [json.loads(path.read_text()) for path in accounting_files]
    if not payloads:
        return ArtifactReport(
            run_count=0,
            variants=[],
            average_route_overhead_ms=0.0,
            average_prompt_tokens=0.0,
            total_write_steps=0,
        )

    return ArtifactReport(
        run_count=len(payloads),
        variants=sorted({item["gemma_variant"] for item in payloads}),
        average_route_overhead_ms=sum(item["route_overhead_ms"] for item in payloads) / len(payloads),
        average_prompt_tokens=sum(item["final_materialized_prompt_length"] for item in payloads) / len(payloads),
        total_write_steps=sum(int(item["total_write_steps"]) for item in payloads),
    )
