from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class GemmaVariant(StrEnum):
    E2B = "gemma-4-E2B-it"
    E4B = "gemma-4-E4B-it"
    A4B = "gemma-4-26b-a4b-it"
    D31B = "gemma-4-31b-it"


@dataclass(slots=True)
class ScaffoldConfig:
    max_scaffold_tokens: int = 768
    max_scaffold_spans: int = 6
    chunk_size: int = 160
    chunk_overlap: int = 32
    lexical_weight: float = 0.7
    tfidf_weight: float = 0.3
    coverage_decay: float = 0.85


@dataclass(slots=True)
class PreselectorConfig:
    shortlist_k: int = 8
    lexical_weight: float = 0.6
    tfidf_weight: float = 0.4


@dataclass(slots=True)
class CacheConfig:
    span_size: int = 48
    max_spans: int = 6
    rarity_floor: float = 1e-8


@dataclass(slots=True)
class BudgetConfig:
    ratios: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5])
    base_latency_ms: float = 1000.0


@dataclass(slots=True)
class AllocationConfig:
    total_budget_units: int = 8
    write_step_cost: int = 1
    cache_span_cost: int = 1
    write_penalty_delta: float = 0.15


@dataclass(slots=True)
class CalibrationConfig:
    gemma_variant: GemmaVariant = GemmaVariant.E2B
    manifests_path: Path = Path("manifests/ruler_mixed_slices.yaml")
    scaffold: ScaffoldConfig = field(default_factory=ScaffoldConfig)
    preselector: PreselectorConfig = field(default_factory=PreselectorConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    sample_query: str = "Summarize the entities relevant to the question."
    sample_document_path: Path | None = None
    output_dir: Path = Path("artifacts")

    @classmethod
    def from_file(cls, path: str | Path) -> "CalibrationConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            gemma_variant=GemmaVariant(raw.get("gemma_variant", GemmaVariant.E2B)),
            manifests_path=Path(raw.get("manifests_path", "manifests/ruler_mixed_slices.yaml")),
            scaffold=_build_dataclass(ScaffoldConfig, raw.get("scaffold", {})),
            preselector=_build_dataclass(PreselectorConfig, raw.get("preselector", {})),
            cache=_build_dataclass(CacheConfig, raw.get("cache", {})),
            budget=_build_dataclass(BudgetConfig, raw.get("budget", {})),
            allocation=_build_dataclass(AllocationConfig, raw.get("allocation", {})),
            sample_query=raw.get("sample_query", "Summarize the entities relevant to the question."),
            sample_document_path=(
                Path(raw["sample_document_path"]) if raw.get("sample_document_path") else None
            ),
            output_dir=Path(raw.get("output_dir", "artifacts")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gemma_variant": self.gemma_variant.value,
            "manifests_path": str(self.manifests_path),
            "scaffold": asdict(self.scaffold),
            "preselector": asdict(self.preselector),
            "cache": asdict(self.cache),
            "budget": asdict(self.budget),
            "allocation": asdict(self.allocation),
            "sample_query": self.sample_query,
            "sample_document_path": str(self.sample_document_path) if self.sample_document_path else None,
            "output_dir": str(self.output_dir),
        }


def _build_dataclass(cls: type[Any], raw: dict[str, Any]) -> Any:
    return cls(**raw)
