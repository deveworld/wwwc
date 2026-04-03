from __future__ import annotations

from dataclasses import dataclass

from .config import GemmaVariant


@dataclass(slots=True)
class RunSpec:
    benchmark: str
    manifest: str
    gemma_variant: str
    budget_ratio: float
    seed: int
    thinking_enabled: bool


def generate_run_matrix(
    *,
    benchmark: str,
    manifest: str,
    variants: list[GemmaVariant],
    budget_ratios: list[float],
    seeds: list[int],
    include_thinking: bool,
) -> list[RunSpec]:
    thinking_flags = [False, True] if include_thinking else [False]
    return [
        RunSpec(
            benchmark=benchmark,
            manifest=manifest,
            gemma_variant=variant.value,
            budget_ratio=budget_ratio,
            seed=seed,
            thinking_enabled=thinking_enabled,
        )
        for variant in variants
        for budget_ratio in budget_ratios
        for seed in seeds
        for thinking_enabled in thinking_flags
    ]
