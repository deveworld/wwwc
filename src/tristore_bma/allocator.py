from __future__ import annotations

from dataclasses import dataclass

from .cache import CacheSpan
from .config import AllocationConfig
from .preselector import RankedChunk


@dataclass(slots=True)
class WriteCandidate:
    chunk_id: int
    score: float
    next_step_index: int = 1


@dataclass(slots=True)
class AllocationDecision:
    step: int
    target: str
    target_id: int
    marginal_gain: float
    budget_remaining: int


def build_write_candidates(
    shortlist: list[RankedChunk],
) -> list[WriteCandidate]:
    return [WriteCandidate(chunk_id=item.chunk.chunk_id, score=item.score) for item in shortlist]


def allocate_interleaved_budget(
    write_candidates: list[WriteCandidate],
    cache_candidates: list[CacheSpan],
    config: AllocationConfig,
) -> list[AllocationDecision]:
    budget_remaining = config.total_budget_units
    decisions: list[AllocationDecision] = []
    write_state = {candidate.chunk_id: candidate.next_step_index for candidate in write_candidates}
    cache_index = 0
    step = 0

    while budget_remaining > 0:
        step += 1
        write_gain, write_chunk_id = _best_write_gain(write_candidates, write_state, config)
        cache_gain, cache_target_id = _best_cache_gain(cache_candidates, cache_index)

        can_write = write_chunk_id is not None and budget_remaining >= config.write_step_cost
        can_cache = cache_target_id is not None and budget_remaining >= config.cache_span_cost

        if not can_write and not can_cache:
            break

        if can_write and (not can_cache or write_gain >= cache_gain):
            assert write_chunk_id is not None
            budget_remaining -= config.write_step_cost
            decisions.append(
                AllocationDecision(
                    step=step,
                    target="write",
                    target_id=write_chunk_id,
                    marginal_gain=write_gain,
                    budget_remaining=budget_remaining,
                )
            )
            write_state[write_chunk_id] += 1
            continue

        assert cache_target_id is not None
        budget_remaining -= config.cache_span_cost
        decisions.append(
            AllocationDecision(
                step=step,
                target="cache",
                target_id=cache_target_id,
                marginal_gain=cache_gain,
                budget_remaining=budget_remaining,
            )
        )
        cache_index += 1

    return decisions


def _best_write_gain(
    candidates: list[WriteCandidate],
    write_state: dict[int, int],
    config: AllocationConfig,
) -> tuple[float, int | None]:
    if not candidates:
        return 0.0, None
    best_gain = float("-inf")
    best_chunk_id: int | None = None
    for candidate in candidates:
        step_idx = write_state[candidate.chunk_id]
        gain = candidate.score - config.write_penalty_delta * (step_idx - 1)
        if gain > best_gain:
            best_gain = gain
            best_chunk_id = candidate.chunk_id
    return max(best_gain, 0.0), best_chunk_id


def _best_cache_gain(cache_candidates: list[CacheSpan], cache_index: int) -> tuple[float, int | None]:
    if cache_index >= len(cache_candidates):
        return 0.0, None
    candidate = cache_candidates[cache_index]
    return candidate.score, candidate.chunk_id
