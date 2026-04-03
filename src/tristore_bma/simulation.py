from __future__ import annotations

from dataclasses import dataclass

from .allocator import AllocationDecision


@dataclass(slots=True)
class ToyTheoryInputs:
    p_write: float
    p_cache: float
    write_decay: float = 0.6
    cache_decay: float = 0.6


@dataclass(slots=True)
class SimulationResult:
    write_units: int
    cache_units: int
    expected_error: float


def simulate_expected_error(
    decisions: list[AllocationDecision],
    theory: ToyTheoryInputs,
) -> SimulationResult:
    write_units = sum(1 for item in decisions if item.target == "write")
    cache_units = sum(1 for item in decisions if item.target == "cache")
    write_term = theory.p_write * _residual(write_units, theory.write_decay)
    cache_term = theory.p_cache * _residual(cache_units, theory.cache_decay)
    return SimulationResult(
        write_units=write_units,
        cache_units=cache_units,
        expected_error=write_term + cache_term,
    )


def _residual(units: int, decay: float) -> float:
    return 1.0 / (1.0 + decay * units)
