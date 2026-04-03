from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BudgetCalibrationInput:
    base_latency_ms: float
    budget_ratios: list[float]
    median_route_overhead_ms: float
    shortlist_k: int


@dataclass(slots=True)
class BudgetCalibrationReport:
    small_budget_ms: float
    medium_budget_ms: float
    route_over_small_ratio: float
    route_over_medium_ratio: float
    recommended_action: str
    recommended_budget_ratios: list[float]
    recommended_shortlist_k: int


def evaluate_budget_calibration(config: BudgetCalibrationInput) -> BudgetCalibrationReport:
    budgets_ms = [config.base_latency_ms * ratio for ratio in config.budget_ratios]
    if len(budgets_ms) < 3:
        raise ValueError("budget_ratios must contain at least three points")

    small_budget_ms = budgets_ms[1]
    medium_budget_ms = budgets_ms[2]
    route_over_small_ratio = config.median_route_overhead_ms / max(small_budget_ms, 1e-8)
    route_over_medium_ratio = config.median_route_overhead_ms / max(medium_budget_ms, 1e-8)

    recommended_action = "keep"
    recommended_budget_ratios = list(config.budget_ratios)
    recommended_shortlist_k = config.shortlist_k

    if route_over_small_ratio > 0.8:
        recommended_action = "drop_small_budget_or_expand_grid"
        recommended_budget_ratios = [0.0, 0.5, 1.0, 1.5]
    if route_over_medium_ratio > 0.4:
        recommended_action = "reduce_k_or_expand_grid"
        recommended_shortlist_k = max(1, config.shortlist_k - 2)
        recommended_budget_ratios = [0.0, 0.5, 1.0, 1.5]

    return BudgetCalibrationReport(
        small_budget_ms=small_budget_ms,
        medium_budget_ms=medium_budget_ms,
        route_over_small_ratio=route_over_small_ratio,
        route_over_medium_ratio=route_over_medium_ratio,
        recommended_action=recommended_action,
        recommended_budget_ratios=recommended_budget_ratios,
        recommended_shortlist_k=recommended_shortlist_k,
    )
