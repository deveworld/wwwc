from tristore_bma.budgeting import BudgetCalibrationInput, evaluate_budget_calibration


def test_budget_calibration_reduces_k_when_route_over_medium_is_too_high() -> None:
    report = evaluate_budget_calibration(
        BudgetCalibrationInput(
            base_latency_ms=1000.0,
            budget_ratios=[0.0, 0.25, 0.5, 1.0],
            median_route_overhead_ms=250.0,
            shortlist_k=8,
        )
    )

    assert report.recommended_action == "reduce_k_or_expand_grid"
    assert report.recommended_shortlist_k == 6
