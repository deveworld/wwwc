from tristore_bma.config import GemmaVariant
from tristore_bma.matrix import generate_run_matrix


def test_generate_run_matrix_counts_all_combinations() -> None:
    matrix = generate_run_matrix(
        benchmark="ruler",
        manifest="ruler_mixed_slices",
        variants=[GemmaVariant.E2B, GemmaVariant.E4B],
        budget_ratios=[0.0, 0.5],
        seeds=[1, 2],
        include_thinking=True,
    )

    assert len(matrix) == 16
