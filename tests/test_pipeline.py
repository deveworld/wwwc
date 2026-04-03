from tristore_bma.config import CalibrationConfig
from tristore_bma.pipeline import run_cpu_calibration


def test_cpu_calibration_produces_accounting() -> None:
    config = CalibrationConfig()
    artifacts = run_cpu_calibration(
        config,
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
    )

    assert artifacts.accounting.gemma_variant == config.gemma_variant.value
    assert artifacts.accounting.raw_document_length > 0
    assert artifacts.accounting.shortlist_k == config.preselector.shortlist_k
    assert artifacts.accounting.total_write_steps == artifacts.simulation.write_units
