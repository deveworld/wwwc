from pathlib import Path
import json

from tristore_bma.reporting import summarize_artifacts


def test_summarize_artifacts_reads_accounting_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "accounting.json").write_text(
        json.dumps(
            {
                "gemma_variant": "gemma-4-E2B-it",
                "route_overhead_ms": 10.0,
                "final_materialized_prompt_length": 120,
                "total_write_steps": 3,
            }
        )
    )

    report = summarize_artifacts(tmp_path)

    assert report.run_count == 1
    assert report.average_route_overhead_ms == 10.0
    assert report.total_write_steps == 3
