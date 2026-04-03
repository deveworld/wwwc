from pathlib import Path

from tristore_bma.repro import capture_environment_snapshot, write_environment_snapshot


def test_capture_and_write_environment_snapshot(tmp_path: Path) -> None:
    snapshot = capture_environment_snapshot()
    output = tmp_path / "env.json"
    write_environment_snapshot(output, snapshot)

    assert output.exists()
    assert "uv " in snapshot.uv_version
