from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import platform
from pathlib import Path
import subprocess


@dataclass(slots=True)
class EnvironmentSnapshot:
    created_at: str
    python_version: str
    platform: str
    uv_version: str


def capture_environment_snapshot() -> EnvironmentSnapshot:
    uv_version = _run_command(["uv", "--version"]).strip()
    return EnvironmentSnapshot(
        created_at=datetime.now(UTC).isoformat(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        uv_version=uv_version,
    )


def write_environment_snapshot(path: str | Path, snapshot: EnvironmentSnapshot) -> None:
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(snapshot), indent=2) + "\n")


def _run_command(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout or result.stderr
