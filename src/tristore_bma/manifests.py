from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_manifest(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": manifest.get("name"),
        "frozen_at": manifest.get("frozen_at"),
        "raw_length_buckets": list(manifest.get("raw_length_buckets", [])),
        "mixed_slices": list(manifest.get("mixed_slices", [])),
        "categories": list(manifest.get("categories", [])),
        "subset_count": len(manifest.get("subset_ids", [])),
    }


def collect_expected_manifest_values(manifest: dict[str, Any]) -> dict[str, set[str]]:
    return {
        "mixed_slices": set(manifest.get("mixed_slices", [])),
        "categories": set(manifest.get("categories", [])),
        "subset_ids": set(manifest.get("subset_ids", [])),
    }
