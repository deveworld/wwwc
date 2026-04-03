from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DocumentRecord:
    record_id: str
    benchmark: str
    slice_name: str
    query: str
    document: str
    answers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_jsonl_records(path: str | Path) -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        records.append(
            DocumentRecord(
                record_id=str(raw["record_id"]),
                benchmark=str(raw["benchmark"]),
                slice_name=str(raw["slice_name"]),
                query=str(raw["query"]),
                document=str(raw["document"]),
                answers=list(raw.get("answers", [])),
                metadata=dict(raw.get("metadata", {})),
            )
        )
    return records


def filter_records(
    records: list[DocumentRecord],
    *,
    benchmark: str | None = None,
    slice_names: set[str] | None = None,
    record_ids: set[str] | None = None,
) -> list[DocumentRecord]:
    selected = records
    if benchmark is not None:
        selected = [record for record in selected if record.benchmark == benchmark]
    if slice_names is not None:
        selected = [record for record in selected if record.slice_name in slice_names]
    if record_ids is not None:
        selected = [record for record in selected if record.record_id in record_ids]
    return selected
