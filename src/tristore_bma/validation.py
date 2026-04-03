from __future__ import annotations

from dataclasses import dataclass

from .datasets import DocumentRecord
from .manifests import collect_expected_manifest_values


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    errors: list[str]


def validate_manifest(manifest: dict) -> ValidationResult:
    errors: list[str] = []
    for key in ("name", "frozen_at"):
        if key not in manifest:
            errors.append(f"missing required key: {key}")

    has_slice_shape = "mixed_slices" in manifest or "categories" in manifest
    if not has_slice_shape:
        errors.append("manifest must define mixed_slices or categories")

    if "subset_ids" in manifest and not isinstance(manifest["subset_ids"], list):
        errors.append("subset_ids must be a list")
    if "raw_length_buckets" in manifest and not isinstance(manifest["raw_length_buckets"], list):
        errors.append("raw_length_buckets must be a list")

    return ValidationResult(ok=not errors, errors=errors)


def validate_manifest_against_dataset(manifest: dict, records: list[DocumentRecord]) -> ValidationResult:
    errors: list[str] = []
    expected = collect_expected_manifest_values(manifest)

    if expected["subset_ids"]:
        record_ids = {record.record_id for record in records}
        missing = sorted(expected["subset_ids"] - record_ids)
        if missing:
            errors.append(f"missing subset_ids in dataset: {missing}")

    if expected["mixed_slices"]:
        slice_names = {record.slice_name for record in records}
        missing = sorted(expected["mixed_slices"] - slice_names)
        if missing:
            errors.append(f"missing mixed_slices in dataset: {missing}")

    if expected["categories"]:
        categories = {record.slice_name for record in records}
        missing = sorted(expected["categories"] - categories)
        if missing:
            errors.append(f"missing categories in dataset: {missing}")

    return ValidationResult(ok=not errors, errors=errors)
