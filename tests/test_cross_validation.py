from tristore_bma.datasets import DocumentRecord
from tristore_bma.validation import validate_manifest_against_dataset


def test_validate_manifest_against_dataset_accepts_matching_records() -> None:
    manifest = {
        "name": "longbench_v2_selection",
        "frozen_at": "2026-04-03",
        "categories": ["multi_document_qa"],
        "subset_ids": ["lbv2-mdqa-001"],
    }
    records = [
        DocumentRecord(
            record_id="lbv2-mdqa-001",
            benchmark="longbench_v2",
            slice_name="multi_document_qa",
            query="q",
            document="d",
        )
    ]

    result = validate_manifest_against_dataset(manifest, records)

    assert result.ok is True
