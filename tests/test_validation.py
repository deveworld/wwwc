from tristore_bma.validation import validate_manifest


def test_validate_manifest_accepts_expected_shape() -> None:
    result = validate_manifest(
        {
            "name": "ruler_mixed_slices",
            "frozen_at": "2026-04-03",
            "mixed_slices": ["retrieval_plus_tracing"],
        }
    )

    assert result.ok is True
    assert result.errors == []
