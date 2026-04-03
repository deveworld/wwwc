from pathlib import Path

from tristore_bma.datasets import filter_records, load_jsonl_records


def test_load_and_filter_jsonl_records(tmp_path: Path) -> None:
    dataset = tmp_path / "sample.jsonl"
    dataset.write_text(
        "\n".join(
            [
                '{"record_id":"1","benchmark":"ruler","slice_name":"retrieval_plus_tracing","query":"q","document":"d"}',
                '{"record_id":"2","benchmark":"longbench_v2","slice_name":"code_repository_understanding","query":"q2","document":"d2"}',
            ]
        )
    )

    records = load_jsonl_records(dataset)
    filtered = filter_records(records, benchmark="ruler")

    assert len(records) == 2
    assert len(filtered) == 1
    assert filtered[0].record_id == "1"
