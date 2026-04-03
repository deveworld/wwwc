from pathlib import Path

from tristore_bma.config import CalibrationConfig, GemmaVariant


def test_config_from_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "gemma_variant: gemma-4-E4B-it",
                "sample_query: test query",
                "scaffold:",
                "  chunk_size: 64",
            ]
        )
    )

    config = CalibrationConfig.from_file(config_path)

    assert config.gemma_variant == GemmaVariant.E4B
    assert config.sample_query == "test query"
    assert config.scaffold.chunk_size == 64
