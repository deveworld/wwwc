from tristore_bma.config import GemmaVariant
from tristore_bma.gemma_runner import (
    CpuEchoGemmaRunner,
    GemmaRequest,
    VARIANT_TO_HF_ID,
    parse_thinking_response,
)


def test_cpu_echo_runner_marks_thinking() -> None:
    runner = CpuEchoGemmaRunner()
    response = runner.run(
        GemmaRequest(
            variant=GemmaVariant.E2B,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
            enable_thinking=True,
        )
    )

    assert "thinking=on" in response.final_text
    assert response.thought_text == "stub_thought"


def test_variant_to_hf_id_mapping() -> None:
    assert VARIANT_TO_HF_ID[GemmaVariant.E2B] == "google/gemma-4-E2B-it"
    assert VARIANT_TO_HF_ID[GemmaVariant.E4B] == "google/gemma-4-E4B-it"


def test_parse_thinking_response_with_thought() -> None:
    raw = "<|channel>thought\nStep 1: think\nStep 2: reason\n<|/channel>\n4 is the answer."
    thought, final = parse_thinking_response(raw)
    assert "Step 1" in thought
    assert "4 is the answer" in final


def test_parse_thinking_response_without_thought() -> None:
    raw = "Just a plain answer."
    thought, final = parse_thinking_response(raw)
    assert thought == ""
    assert final == "Just a plain answer."
