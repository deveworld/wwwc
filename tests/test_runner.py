from tristore_bma.config import GemmaVariant
from tristore_bma.gemma_runner import CpuEchoGemmaRunner, GemmaRequest


def test_cpu_echo_runner_marks_thinking() -> None:
    runner = CpuEchoGemmaRunner()
    response = runner.run(
        GemmaRequest(
            variant=GemmaVariant.E2B,
            messages=[{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hello"}],
            enable_thinking=True,
        )
    )

    assert "thinking=on" in response.final_text
    assert response.thought_text == "stub_thought"
