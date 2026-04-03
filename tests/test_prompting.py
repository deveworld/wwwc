from tristore_bma.prompting import build_final_prompt, build_gemma_messages


def test_build_gemma_messages_includes_think_token() -> None:
    prompt = build_final_prompt(
        instruction="Use bounded evidence.",
        query="What matters?",
        scaffold=["A"],
        cache=["B"],
    )
    messages = build_gemma_messages(
        system_instruction="Use bounded evidence.",
        query="What matters?",
        scaffold=["A"],
        cache=["B"],
        enable_thinking=True,
    )

    assert "[scaffold]" in prompt.render_text()
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert "thinking-enabled-via-chat-template" in messages[1]["content"]
