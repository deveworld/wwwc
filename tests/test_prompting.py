from tristore_bma.prompting import build_final_prompt, build_gemma_messages


def test_build_final_prompt_sections() -> None:
    prompt = build_final_prompt(
        instruction="Use bounded evidence.",
        query="What matters?",
        scaffold=["A"],
        cache=["B"],
    )
    text = prompt.render_text()
    assert "[scaffold]" in text
    assert "[cache]" in text
    assert "What matters?" in text


def test_build_gemma_messages_multimodal_format() -> None:
    messages = build_gemma_messages(
        system_instruction="Use bounded evidence.",
        query="What matters?",
        scaffold=["A"],
        cache=["B"],
        enable_thinking=True,
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "[scaffold]" in content[0]["text"]


def test_build_gemma_messages_no_assistant_message() -> None:
    """enable_thinking is handled by processor, not in messages."""
    messages = build_gemma_messages(
        system_instruction="Be helpful.",
        query="Hello",
        scaffold=[],
        cache=[],
        enable_thinking=False,
    )
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
