from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FinalPrompt:
    instruction: str
    query: str
    scaffold: list[str]
    cache: list[str]

    def render_text(self) -> str:
        sections = [
            "[instruction]",
            self.instruction.strip(),
            "[query]",
            self.query.strip(),
            "[scaffold]",
            "\n".join(span.strip() for span in self.scaffold if span.strip()),
            "[cache]",
            "\n".join(span.strip() for span in self.cache if span.strip()),
        ]
        return "\n".join(sections).strip()


def build_final_prompt(
    *,
    instruction: str,
    query: str,
    scaffold: list[str],
    cache: list[str],
) -> FinalPrompt:
    return FinalPrompt(
        instruction=instruction,
        query=query,
        scaffold=scaffold,
        cache=cache,
    )


def build_gemma_messages(
    *,
    system_instruction: str,
    query: str,
    scaffold: list[str],
    cache: list[str],
    enable_thinking: bool,
) -> list[dict]:
    """Build Gemma 4 chat messages in multimodal content format.

    Gemma 4 processor.apply_chat_template expects content as a list
    of typed dicts: [{"type": "text", "text": "..."}].
    The enable_thinking flag is passed to apply_chat_template, not
    embedded in the messages themselves.
    """
    prompt = build_final_prompt(
        instruction=system_instruction,
        query=query,
        scaffold=scaffold,
        cache=cache,
    )
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt.render_text()}],
        },
    ]
