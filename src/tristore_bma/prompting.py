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
) -> list[dict[str, str]]:
    prompt = build_final_prompt(
        instruction=system_instruction,
        query=query,
        scaffold=scaffold,
        cache=cache,
    )
    return [
        {
            "role": "user",
            "content": prompt.render_text(),
        },
        {
            "role": "assistant",
            "content": "" if not enable_thinking else "[thinking-enabled-via-chat-template]",
        },
    ]
