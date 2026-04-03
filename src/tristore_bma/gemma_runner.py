from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .config import GemmaVariant

VARIANT_TO_HF_ID: dict[GemmaVariant, str] = {
    GemmaVariant.E2B: "google/gemma-4-E2B-it",
    GemmaVariant.E4B: "google/gemma-4-E4B-it",
    GemmaVariant.A4B: "google/gemma-4-26b-a4b-it",
    GemmaVariant.D31B: "google/gemma-4-31b-it",
}

# Gemma 4 uses <|channel>thought ... <|/channel> to wrap thinking output.
_THOUGHT_RE = re.compile(
    r"<\|channel>thought\s*(.*?)\s*<\|/channel>",
    re.DOTALL,
)


@dataclass(slots=True)
class GemmaRequest:
    variant: GemmaVariant
    messages: list[dict]
    enable_thinking: bool
    max_new_tokens: int = 256


@dataclass(slots=True)
class GemmaResponse:
    final_text: str
    thought_text: str
    raw_text: str


def parse_thinking_response(raw: str) -> tuple[str, str]:
    """Split raw generation into (thought_text, final_text).

    If no thinking block is found, thought_text is empty and
    final_text is the full raw text (stripped).
    """
    match = _THOUGHT_RE.search(raw)
    if match:
        thought = match.group(1).strip()
        final = raw[match.end():].strip()
        return thought, final
    return "", raw.strip()


class GemmaRunner(Protocol):
    def run(self, request: GemmaRequest) -> GemmaResponse: ...


class CpuEchoGemmaRunner:
    """CPU-only stand-in that validates the intended request shape."""

    def run(self, request: GemmaRequest) -> GemmaResponse:
        roles = ",".join(message["role"] for message in request.messages)
        final_text = (
            f"CPU stub for {request.variant.value}; roles={roles}; "
            f"thinking={'on' if request.enable_thinking else 'off'}"
        )
        thought_text = "stub_thought" if request.enable_thinking else ""
        return GemmaResponse(
            final_text=final_text,
            thought_text=thought_text,
            raw_text=f"{thought_text}\n{final_text}".strip(),
        )


class TransformersGemmaRunner:
    """GPU runner using HuggingFace Transformers for Gemma 4 models."""

    def __init__(self, variant: GemmaVariant) -> None:
        import torch  # pyright: ignore[reportMissingImports]
        from transformers import AutoModelForCausalLM, AutoProcessor  # pyright: ignore[reportMissingImports]

        hf_id = VARIANT_TO_HF_ID[variant]
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self._variant = variant

    def run(self, request: GemmaRequest) -> GemmaResponse:
        import torch  # pyright: ignore[reportMissingImports]

        inputs = self.processor.apply_chat_template(
            request.messages,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=request.enable_thinking,
            tokenize=True,
            return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
            )

        n_input = inputs["input_ids"].shape[1]
        raw_text = self.processor.decode(
            output_ids[0][n_input:],
            skip_special_tokens=False,
        )

        thought_text, final_text = parse_thinking_response(raw_text)
        return GemmaResponse(
            final_text=final_text,
            thought_text=thought_text,
            raw_text=raw_text,
        )
