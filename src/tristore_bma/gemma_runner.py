from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Protocol

from .config import GemmaVariant


@dataclass(slots=True)
class GemmaRequest:
    variant: GemmaVariant
    messages: list[dict[str, str]]
    enable_thinking: bool
    max_new_tokens: int = 256


@dataclass(slots=True)
class GemmaResponse:
    final_text: str
    thought_text: str
    raw_text: str


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
    """GPU-ready placeholder. It validates imports but does not load a model until invoked."""

    def run(self, request: GemmaRequest) -> GemmaResponse:
        try:
            importlib.import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "TransformersGemmaRunner requires the optional model dependencies. "
                "Install them with `uv sync --extra model`."
            ) from exc
        raise RuntimeError(
            "TransformersGemmaRunner is a GPU-dependent follow-up and is intentionally "
            "not wired for CPU-only execution."
        )
