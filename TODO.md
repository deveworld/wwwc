# TODO

## Done

- [x] Lock research plan to Gemma 4 family
- [x] Build `uv`-based Python project
- [x] Add CPU-only chunking, scaffold, preselector, cache proposal, allocator simulation
- [x] Add manifest inspection, validation, cross-validation, run-matrix, budget-check, and prompt-smoke CLIs
- [x] Add artifact summary and environment snapshot utilities
- [x] Add CPU sample datasets and preregistered manifests
- [x] Add tests for config, pipeline, allocator, validation, reporting, and prompt helpers
- [x] Pass `ruff`, `pyright`, `ty`, and `pytest`
- [x] Install model dependencies with `uv sync --extra model --extra dev`
- [x] Implement real Gemma 4 loading in `TransformersGemmaRunner`
- [x] Verify Hugging Face `apply_chat_template(..., enable_thinking=...)` for the chosen Gemma 4 variant
- [x] Verify response parsing for thinking and non-thinking outputs

## Done — Go/No-Go Checks (A100 MIG 2g-20GB, 2026-04-03)

- [x] Stable scaffold works with real Gemma 4 tokenizer/template
- [x] Thinking vs non-thinking decode path is reproducible
- [x] Write-only beats stable-only on dependency-heavy synthetic slices
- [x] Cache-only beats write-only on exact-recall slices

## Next — RTX PRO 6000 Calibration

- [ ] Gemma 4 E4B bring-up (`google/gemma-4-E4B-it`)
- [ ] Implement LoRA write-step update (adapter inference-time training)
- [ ] Measure real route overhead and run `budget-check` with measured numbers
- [ ] Lock budget grid based on measured overhead
- [ ] Run first `configs/gpu_handoff.yaml` calibration on RULER mini
- [ ] Lock chat template / parsing rule / default variant after first successful run
- [ ] Verify route overhead stays within the planned budget rule
- [ ] Verify hybrid shows plausible interior split behavior on mixed slices

## Nice To Have

- [ ] Add artifact export for run matrices
- [ ] Add real benchmark loaders beyond sample JSONL files
- [ ] Add GPU integration tests once hardware is available
- [ ] Gemma 4 26B A4B appendix experiments (after core claim is closed)
