# TriStore-BMA

CPU-first scaffolding for the Go/No-Go calibration stage in `research_plan.md`.

## Status

This repository is ready for the GPU-dependent calibration phase.

- CPU-only scaffolding is implemented
- manifests/configs are frozen for the current plan
- static checks pass: `ruff`, `pyright`, `ty`
- tests pass: `13 passed`
- real Gemma 4 loading and write-step validation remain GPU-blocked

## Quickstart

```bash
uv sync --extra dev
uv run tristore-calibrate --config configs/calibration_cpu.yaml --dry-run
uv run tristore-calibrate inspect-manifest --manifest manifests/ruler_mixed_slices.yaml
uv run tristore-calibrate validate-manifest --manifest manifests/ruler_mixed_slices.yaml
uv run tristore-calibrate cross-validate --manifest manifests/longbench_v2_selection.yaml --dataset data/samples/longbench_v2_mini.jsonl
uv run tristore-calibrate inspect-dataset --dataset data/samples/ruler_mini.jsonl
uv run tristore-calibrate runner-smoke --config configs/calibration_cpu.yaml --thinking
uv run tristore-calibrate budget-check --config configs/calibration_cpu.yaml --median-route-overhead-ms 250
uv run tristore-calibrate generate-matrix --config configs/calibration_cpu.yaml --benchmark ruler --manifest ruler_mixed_slices --include-thinking
uv run tristore-calibrate prompt-smoke --config configs/calibration_cpu.yaml --thinking
uv run tristore-calibrate env-snapshot --output artifacts/env_snapshot.json
uv run tristore-calibrate summarize-artifacts --root artifacts
uv run pytest
```

## GPU Handoff

When a GPU is available:

```bash
uv sync --extra model --extra dev
uv run tristore-calibrate calibrate --config configs/gpu_handoff.yaml
```

Read [`GPU_HANDOFF.md`](docs/GPU_HANDOFF.md) before the first model run.

## What Exists

This repository currently focuses on the GPU-free portion of Week 1:

- experiment config and manifest locking
- chunking
- CPU-only stable scaffold construction
- CPU-only preselector
- heuristic cache span proposal
- marginal-utility interleaving allocator simulation
- manifest and dataset inspection utilities
- budget calibration rule checker
- run-matrix generator
- prompt formatting utilities
- Gemma 4 runner interface with CPU stub
- latency/accounting log schema
- environment snapshot and artifact summary utilities
- dry-run calibration CLI

Gemma 4 model execution and write updates remain GPU-dependent follow-up work.

## Repository Layout

- [`research_plan.md`](research_plan.md): locked research plan
- [`configs/`](configs): calibration and handoff configs
- [`manifests/`](manifests): preregistered benchmark and experiment manifests
- [`data/samples/`](data/samples): small sample datasets for CPU checks
- [`src/tristore_bma/`](src/tristore_bma): library and CLI code
- [`tests/`](tests): CPU-only validation tests
- [`docs/GPU_HANDOFF.md`](docs/GPU_HANDOFF.md): first GPU execution checklist

## Useful Commands

```bash
uvx ruff check .
uvx pyright
uvx ty check .
uv run pytest
uv run tristore-calibrate validate-manifest --manifest manifests/ruler_mixed_slices.yaml
uv run tristore-calibrate cross-validate --manifest manifests/longbench_v2_selection.yaml --dataset data/samples/longbench_v2_mini.jsonl
uv run tristore-calibrate summarize-artifacts --root artifacts
```

## Current Limits

- `TransformersGemmaRunner` is still a placeholder until GPU-backed model loading is wired
- prompt/message helpers are conservative placeholders, not the final Hugging Face `apply_chat_template(...)` integration
- Go/No-Go evidence cannot be produced until actual Gemma 4 inference and write updates run on GPU
