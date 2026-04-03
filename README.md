# TriStore-BMA

CPU-first scaffolding for the Go/No-Go calibration stage in `research_plan.md`.

## Status

**Go/No-Go: PASS (2026-04-03, A100 MIG 2g-20GB)**

All four critical checks passed. Ready for full calibration on RTX PRO 6000.

| Check | Status | Summary |
|-------|--------|---------|
| Scaffold quality | PASS | Model correctly uses bounded scaffold context |
| Write vs Stable | PASS | Richer context enables multi-hop reasoning |
| Cache exact recall | PASS | Exact spans eliminate hallucination |
| Thinking mode | PASS | Distinct thinking/non-thinking decode paths |

See [`docs/BRINGUP_LOG.md`](docs/BRINGUP_LOG.md) for detailed bring-up history.

## Verified Model Configuration

- **Model:** `google/gemma-4-E2B-it` (5.12B params, bfloat16)
- **API:** `AutoProcessor` + `AutoModelForCausalLM` (not AutoTokenizer)
- **Message format:** Multimodal — `[{"type": "text", "text": "..."}]`
- **Thinking:** `enable_thinking=True/False` in `processor.apply_chat_template()`
- **Thinking tags:** `<|channel>thought ... <|/channel>`

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

## GPU Execution

When a GPU is available:

```bash
# For CUDA 13+ (torch 2.11)
uv sync --extra model --extra dev

# For CUDA 12.x (torch 2.5.x)
python3.10 -m venv .venv
.venv/bin/pip install 'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121' --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install 'transformers>=4.51' accelerate safetensors tokenizers pillow pyyaml scikit-learn tqdm -e .
```

Run go/no-go checks:

```bash
.venv/bin/python scripts/go_nogo.py
```

Start calibration:

```bash
uv run tristore-calibrate calibrate --config configs/gpu_handoff.yaml
```

Read [`docs/GPU_HANDOFF.md`](docs/GPU_HANDOFF.md) for the full GPU execution checklist.

## What Exists

### Core Pipeline
- **Chunking:** sliding window with configurable overlap
- **Scaffold:** TF-IDF + lexical query matching with coverage penalty
- **Preselector:** top-K chunk shortlisting
- **Cache:** rarity-based exact span selection
- **Allocator:** marginal-utility interleaving (write vs cache)
- **Simulation:** toy error model for plausibility checks

### Model Integration (GPU-verified)
- **TransformersGemmaRunner:** real model loading, inference, thinking/non-thinking generation
- **Response parsing:** `<|channel>thought` tag extraction
- **VARIANT_TO_HF_ID mapping:** E2B, E4B, A4B, D31B

### Infrastructure
- Experiment config and manifest locking
- Manifest and dataset validation / cross-validation
- Budget calibration rule checker
- Run-matrix generator
- Prompt formatting (multimodal Gemma 4 format)
- Latency/accounting log schema
- Environment snapshot and artifact summary utilities
- Dry-run calibration CLI with 10+ subcommands
- 18 passing tests, ruff/pyright/ty clean

## Repository Layout

- [`research_plan.md`](research_plan.md): locked research plan (v5.1, Gemma 4)
- [`configs/`](configs): calibration and handoff configs
- [`manifests/`](manifests): preregistered benchmark and experiment manifests
- [`data/samples/`](data/samples): small sample datasets for CPU checks
- [`src/tristore_bma/`](src/tristore_bma): library and CLI code
- [`tests/`](tests): CPU-only validation tests (18 tests)
- [`scripts/`](scripts): GPU bring-up and go/no-go scripts
- [`docs/GPU_HANDOFF.md`](docs/GPU_HANDOFF.md): GPU execution checklist and locked specs
- [`docs/BRINGUP_LOG.md`](docs/BRINGUP_LOG.md): detailed bring-up history and troubleshooting

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

- LoRA write-step updates not yet implemented (next: RTX PRO 6000 calibration)
- Route overhead and budget grid not yet locked (hardware-dependent)
- Full RULER/LongBench v2 benchmark loaders beyond sample JSONL files
- Hybrid interior split behavior not yet empirically confirmed
