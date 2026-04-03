# GPU Handoff

This repository is prepared for the GPU-dependent part of the calibration sprint.

## Already locked on CPU

- Gemma 4 variant enum and config schema
- manifest structure and preregistered slice files
- stable scaffold / preselector / cache proposal interfaces
- interleaving allocator simulation
- accounting log schema
- CLI utilities for manifest inspection, matrix generation, and budget calibration checks
- conservative prompt/message placeholders; real HF `apply_chat_template(..., enable_thinking=...)` remains the source of truth

## First GPU tasks

1. `uv sync --extra model --extra dev`
2. Replace `CpuEchoGemmaRunner` with a real `TransformersGemmaRunner`
3. Verify Gemma 4 chat template and `enable_thinking` parsing on the selected variant
4. Record measured `route_overhead_ms` and run:

```bash
uv run tristore-calibrate budget-check --config configs/gpu_handoff.yaml --median-route-overhead-ms 220
```

5. Start with:

```bash
uv run tristore-calibrate calibrate --config configs/gpu_handoff.yaml
```

## Go/No-Go targets

- default Gemma 4 variant lock
- chat template / response parsing lock
- shortlist `K` lock
- chunk size lock
- budget grid lock
- write-only vs stable-only first check on dependency-heavy slices
