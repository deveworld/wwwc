# TriStore-BMA [ARCHIVED]

> **Status: Project discontinued (2026-04-03).** See [POSTMORTEM](docs/POSTMORTEM.md) for details.
>
> Blog post: [My LoRA Memorized a Document and Still Couldn't Answer Questions About It](https://blog.worldsw.dev/tristore-bma/)

## What was this?

Research project targeting ICLR main track: budgeted test-time memory allocation between parametric write (LoRA) and exact cache (verbatim spans) for long-context LLMs.

## Why discontinued?

The core hypothesis required LoRA inference-time updates to create usable parametric memory. After rigorous testing on A100 80GB with Gemma 4 E2B-it:

- LoRA NTP loss converges (10.10 → 0.004) ✅
- LoRA weights update correctly ✅
- **Model cannot recall LoRA-stored information during generation** ❌

This is a **write-decode distribution mismatch**: LoRA memorizes next-token patterns from document text, but these patterns don't activate when the model is prompted with a different query format. The failure is structural, not a hyperparameter issue.

See [`docs/POSTMORTEM.md`](docs/POSTMORTEM.md) for full details.

## What's reusable?

- CPU pipeline: chunking, scaffold, preselector, cache, allocator (18 tests)
- Gemma 4 integration: AutoProcessor, multimodal format, thinking mode
- Budget-AUC evaluation framework
- Preregistered manifests (RULER, LongBench v2)

## Key lesson

> NTP loss convergence ≠ usable parametric memory. LoRA can memorize token patterns but cannot recall them under a different prompt distribution.
