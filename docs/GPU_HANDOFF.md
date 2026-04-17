# GPU Handoff

This document records the GPU bring-up process and verified findings.

## Bring-Up History

### Phase 1: Colab T4 (2026-04-03)

**Environment:** Google Colab, Tesla T4 16GB, Python 3.12, CUDA (Colab-managed)

**Findings:**

1. **Model ID confirmed:** `google/gemma-4-E2B-it` (note uppercase `E2B`)
   - HuggingFace model page: `https://huggingface.co/google/gemma-4-E2B-it`
   - Gated model: requires HF token + license acceptance
   - Model size: 10.2GB safetensors, 5.12B parameters

2. **API pattern verified:**
   - Must use `AutoProcessor` (not `AutoTokenizer`); Gemma 4 is multimodal
   - `AutoProcessor` requires `pillow` and `torchvision` even for text-only
   - Messages must use multimodal content format:
     ```python
     messages = [
         {"role": "user", "content": [{"type": "text", "text": "..."}]}
     ]
     ```
   - `enable_thinking` is passed to `processor.apply_chat_template()`, NOT embedded in messages
   - `return_dict=True` and `tokenize=True` required for `apply_chat_template`

3. **Chat template verified:**
   - `enable_thinking=False`: 16 input tokens for "What is 2+2?" → "2+2 is **4**."
   - `enable_thinking=True`: 22 input tokens (6 extra) → `<|channel>thought` ... thinking process ... `<|/channel>` ... answer

4. **Thinking output format:**
   - Opening tag: `<|channel>thought`
   - Closing tag: `<|/channel>`
   - Thought content contains structured reasoning (numbered steps)
   - Final answer follows after closing tag

5. **Deprecation warning:** `torch_dtype` parameter is deprecated; use `dtype` instead (non-breaking)

### Phase 2: A100 MIG 2g-20GB (2026-04-03)

**Environment:** NVIDIA A100 80GB PCIe MIG 2g.20gb, 20.9GB VRAM, 48GB RAM, Python 3.10, CUDA 12.2, Driver 535.161.07

**Compatibility notes:**
- torch 2.11.0 requires CUDA 13+; **torch 2.5.1+cu121** is the latest compatible version for CUDA 12.2 / Driver 535
- `requires-python` lowered to `>=3.10` for this environment
- Model loading time: 53.3s (including weight loading)

**Go/No-Go results (all PASS):**

| Check | Result | Key Data |
|-------|--------|----------|
| Scaffold quality | **PASS** | Model correctly identifies all 3 substrates from scaffold context |
| Write vs Stable | **PASS** | Stable→"Carol" (1 hop), Write→"Alice,Bob,Carol,Dave,Eve" (full chain) |
| Cache exact recall | **PASS** | No-cache→"42068" (hallucination), Cache→"ALPHA-9382-ZULU" (exact) |
| Thinking mode | **PASS** | Thought tag present, 256 tok/14.9s vs 21 tok/1.2s non-thinking |

**Verdict: GO, proceed to calibration**

## Locked Specifications

The following are now confirmed and locked based on GPU bring-up:

### Model
- **Default variant:** `google/gemma-4-E2B-it` (bring-up / robustness)
- **Main paper variant:** `google/gemma-4-E4B-it` (pending bring-up on RTX PRO 6000)
- **Parameters (E2B):** 5.12B
- **Loading:** `AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)`

### Processor
- **Class:** `AutoProcessor.from_pretrained(model_id)`
- **Chat template:** `processor.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=<bool>, tokenize=True, return_dict=True)`
- **Decoding:** `processor.decode(output_ids, skip_special_tokens=False)` for raw (with thought tags), `skip_special_tokens=True` for clean

### Message Format
```python
messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "<prompt text>"}],
    },
]
```

### Thinking Response Parsing
```python
import re
_THOUGHT_RE = re.compile(r"<\|channel>thought\s*(.*?)\s*<\|/channel>", re.DOTALL)
```
- Match group 1 = thought text
- Everything after match end = final answer

### Dependencies (for CUDA 12.x environments)
```
torch>=2.4,<2.6  (with +cu121 index for CUDA 12.2)
torchvision>=0.19
transformers>=4.51
accelerate>=1.4
pillow>=10.0
```

## Already Locked on CPU

- Gemma 4 variant enum and config schema
- Manifest structure and preregistered slice files (ruler_mixed_slices, longbench_v2_selection)
- Stable scaffold / preselector / cache proposal interfaces
- Interleaving allocator simulation
- Accounting log schema
- CLI utilities (10+ subcommands)
- 18 passing tests, ruff/pyright clean

## Remaining GPU Tasks

### On RTX PRO 6000 (production hardware)

1. **Gemma 4 E4B bring-up**: load `google/gemma-4-E4B-it`, verify chat template
2. **Latency calibration**: measure real route overhead, signal pass cost, write step cost
3. **Budget grid lock**: run `budget-check` with measured overhead, finalize grid
4. **RULER mini sweep**: first real calibration run with `gpu_handoff.yaml`
5. **Write branch validation**: LoRA adapter update on dependency-heavy slices
6. **Hybrid interior split**: verify allocator produces non-trivial write/cache splits

### Go/No-Go Items Still Pending

- [ ] Route overhead stays within the planned budget rule
- [ ] Hybrid shows plausible interior split behavior on mixed slices
