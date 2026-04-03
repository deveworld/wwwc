# Bring-Up Log

Chronological record of GPU bring-up sessions, decisions, and troubleshooting.

## 2026-04-03: Colab T4 — First Model Load

### Goal
Load Gemma 4 E2B-it, verify chat template with thinking/non-thinking modes.

### Environment
- Google Colab Pro, Tesla T4, 15.6GB VRAM
- Python 3.12.13, torch 2.11.0, transformers 5.5.0
- uv 0.11.3 for dependency management

### Timeline

| Time | Action | Result |
|------|--------|--------|
| 17:06 | `uv sync --extra model --extra dev` | 63 packages installed in 470ms |
| 17:10 | GPU check | CUDA True, Tesla T4, 15.6GB |
| 17:11 | `import torch` | `total_mem` AttributeError (use `total_memory` in torch 2.11) |
| 17:14 | First model load attempt with `google/gemma-4-2b-it` | OSError: not a valid model identifier |
| 17:17 | Web search for correct model ID | Found: `google/gemma-4-E2B-it` (uppercase E2B) |
| 17:18 | HF token setup via Colab Secrets | `notebook_login()` widget didn't render; used `userdata.get("HF_TOKEN")` |
| 17:27 | Model download | 10.2GB safetensors, ~4 min at ~45MB/s |
| 17:28 | First `AutoTokenizer.apply_chat_template` | KeyError: 'shape' — tokenizer doesn't work, need processor |
| 17:31 | Switch to `AutoProcessor` | ImportError: PIL required |
| 17:33 | Add pillow | ImportError: torchvision required |
| 17:35 | Add torchvision, retry | TypeError: string indices must be integers |
| 17:36 | Fix message format to multimodal `[{"type": "text", "text": "..."}]` | **SUCCESS** |
| 17:38 | Full test results | thinking=False: "2+2 is **4**." / thinking=True: `<\|channel>thought` + reasoning |

### Issues Encountered & Resolutions

1. **Model ID naming:** `gemma-4-2b-it` → `gemma-4-E2B-it` (Google uses E2B/E4B notation, not 2b/4b)
2. **Tokenizer vs Processor:** Gemma 4 is multimodal; `AutoTokenizer` alone cannot handle `apply_chat_template`. Must use `AutoProcessor`.
3. **Missing deps:** `AutoProcessor` → `Gemma4Processor` → requires PIL → `Gemma4VideoProcessor` → requires torchvision. Both needed even for text-only inference.
4. **Message format:** Standard `{"role": "user", "content": "text"}` fails. Must use multimodal format: `{"role": "user", "content": [{"type": "text", "text": "..."}]}`.
5. **Colab auto-indent:** `%%writefile` magic captures Colab's auto-indentation, producing IndentationError. Solution: write scripts locally, git push, then run from venv.

### Key Measurements (T4)

| Metric | thinking=False | thinking=True |
|--------|---------------|---------------|
| Input tokens ("What is 2+2?") | 16 | 22 |
| Output tokens | short answer | 256 (max) |
| Thinking overhead (input) | baseline | +6 tokens |

---

## 2026-04-03: A100 MIG 2g-20GB — Go/No-Go

### Goal
Run 4 go/no-go validation checks on production-class hardware.

### Environment
- NVIDIA A100 80GB PCIe MIG 2g.20gb, 20.9GB VRAM, 48GB RAM
- Python 3.10.14, CUDA 12.2, Driver 535.161.07
- torch 2.5.1+cu121 (torch 2.11 incompatible with CUDA 12.2)
- Elice Cloud on-demand instance

### Timeline

| Time | Action | Result |
|------|--------|--------|
| 17:01 | SSH connection, nvidia-smi | A100 MIG confirmed, 20.9GB VRAM |
| 17:02 | `uv sync` | Python 3.14 selected by uv (too new for torch) |
| 17:03 | Recreate venv with `python3.10 -m venv` | Python 3.10.14 |
| 17:04 | Install torch 2.11 | CUDA driver too old error (needs CUDA 13) |
| 17:05 | Install `torch==2.5.1+cu121` from PyTorch index | Success |
| 17:06 | `pip install transformers -e .` | requires-python >=3.11 error |
| 17:07 | Lower requires-python to >=3.10, reinstall | All deps installed |
| 17:08 | CUDA verification | `torch.cuda.is_available()=True` |
| 17:09 | Run go_nogo.py | Model loaded in 53.3s |
| 17:12 | All 4 checks | **PASS** |

### Issues Encountered & Resolutions

1. **uv auto-installs Python 3.14:** On a system with Python 3.10, `uv venv` downloaded Python 3.14 which has no torch wheels. Solution: `python3.10 -m venv .venv` to force system Python.
2. **torch/CUDA version mismatch:** torch 2.11.0 ships with CUDA 13 bindings, incompatible with Driver 535 (CUDA 12.2). Solution: `torch==2.5.1+cu121` from `https://download.pytorch.org/whl/cu121`.
3. **requires-python >=3.11:** Server only has Python 3.10.14. Solution: lowered to `>=3.10` in pyproject.toml.

### Go/No-Go Results

#### Check 1: Scaffold Quality
- **Prompt:** Scaffold describing 3 substrates + query "What are the three substrates?"
- **Answer:** "The three substrates in the TriStore system are write, cache, and the stable scaffold."
- **Latency:** 2123ms for 19 output tokens
- **Verdict:** PASS — model correctly uses scaffold context

#### Check 2: Write vs Stable (Dependency-Heavy)
- **Stable (minimal context):** "Alice knows Bob. Bob knows Carol." → Answer: "Carol" (1-hop)
- **Write (full chain):** 5-person chain → Answer: "Alice,Bob,Carol,Dave,Eve" (full chain)
- **Verdict:** PASS — richer context enables multi-hop reasoning

#### Check 3: Cache Exact Recall
- **No cache:** "What is Lab 7 access code?" → "42068" (hallucination)
- **With cache span:** `[cache] Lab 7 access code: ALPHA-9382-ZULU` → "ALPHA-9382-ZULU" (exact)
- **Verdict:** PASS — cache eliminates hallucination, enables verbatim recall

#### Check 4: Thinking Mode
- **Non-thinking:** 22 input tokens → 21 output tokens, 1236ms
- **Thinking:** 28 input tokens → 256 output tokens, 14908ms
- **Thinking tag:** `<|channel>thought` present with structured reasoning steps
- **Reproducibility:** consistent input tokenization across runs
- **Verdict:** PASS — distinct paths, ~12x latency difference

### Key Measurements (A100 MIG 2g-20GB)

| Metric | Value |
|--------|-------|
| Model load time | 53.3s |
| Non-thinking latency (21 tokens) | 1236ms |
| Thinking latency (256 tokens) | 14908ms |
| Non-thinking throughput | ~17 tok/s |
| Thinking throughput | ~17 tok/s |
| VRAM used (inference) | ~11GB of 20.9GB |

---

## Hardware Compatibility Matrix

| Hardware | CUDA | torch | Status |
|----------|------|-------|--------|
| Colab T4 (15.6GB) | Colab-managed | 2.11.0 | E2B loads, inference works |
| A100 MIG 2g-20GB | 12.2 / Driver 535 | 2.5.1+cu121 | E2B loads, go/no-go PASS |
| RTX PRO 6000 (96GB) | TBD | TBD | Pending — target for full calibration |

## Lessons for Future Environments

1. Always check `nvidia-smi` driver version before installing torch
2. For CUDA 12.x drivers, use `torch==2.5.x+cu121` from PyTorch wheel index
3. Use system Python (`python3.x -m venv`) instead of letting uv pick a version
4. Gemma 4 requires `AutoProcessor`, `pillow`, `torchvision` even for text-only
5. Message format is multimodal: `[{"type": "text", "text": "..."}]`
