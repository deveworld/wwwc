"""LoRA Write Branch Fail-Fast Test

1-step LoRA inference-time update가 Gemma 4 E2B에서 실제로 작동하는지 검증.
dependency-heavy task에서 write-only > stable-only 인지 확인.

판정 기준:
  - write-only > stable-only by ≥3%: GO
  - 0-3%: 약한 효과, 추가 조사 필요
  - write-only ≤ stable-only: FAIL → pivot
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

MODEL_ID = "google/gemma-4-E2B-it"
DEVICE = "cuda:0"
OUTPUT_DIR = Path(os.path.expanduser("~/wwwc/artifacts/lora_write_test"))

# ──────────────────────────────────────────────
# Test data: dependency-heavy multi-hop chains
# ──────────────────────────────────────────────

DEPENDENCY_TESTS = [
    {
        "id": "chain_5",
        "document": (
            "Alice works at company X. Bob is Alice's manager at company X. "
            "Carol is Bob's supervisor. Dave is Carol's department head. "
            "Eve is the CEO who oversees Dave's department. "
            "The company policy states that budget approvals must go through "
            "the direct chain of command from requester to CEO."
        ),
        "query": "If Alice wants budget approval, list every person in the approval chain from Alice to the CEO, in order.",
        "expected_names": ["alice", "bob", "carol", "dave", "eve"],
    },
    {
        "id": "chain_transfer",
        "document": (
            "Server A sends data to Server B every hour. "
            "Server B processes the data and forwards results to Server C. "
            "Server C aggregates results and sends a summary to Server D. "
            "Server D stores the summary in Database E. "
            "If Server A goes offline, all downstream servers lose their input."
        ),
        "query": "Trace the complete data flow path from origin to storage. List each node in order.",
        "expected_names": ["a", "b", "c", "d", "e"],
    },
    {
        "id": "chain_7",
        "document": (
            "In the relay race, Runner 1 passes the baton to Runner 2. "
            "Runner 2 hands it to Runner 3. Runner 3 gives it to Runner 4. "
            "Runner 4 passes to Runner 5. Runner 5 hands to Runner 6. "
            "Runner 6 gives the final pass to Runner 7 who crosses the finish line. "
            "Each runner must complete their 100m segment before passing."
        ),
        "query": "List all runners in the order they receive the baton, from first to last.",
        "expected_names": ["1", "2", "3", "4", "5", "6", "7"],
    },
    {
        "id": "cause_effect",
        "document": (
            "The drought reduced crop yields by 40%. Lower crop yields caused "
            "food prices to increase by 25%. Higher food prices led to a 15% "
            "decrease in consumer spending on non-essentials. Reduced consumer "
            "spending triggered layoffs in the retail sector. Retail layoffs "
            "increased the unemployment rate by 3 percentage points. Higher "
            "unemployment led to increased government welfare spending."
        ),
        "query": "Starting from the drought, list each step in the causal chain that leads to increased government welfare spending.",
        "expected_names": ["drought", "crop", "food price", "consumer", "layoff", "unemployment", "welfare"],
    },
    {
        "id": "inheritance",
        "document": (
            "Class Vehicle has properties: wheels, engine. "
            "Class Car extends Vehicle and adds: doors, trunk. "
            "Class SportsCar extends Car and adds: turbo, spoiler. "
            "Class RaceCar extends SportsCar and adds: rollCage, fireExtinguisher. "
            "Each class inherits all properties from its parent classes."
        ),
        "query": "List ALL properties that a RaceCar object has, including inherited ones.",
        "expected_names": ["wheels", "engine", "doors", "trunk", "turbo", "spoiler", "rollcage", "fireextinguisher"],
    },
]


def load_model():
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print(f"Loaded in {time.time()-t0:.1f}s, device={model.device}")
    return processor, model


def attach_lora(model):
    """Attach LoRA adapters to the model for inference-time training.

    Gemma 4 uses Gemma4ClippableLinear wrappers. We target the inner
    Linear modules via 'q_proj.linear' and 'v_proj.linear' paths.
    If that fails, fall back to matching all nn.Linear with 'q_proj' or 'v_proj' in name.
    """
    # First, discover what module names exist
    linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ("q_proj" in name or "v_proj" in name):
            linear_names.append(name)

    if not linear_names:
        raise RuntimeError("Could not find q_proj/v_proj Linear modules in model")

    # Deduce pattern: e.g. "model.layers.0.self_attn.q_proj.linear"
    # We need the suffix pattern that PEFT can match
    sample = linear_names[0]
    print(f"Sample target module path: {sample}")

    # Try with explicit '.linear' suffix if Gemma4ClippableLinear wraps Linear
    target_modules = ["q_proj.linear", "v_proj.linear"]
    # Verify these paths exist
    found = any(t in name for name in linear_names for t in target_modules)
    if not found:
        # Fallback: use exact discovered names
        target_modules = list(set(
            name.split(".")[-2] + "." + name.split(".")[-1]
            for name in linear_names
        ))
    print(f"LoRA target modules: {target_modules[:4]}...")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"LoRA attached: {trainable/1e6:.1f}M trainable / {total/1e9:.2f}B total ({100*trainable/total:.2f}%)")
    return peft_model


def lora_write_step(peft_model, processor, document_text, lr=1e-4):
    """Perform 1-step LoRA update on document chunks.

    This simulates the 'write' operation: the model learns from the document
    by doing a single gradient step on self-supervised NTP loss.
    """
    peft_model.train()
    peft_model.enable_input_require_grads()  # Required for gradient flow with device_map

    # Tokenize document as a single chunk for simplicity
    inputs = processor.tokenizer(
        document_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(peft_model.device)

    # Forward pass: self-supervised NTP loss
    outputs = peft_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

    # 1-step gradient update
    loss.backward()
    with torch.no_grad():
        for p in peft_model.parameters():
            if p.requires_grad and p.grad is not None:
                p.data -= lr * p.grad
                p.grad.zero_()

    peft_model.eval()
    return loss.item()


def generate_answer(model, processor, query_text, max_new_tokens=128):
    """Generate an answer using the model (with or without LoRA state)."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": query_text}]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    n_input = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][n_input:], skip_special_tokens=True)


def score_answer(answer, expected_names):
    """Score: fraction of expected names found in the answer."""
    answer_lower = answer.lower().replace("-", "").replace("_", "")
    found = sum(1 for name in expected_names if name.lower() in answer_lower)
    return found / len(expected_names)


def reset_lora(peft_model):
    """Reset LoRA weights to zero (fresh adapter)."""
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            param.data.zero_()


def run_test(processor, base_model, peft_model):
    results = []

    for test in DEPENDENCY_TESTS:
        print(f"\n{'='*60}")
        print(f"Test: {test['id']}")
        print(f"{'='*60}")

        prompt_with_doc = f"[document]\n{test['document']}\n\n[query]\n{test['query']}\n\nAnswer precisely and completely."

        # ── Stable-only: no LoRA, just ask ──
        reset_lora(peft_model)
        peft_model.eval()
        stable_answer = generate_answer(peft_model, processor, prompt_with_doc)
        stable_score = score_answer(stable_answer, test["expected_names"])
        print(f"Stable-only: score={stable_score:.2f}")
        print(f"  Answer: {stable_answer[:200]}")

        # ── Write-only: LoRA 1-step on document, then ask ──
        reset_lora(peft_model)
        loss = lora_write_step(peft_model, processor, test["document"])
        write_answer = generate_answer(peft_model, processor, prompt_with_doc)
        write_score = score_answer(write_answer, test["expected_names"])
        print(f"Write-only: score={write_score:.2f}, write_loss={loss:.4f}")
        print(f"  Answer: {write_answer[:200]}")

        # ── Write-only WITHOUT document in prompt: pure adapter recall ──
        prompt_no_doc = f"[query]\n{test['query']}\n\nAnswer precisely and completely based on what you know."
        write_nodoc_answer = generate_answer(peft_model, processor, prompt_no_doc)
        write_nodoc_score = score_answer(write_nodoc_answer, test["expected_names"])
        print(f"Write-only (no doc in prompt): score={write_nodoc_score:.2f}")
        print(f"  Answer: {write_nodoc_answer[:200]}")

        results.append({
            "id": test["id"],
            "stable_score": stable_score,
            "write_score": write_score,
            "write_nodoc_score": write_nodoc_score,
            "write_loss": loss,
            "stable_answer": stable_answer[:300],
            "write_answer": write_answer[:300],
            "write_nodoc_answer": write_nodoc_answer[:300],
        })

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("LoRA Write Branch Fail-Fast Test")
    print(f"Model: {MODEL_ID}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print("="*60)

    processor, base_model = load_model()
    peft_model = attach_lora(base_model)

    results = run_test(processor, base_model, peft_model)

    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    stable_avg = sum(r["stable_score"] for r in results) / len(results)
    write_avg = sum(r["write_score"] for r in results) / len(results)
    write_nodoc_avg = sum(r["write_nodoc_score"] for r in results) / len(results)
    delta = write_avg - stable_avg

    for r in results:
        d = r["write_score"] - r["stable_score"]
        print(f"  {r['id']:20s}  stable={r['stable_score']:.2f}  write={r['write_score']:.2f}  write_nodoc={r['write_nodoc_score']:.2f}  Δ={d:+.2f}")

    print(f"\n  Average: stable={stable_avg:.3f}  write={write_avg:.3f}  write_nodoc={write_nodoc_avg:.3f}")
    print(f"  Delta (write - stable): {delta:+.3f}")

    if delta >= 0.03:
        verdict = "GO — write branch shows meaningful improvement"
    elif delta > 0:
        verdict = "MARGINAL — small positive effect, investigate further"
    else:
        verdict = "FAIL — write branch does not improve over stable-only → pivot"

    print(f"\n>>> VERDICT: {verdict}")

    # Also check: does LoRA retain info WITHOUT document in prompt?
    print(f"\n  Write-nodoc avg: {write_nodoc_avg:.3f} (adapter-only recall, no document)")
    if write_nodoc_avg > 0.1:
        print("  → LoRA adapter retains some document information (parametric write works)")
    else:
        print("  → LoRA adapter retains little document information")

    # Save
    out = {
        "model": MODEL_ID,
        "device": torch.cuda.get_device_name(0),
        "stable_avg": stable_avg,
        "write_avg": write_avg,
        "write_nodoc_avg": write_nodoc_avg,
        "delta": delta,
        "verdict": verdict,
        "results": results,
    }
    out_path = OUTPUT_DIR / "lora_write_test_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
