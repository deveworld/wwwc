"""LoRA Write Branch Fail-Fast Test v2

v1의 문제: document가 prompt에 다 들어가서 LoRA의 marginal gain이 0 (ceiling effect).

v2 설계:
  - 긴 document (수천 tokens) 사용
  - scaffold는 document의 일부만 커버 (bounded)
  - stable-only: scaffold만 prompt에
  - write-only: scaffold + LoRA가 전체 document chunks를 학습한 adapter state
  - 둘 다 query에 대해 답변

이것이 연구 가설의 정확한 테스트:
  "prompt에 다 안 들어가는 긴 문서에서, LoRA write가 scaffold만으로는 놓치는 정보를 adapter에 저장하여 accuracy를 올리는가?"
"""

import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

MODEL_ID = "google/gemma-4-E2B-it"
OUTPUT_DIR = Path(os.path.expanduser("~/wwwc/artifacts/lora_write_test_v2"))

# ──────────────────────────────────────────────
# Long documents with multi-hop dependency
# The key: scaffold only covers PART of the document.
# The answer requires information spread across the ENTIRE document.
# ──────────────────────────────────────────────

LONG_TESTS = [
    {
        "id": "scattered_facts",
        "chunks": [
            "Report Section 1: The Nexus project was initiated on March 15, 2024. The project lead is Dr. Sarah Chen from the computational biology division. Initial funding was $2.4 million from the NIH grant R01-GM-2024-0847.",
            "Report Section 2: The primary research target is the protein BRCA2-variant-K3289R. This variant was first identified in a Finnish population study published in Nature Genetics (2023). The protein structure was resolved using cryo-EM at 2.8 angstrom resolution.",
            "Report Section 3: Dr. Chen's team developed a novel computational pipeline called MolDock-X. The pipeline uses a combination of molecular dynamics simulation and machine learning-based docking. Runtime on a single A100 GPU is approximately 4.2 hours per protein complex.",
            "Report Section 4: Preliminary results show that compound ZX-7734 binds to BRCA2-K3289R with an IC50 of 23 nanomolar. This is a 15-fold improvement over the previous best compound (IC50 = 345 nM). The binding affinity was validated using surface plasmon resonance.",
            "Report Section 5: Side effect analysis indicates that ZX-7734 has minimal off-target binding. The selectivity index against the wild-type BRCA2 is 847:1. Cytotoxicity tests in HEK293 cells show an LD50 greater than 100 micromolar.",
            "Report Section 6: The next phase involves in-vivo testing in mouse xenograft models. Dr. James Liu from the animal studies facility will oversee this phase. The estimated timeline is 6 months, with an additional budget request of $800,000.",
            "Report Section 7: Intellectual property considerations: a provisional patent (US-2024-0847-P) was filed on June 1, 2024. The patent covers the MolDock-X pipeline and the ZX-7734 compound structure. Legal review by Morrison & Foerster LLP is pending.",
            "Report Section 8: Collaboration with Kyoto University's Prof. Tanaka on crystallography validation is ongoing. Prof. Tanaka's group will attempt to co-crystallize ZX-7734 with BRCA2-K3289R for definitive binding mode confirmation.",
        ],
        # Scaffold only shows sections 1 and 4 (partial coverage)
        "scaffold_indices": [0, 3],
        "queries": [
            {
                "q": "What is the name of the computational pipeline developed in this project, and what is its runtime on a single A100 GPU?",
                "expected": ["moldock-x", "4.2"],
                "info_in": "section 3 (NOT in scaffold)",
            },
            {
                "q": "Who will oversee the in-vivo testing phase, and what is the additional budget requested?",
                "expected": ["james liu", "800,000"],
                "info_in": "section 6 (NOT in scaffold)",
            },
            {
                "q": "What is the selectivity index of ZX-7734 against wild-type BRCA2?",
                "expected": ["847"],
                "info_in": "section 5 (NOT in scaffold)",
            },
            {
                "q": "What is the provisional patent number, and which law firm is handling the legal review?",
                "expected": ["0847", "morrison"],
                "info_in": "section 7 (NOT in scaffold)",
            },
            {
                "q": "Which university professor is collaborating on crystallography validation?",
                "expected": ["tanaka"],
                "info_in": "section 8 (NOT in scaffold)",
            },
        ],
    },
    {
        "id": "chain_reasoning",
        "chunks": [
            "Memo 1: The annual budget for Department Alpha is $5 million. Department Alpha allocates 30% of its budget to Project Mercury.",
            "Memo 2: Project Mercury employs 12 full-time researchers. Each researcher costs $120,000 per year in salary and benefits. The remaining Mercury budget goes to equipment.",
            "Memo 3: The equipment budget for Project Mercury was used to purchase 3 high-performance computing clusters. Each cluster costs $180,000.",
            "Memo 4: Project Mercury's computing clusters process geological survey data from Region Zeta-7. Each cluster can process 500 terabytes of data per month.",
            "Memo 5: Region Zeta-7 contains an estimated 2.3 billion barrels of recoverable oil reserves. The geological confidence level is 78%, based on seismic analysis by GeoTech Solutions Inc.",
            "Memo 6: GeoTech Solutions Inc. was contracted at a rate of $450 per hour. The total contract duration was 2,200 hours. This contract was funded by Department Beta, not Department Alpha.",
        ],
        "scaffold_indices": [0, 1],
        "queries": [
            {
                "q": "How much money does Project Mercury spend on equipment (not salary)?",
                "expected": ["60,000", "60000"],
                "info_in": "requires chain: section 1 (30% of 5M = 1.5M) + section 2 (12 × 120K = 1.44M salary, remainder = 60K equipment). Section 1 in scaffold, section 2 in scaffold.",
            },
            {
                "q": "How many terabytes of data can Project Mercury process per month in total across all its clusters?",
                "expected": ["1500", "1,500"],
                "info_in": "section 3 (3 clusters) + section 4 (500TB each) = 1500TB. Neither in scaffold.",
            },
            {
                "q": "What is the estimated recoverable oil in Region Zeta-7 and what company performed the analysis?",
                "expected": ["2.3", "geotech"],
                "info_in": "section 5. NOT in scaffold.",
            },
            {
                "q": "What was the total cost of the GeoTech Solutions contract, and which department funded it?",
                "expected": ["990,000", "beta"],
                "info_in": "section 6 (450 × 2200 = 990,000, funded by Beta). NOT in scaffold.",
            },
        ],
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
    """Attach LoRA to language_model attention layers ONLY.

    Critical: Gemma 4 is multimodal. q_proj/v_proj exist in vision_tower,
    audio_tower, AND language_model. We must target ONLY language_model,
    otherwise text NTP loss gradient won't reach the LoRA params.
    """
    lang_targets = []
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.Linear)
                and "language_model" in name
                and ("q_proj" in name or "v_proj" in name)):
            lang_targets.append(name)

    if not lang_targets:
        raise RuntimeError("Could not find language_model q_proj/v_proj")

    print(f"Language model LoRA targets: {len(lang_targets)} modules")
    print(f"Sample: {lang_targets[0]}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=lang_targets,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    # Critical: cast LoRA params to float32 to avoid bf16 precision loss
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"LoRA: {trainable/1e6:.1f}M trainable / {total/1e9:.2f}B total ({100*trainable/total:.2f}%)")
    return peft_model


def reset_lora(peft_model):
    for _, param in peft_model.named_parameters():
        if param.requires_grad:
            param.data.zero_()


def lora_write_step(peft_model, processor, chunk_text, lr=2e-4, n_steps=1):
    """1-step (or n-step) LoRA update on a single chunk."""
    peft_model.train()
    peft_model.enable_input_require_grads()

    inputs = processor.tokenizer(
        chunk_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(peft_model.device)

    total_loss = 0.0
    for step in range(n_steps):
        outputs = peft_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        with torch.no_grad():
            for p in peft_model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.data -= lr * p.grad
                    p.grad.zero_()
        total_loss += loss.item()

    peft_model.eval()
    return total_loss / n_steps


def generate_answer(model, processor, prompt, max_new_tokens=128):
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
        enable_thinking=False, tokenize=True, return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    n_input = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][n_input:], skip_special_tokens=True)


def score_answer(answer, expected):
    answer_lower = answer.lower().replace(",", "").replace("-", "").replace("_", "")
    found = sum(1 for e in expected if e.lower() in answer_lower)
    return found / len(expected)


def verify_lora_update(peft_model):
    """Check that LoRA weights are non-zero after update."""
    norms = []
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            norms.append((name.split(".")[-1], param.data.abs().mean().item()))
    return norms[:4]


def run_test(processor, peft_model, n_steps=1, lr=2e-4):
    all_results = []

    for test in LONG_TESTS:
        print(f"\n{'='*70}")
        print(f"Test: {test['id']} | n_steps={n_steps}, lr={lr}")
        print(f"{'='*70}")

        # Build scaffold (only selected chunks)
        scaffold_text = "\n".join(test["chunks"][i] for i in test["scaffold_indices"])
        full_doc = "\n".join(test["chunks"])

        print(f"  Full document: {len(test['chunks'])} chunks, ~{len(full_doc.split())} words")
        print(f"  Scaffold covers: {len(test['scaffold_indices'])}/{len(test['chunks'])} chunks")

        for qi, qdata in enumerate(test["queries"]):
            print(f"\n  Query {qi+1}: {qdata['q'][:80]}...")
            print(f"  Info location: {qdata['info_in'][:80]}...")

            scaffold_prompt = f"[scaffold]\n{scaffold_text}\n\n[query]\n{qdata['q']}\n\nAnswer based on available information. Be precise."

            # ── Condition 1: Stable-only (scaffold in prompt, no LoRA) ──
            reset_lora(peft_model)
            peft_model.eval()
            stable_answer = generate_answer(peft_model, processor, scaffold_prompt)
            stable_score = score_answer(stable_answer, qdata["expected"])

            # ── Condition 2: Write-only (LoRA trained on ALL chunks, scaffold in prompt) ──
            reset_lora(peft_model)
            losses = []
            for chunk in test["chunks"]:
                loss = lora_write_step(peft_model, processor, chunk, lr=lr, n_steps=n_steps)
                losses.append(loss)

            # Verify LoRA weights changed
            lora_norms = verify_lora_update(peft_model)
            write_answer = generate_answer(peft_model, processor, scaffold_prompt)
            write_score = score_answer(write_answer, qdata["expected"])

            # ── Condition 3: Full-doc (everything in prompt, no LoRA, as upper bound) ──
            reset_lora(peft_model)
            peft_model.eval()
            full_prompt = f"[document]\n{full_doc}\n\n[query]\n{qdata['q']}\n\nAnswer precisely."
            full_answer = generate_answer(peft_model, processor, full_prompt, max_new_tokens=128)
            full_score = score_answer(full_answer, qdata["expected"])

            delta = write_score - stable_score
            print(f"    stable={stable_score:.2f}  write={write_score:.2f}  full={full_score:.2f}  Δ(write-stable)={delta:+.2f}")
            print(f"    LoRA norms: {lora_norms[:2]}")
            print(f"    Avg write loss: {sum(losses)/len(losses):.3f}")
            print(f"    Stable: {stable_answer[:150]}")
            print(f"    Write:  {write_answer[:150]}")
            print(f"    Full:   {full_answer[:150]}")

            all_results.append({
                "test_id": test["id"],
                "query_id": qi,
                "info_location": qdata["info_in"][:80],
                "stable_score": stable_score,
                "write_score": write_score,
                "full_score": full_score,
                "delta": delta,
                "avg_loss": sum(losses) / len(losses),
                "stable_answer": stable_answer[:300],
                "write_answer": write_answer[:300],
                "full_answer": full_answer[:300],
            })

    return all_results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("LoRA Write Branch Fail-Fast Test v2")
    print(f"Model: {MODEL_ID}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print("="*70)

    processor, base_model = load_model()
    peft_model = attach_lora(base_model)

    # Run with multiple configs
    configs = [
        {"n_steps": 1, "lr": 2e-4},
        {"n_steps": 3, "lr": 2e-4},
        {"n_steps": 5, "lr": 1e-4},
    ]

    all_config_results = {}
    for cfg in configs:
        label = f"steps={cfg['n_steps']}_lr={cfg['lr']}"
        print(f"\n\n{'#'*70}")
        print(f"CONFIG: {label}")
        print(f"{'#'*70}")
        results = run_test(processor, peft_model, **cfg)
        all_config_results[label] = results

    # ── Final Summary ──
    print("\n\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    for label, results in all_config_results.items():
        stable_avg = sum(r["stable_score"] for r in results) / len(results)
        write_avg = sum(r["write_score"] for r in results) / len(results)
        full_avg = sum(r["full_score"] for r in results) / len(results)
        delta = write_avg - stable_avg
        gap_closed = (write_avg - stable_avg) / max(full_avg - stable_avg, 1e-9)

        print(f"\n  {label}:")
        print(f"    stable={stable_avg:.3f}  write={write_avg:.3f}  full={full_avg:.3f}")
        print(f"    Δ(write-stable)={delta:+.3f}  gap_closed={gap_closed:.1%}")

        if delta >= 0.03:
            v = "GO"
        elif delta > 0:
            v = "MARGINAL"
        else:
            v = "FAIL"
        print(f"    verdict: {v}")

    best_label = max(all_config_results, key=lambda k: sum(r["write_score"] - r["stable_score"] for r in all_config_results[k]))
    best_results = all_config_results[best_label]
    best_delta = sum(r["write_score"] - r["stable_score"] for r in best_results) / len(best_results)
    print(f"\n  Best config: {best_label} (Δ={best_delta:+.3f})")

    if best_delta >= 0.03:
        print(f"\n>>> FINAL VERDICT: GO — write branch works with {best_label}")
    elif best_delta > 0:
        print(f"\n>>> FINAL VERDICT: MARGINAL — small effect with {best_label}, needs more investigation")
    else:
        print(f"\n>>> FINAL VERDICT: FAIL — write branch ineffective across all configs → pivot")

    # Save
    out = {
        "model": MODEL_ID,
        "device": torch.cuda.get_device_name(0),
        "configs": {k: v for k, v in all_config_results.items()},
        "best_config": best_label,
        "best_delta": best_delta,
    }
    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
