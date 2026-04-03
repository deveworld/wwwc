"""LoRA Write Branch Fail-Fast Test v3

v1 bugs: short docs + doc in prompt (ceiling effect)
v2 bugs: LoRA on vision/audio tower + bf16 precision + manual SGD broken
v3 fixes:
  - Language model only LoRA targets
  - float32 LoRA params
  - Adam optimizer (not manual SGD)
  - 20-30 steps per chunk (not 1)
  - Bounded prompt regime (scaffold ≠ full doc)
  - Loss convergence verification
  - Multiple LR/step configs
"""

import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

MODEL_ID = "google/gemma-4-E2B-it"
OUTPUT_DIR = Path(os.path.expanduser("~/wwwc/artifacts/lora_write_test_v3"))

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
        "scaffold_indices": [0, 3],
        "queries": [
            {"q": "What is the name of the computational pipeline developed in this project, and what is its runtime on a single A100 GPU?",
             "expected": ["moldock", "4.2"], "info_in": "section 3 (NOT in scaffold)"},
            {"q": "Who will oversee the in-vivo testing phase, and what is the additional budget requested?",
             "expected": ["james liu", "800"], "info_in": "section 6 (NOT in scaffold)"},
            {"q": "What is the selectivity index of ZX-7734 against wild-type BRCA2?",
             "expected": ["847"], "info_in": "section 5 (NOT in scaffold)"},
            {"q": "What is the provisional patent number, and which law firm is handling the legal review?",
             "expected": ["0847", "morrison"], "info_in": "section 7 (NOT in scaffold)"},
            {"q": "Which university professor is collaborating on crystallography validation?",
             "expected": ["tanaka"], "info_in": "section 8 (NOT in scaffold)"},
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
            {"q": "How many terabytes of data can Project Mercury process per month in total across all its clusters?",
             "expected": ["1500", "1,500"], "info_in": "section 3+4 (NOT in scaffold)"},
            {"q": "What is the estimated recoverable oil in Region Zeta-7 and what company performed the analysis?",
             "expected": ["2.3", "geotech"], "info_in": "section 5 (NOT in scaffold)"},
            {"q": "What was the total cost of the GeoTech Solutions contract, and which department funded it?",
             "expected": ["990", "beta"], "info_in": "section 6 (NOT in scaffold)"},
        ],
    },
]


def load_model():
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16,
    )
    print(f"Loaded in {time.time()-t0:.1f}s, device={model.device}")
    return processor, model


def attach_lora(model):
    lang_targets = [
        n for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and "language_model" in n
        and ("q_proj" in n or "v_proj" in n)
    ]
    if not lang_targets:
        raise RuntimeError("No language_model q/v_proj found")
    print(f"LoRA targets: {len(lang_targets)} language_model modules")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.0, target_modules=lang_targets, bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    # Cast LoRA to fp32 to avoid bf16 precision loss
    for _, p in peft_model.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"LoRA: {trainable/1e6:.1f}M trainable / {total/1e9:.2f}B total ({100*trainable/total:.2f}%)")
    return peft_model


def reset_lora(peft_model):
    """Reset LoRA to fresh initialization state.

    LoRA = x @ A @ B. B is zero-initialized, A is Kaiming-initialized.
    Zeroing both kills gradient flow (output always 0 → grad 0).
    Must re-initialize A with Kaiming and B with zeros.
    """
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if p.requires_grad:
                if "lora_A" in name:
                    torch.nn.init.kaiming_uniform_(p.data, a=5**0.5)
                    p.data = p.data.to(torch.float32)
                elif "lora_B" in name:
                    p.data.zero_()
                else:
                    p.data.zero_()


def lora_write_chunks(peft_model, processor, chunks, n_steps=20, lr=1e-3):
    """Train LoRA on all chunks using Adam optimizer."""
    peft_model.train()
    peft_model.enable_input_require_grads()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, peft_model.parameters()), lr=lr,
    )

    all_losses = []
    for chunk in chunks:
        inputs = processor.tokenizer(
            chunk, return_tensors="pt", truncation=True, max_length=512,
        ).to(peft_model.device)

        for step in range(n_steps):
            optimizer.zero_grad()
            outputs = peft_model(**inputs, labels=inputs["input_ids"])
            outputs.loss.backward()
            optimizer.step()
            all_losses.append(outputs.loss.item())

    peft_model.eval()
    return all_losses


def generate_answer(model, processor, prompt, max_new_tokens=128):
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
        enable_thinking=False, tokenize=True, return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    n = inputs["input_ids"].shape[1]
    return processor.decode(out_ids[0][n:], skip_special_tokens=True)


def score_answer(answer, expected):
    a = answer.lower().replace(",", "").replace("-", "").replace("_", "")
    return sum(1 for e in expected if e.lower() in a) / len(expected)


def verify_lora_state(peft_model):
    norms = []
    for n, p in peft_model.named_parameters():
        if p.requires_grad and "language_model" in n:
            norms.append(p.data.abs().mean().item())
    nonzero = sum(1 for v in norms if v > 1e-8)
    avg = sum(norms) / len(norms) if norms else 0
    return nonzero, len(norms), avg


def run_test(processor, peft_model, n_steps, lr):
    results = []
    for test in LONG_TESTS:
        print(f"\n{'='*70}")
        print(f"Test: {test['id']} | steps_per_chunk={n_steps}, lr={lr}")
        print(f"{'='*70}")

        scaffold_text = "\n".join(test["chunks"][i] for i in test["scaffold_indices"])
        full_doc = "\n".join(test["chunks"])
        all_chunks = test["chunks"]

        print(f"  {len(all_chunks)} chunks, scaffold covers {len(test['scaffold_indices'])}/{len(all_chunks)}")

        for qi, qdata in enumerate(test["queries"]):
            print(f"\n  Q{qi}: {qdata['q'][:70]}...")
            scaffold_prompt = f"[scaffold]\n{scaffold_text}\n\n[query]\n{qdata['q']}\n\nAnswer based on available information. Be precise."
            full_prompt = f"[document]\n{full_doc}\n\n[query]\n{qdata['q']}\n\nAnswer precisely."

            # Condition 1: Stable-only
            reset_lora(peft_model)
            stable_ans = generate_answer(peft_model, processor, scaffold_prompt)
            stable_score = score_answer(stable_ans, qdata["expected"])

            # Condition 2: Write-only (LoRA on all chunks, scaffold in prompt)
            reset_lora(peft_model)
            losses = lora_write_chunks(peft_model, processor, all_chunks, n_steps=n_steps, lr=lr)
            nz, total_p, avg_norm = verify_lora_state(peft_model)
            write_ans = generate_answer(peft_model, processor, scaffold_prompt)
            write_score = score_answer(write_ans, qdata["expected"])

            # Condition 3: Full-doc (upper bound)
            reset_lora(peft_model)
            full_ans = generate_answer(peft_model, processor, full_prompt)
            full_score = score_answer(full_ans, qdata["expected"])

            delta = write_score - stable_score
            loss_start = losses[0] if losses else 0
            loss_end = losses[-1] if losses else 0
            print(f"    s={stable_score:.2f} w={write_score:.2f} f={full_score:.2f} Δ={delta:+.2f}")
            print(f"    loss: {loss_start:.2f}→{loss_end:.4f}  LoRA nonzero: {nz}/{total_p} avg_norm={avg_norm:.6f}")
            print(f"    Stable: {stable_ans[:120]}")
            print(f"    Write:  {write_ans[:120]}")
            print(f"    Full:   {full_ans[:120]}")

            results.append({
                "test_id": test["id"], "query_id": qi,
                "info_in": qdata["info_in"],
                "stable_score": stable_score, "write_score": write_score,
                "full_score": full_score, "delta": delta,
                "loss_start": loss_start, "loss_end": loss_end,
                "lora_nonzero": nz, "lora_total": total_p, "lora_avg_norm": avg_norm,
                "stable_answer": stable_ans[:300],
                "write_answer": write_ans[:300],
                "full_answer": full_ans[:300],
            })
    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("="*70)
    print("LoRA Write Branch Fail-Fast Test v3 (fixed)")
    print(f"Model: {MODEL_ID}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print("="*70)

    processor, base_model = load_model()
    peft_model = attach_lora(base_model)

    configs = [
        {"n_steps": 10, "lr": 1e-3},
        {"n_steps": 20, "lr": 1e-3},
        {"n_steps": 30, "lr": 5e-4},
    ]

    all_results = {}
    for cfg in configs:
        label = f"steps={cfg['n_steps']}_lr={cfg['lr']}"
        print(f"\n\n{'#'*70}")
        print(f"CONFIG: {label}")
        print(f"{'#'*70}")
        results = run_test(processor, peft_model, **cfg)
        all_results[label] = results

    # Summary
    print("\n\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    for label, results in all_results.items():
        sa = sum(r["stable_score"] for r in results) / len(results)
        wa = sum(r["write_score"] for r in results) / len(results)
        fa = sum(r["full_score"] for r in results) / len(results)
        d = wa - sa
        gc = d / max(fa - sa, 1e-9)
        loss_s = sum(r["loss_start"] for r in results) / len(results)
        loss_e = sum(r["loss_end"] for r in results) / len(results)
        print(f"\n  {label}:")
        print(f"    stable={sa:.3f}  write={wa:.3f}  full={fa:.3f}")
        print(f"    Δ={d:+.3f}  gap_closed={gc:.1%}")
        print(f"    avg loss: {loss_s:.2f}→{loss_e:.4f}")
        if d >= 0.03:
            print(f"    >>> GO")
        elif d > 0:
            print(f"    >>> MARGINAL")
        else:
            print(f"    >>> FAIL")

    best = max(all_results, key=lambda k: sum(r["delta"] for r in all_results[k]))
    bd = sum(r["delta"] for r in all_results[best]) / len(all_results[best])
    print(f"\n  Best: {best} (avg Δ={bd:+.3f})")
    if bd >= 0.03:
        print(f"\n>>> FINAL VERDICT: GO")
    elif bd > 0:
        print(f"\n>>> FINAL VERDICT: MARGINAL")
    else:
        print(f"\n>>> FINAL VERDICT: FAIL")

    out = {"model": MODEL_ID, "device": torch.cuda.get_device_name(0),
           "configs": all_results, "best_config": best, "best_delta": bd}
    p = OUTPUT_DIR / "results.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {p}")


if __name__ == "__main__":
    main()
