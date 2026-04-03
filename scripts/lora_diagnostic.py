"""Diagnostic: verify LoRA gradient flow and weight update on language_model."""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

MODEL_ID = "google/gemma-4-E2B-it"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)
proc = AutoProcessor.from_pretrained(MODEL_ID)

# Attach LoRA to language_model only
lang_targets = []
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear) and "language_model" in name and ("q_proj" in name or "v_proj" in name):
        lang_targets.append(name)

print(f"Language model targets: {len(lang_targets)}")
lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.0, target_modules=lang_targets, bias="none")
peft_model = get_peft_model(model, lora_config)
peft_model.enable_input_require_grads()
peft_model.train()

trainable = [(n, p) for n, p in peft_model.named_parameters() if p.requires_grad]
lang_trainable = [(n, p) for n, p in trainable if "language_model" in n]
other_trainable = [(n, p) for n, p in trainable if "language_model" not in n]
print(f"Trainable total: {len(trainable)}, language_model: {len(lang_trainable)}, other: {len(other_trainable)}")
if other_trainable:
    print(f"  WARNING: non-language_model trainable: {other_trainable[0][0]}")
print(f"First lang param: {lang_trainable[0][0]}")

# === Before update ===
first_param_name, first_param = lang_trainable[0]
print(f"\nBEFORE update: {first_param_name} abs_mean={first_param.data.abs().mean().item():.8f}")

# Forward + backward on a document
doc = (
    "The Nexus project lead is Dr. Sarah Chen. "
    "The computational pipeline is called MolDock-X. "
    "Compound ZX-7734 has an IC50 of 23 nanomolar. "
    "The selectivity index against wild-type BRCA2 is 847:1. "
    "Dr. James Liu oversees in-vivo testing. "
    "Patent number US-2024-0847-P was filed. "
    "Morrison & Foerster LLP handles legal review. "
    "Prof. Tanaka from Kyoto University collaborates on crystallography."
)
inputs = proc.tokenizer(doc, return_tensors="pt", truncation=True, max_length=256).to(peft_model.device)
print(f"Input tokens: {inputs['input_ids'].shape[1]}")

# Do 5 steps
lr = 1e-4
for step in range(5):
    outputs = peft_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()

    # Count grads
    n_nonzero = sum(1 for n, p in lang_trainable if p.grad is not None and p.grad.abs().max().item() > 0)
    n_none = sum(1 for n, p in lang_trainable if p.grad is None)

    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if p.requires_grad and p.grad is not None:
                p.data -= lr * p.grad
                p.grad.zero_()

    print(f"  Step {step}: loss={loss.item():.4f}, lang grads nonzero={n_nonzero}/{len(lang_trainable)}, none={n_none}")

# === After update ===
print(f"\nAFTER update: {first_param_name} abs_mean={first_param.data.abs().mean().item():.8f}")

# Check all language_model LoRA params
nonzero_params = 0
for n, p in lang_trainable:
    if p.data.abs().max().item() > 0:
        nonzero_params += 1
print(f"Language model LoRA params with nonzero weights: {nonzero_params}/{len(lang_trainable)}")

# === Test generation ===
peft_model.eval()

# With LoRA state, WITHOUT document in prompt
test_queries = [
    "What is the name of the computational pipeline?",
    "What is the selectivity index against wild-type BRCA2?",
    "Who handles the legal review?",
    "What is the patent number?",
]

print("\n=== Generation test (LoRA active, NO document in prompt) ===")
for q in test_queries:
    messages = [{"role": "user", "content": [{"type": "text", "text": q}]}]
    gen_inputs = proc.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
        enable_thinking=False, tokenize=True, return_dict=True,
    ).to(peft_model.device)
    with torch.no_grad():
        out = peft_model.generate(**gen_inputs, max_new_tokens=64)
    answer = proc.decode(out[0][gen_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  Q: {q}")
    print(f"  A: {answer[:150]}")
    print()
