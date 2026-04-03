import sys
sys.path.insert(0, "/content/wwwc/.venv/lib/python3.12/site-packages")
sys.path.insert(0, "/content/wwwc/src")

import transformers
print(f"transformers: {transformers.__version__}")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "google/gemma-4-E2B-it"
print(f"Loading {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
print(f"Model loaded! Device: {model.device}")
print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# Test chat template
messages = [{"role": "user", "content": "What is 2+2?"}]
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False,
).to(model.device)
print(f"\nChat template test (thinking=False):")
print(f"Input tokens: {inputs.shape[1]}")

with torch.no_grad():
    outputs = model.generate(inputs, max_new_tokens=64)
response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(f"Response: {response}")

# Test with thinking enabled
inputs_think = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=True,
).to(model.device)
print(f"\nChat template test (thinking=True):")
print(f"Input tokens: {inputs_think.shape[1]}")

with torch.no_grad():
    outputs_think = model.generate(inputs_think, max_new_tokens=256)
response_think = tokenizer.decode(outputs_think[0][inputs_think.shape[1]:], skip_special_tokens=False)
print(f"Response (raw): {response_think[:500]}")
