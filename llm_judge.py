from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json


# load model
model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map = "auto",
    trust_remote_code = True
)

SYSTEM_PROMPT = (
    "You are a semantic equivalence judge. "
    "Given two sentences, output STRICT JSON with keys: "
    "label (SAME or DIFFERENT), confidence (0.00-1.00)."
)

USER_TEMPLATE = (
    "Sentence A: {a}\nSentence B: {b}\n\n"
    "Rules:\n"
    "1) SAME if their meanings are equivalent in everyday context.\n"
    "2) DIFFERENT if meaning changes or facts conflict.\n"
    "Respond with JSON ONLY."
)

def build_prompt(a, b):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(a = a, b = b)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

@torch.inference_mode()
def judge_equivalence(a: str, b: str):
    prompt = build_prompt(a, b)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens = 64,
    )

    out = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens = True).strip()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"label": "DIFFERENT", "confidence": 0.0, "raw": out}
