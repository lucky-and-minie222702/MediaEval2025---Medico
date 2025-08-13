import json
from transformers import AutoModelForCausalLM, AutoTokenizer,  BitsAndBytesConfig
from my_tools import *

config = MyConfig.load_json(sys.argv[1])
checkpoint = config.get("checkpoint", MyUtils.get_latest_checkpoint(config['dir']))

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

bnb_4bit = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype = torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map = "auto",
    trust_remote_code = True,
    
    torch_dtype = torch.bfloat16,
    quantization_config = bnb_4bit,
    low_cpu_mem_usage = True,
)

SYSTEM_PROMPT = (
    "You are a semantic equivalence judge, your task is to evaluate wether the two sentences are the same or different in terms of meaning."
    "You need to provide the label which is SAME or DIFFERENT and your confidence on the answer."
    "Given two sentences, output STRICT JSON with keys: "
    "label (SAME or DIFFERENT), confidence (0.000-1.000)"
)

USER_TEMPLATE = (
    "Sentence A: {a}\nSentence B: {b}\n\n"
    "Rules:\n"
    "1) SAME if their meanings are equivalent in everyday context.\n"
    "2) DIFFERENT if meaning changes or facts conflict.\n"
    "Respond with JSON ONLY, no extra text."
)

def build_prompt(a, b):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(a = a, b = b)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

@torch.inference_mode()
def judge(a, b):
    prompt = build_prompt(a, b)
    inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens = 64,
        temperature = 0.0,
        do_sample = False,
    )

    out = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        print(f"Error: {out}")
        return {"label": "DIFFERENT", "confidence": 0.0}


reader = MyUtils.TestLogger.ResultsReader(
    dir = config["dir"],
    checkpoint = checkpoint,
)

pbar = tqdm(zip(reader.labels, reader.predictions), total = len(reader.labels))
results = {
    "labels": [],
    "confidence": [],
}
for l, p in pbar:
    res = judge(l, p)
    results["labels"].append(1 if res["label"] == "same" else 0)
    results["confidence"].append(res["confidence"])
    
    pbar.set_postfix(
        accuracy = round(np.mean(results["labels"]), 3),
        avg_confidence = round(np.mean(results["confidence"]), 3),
        cur_confidence = round(results["confidence"][-1], 3)
    )

df = pd.read_csv("data/test.csv")
results_df = pd.DataFrame({
    "img_id": df["img_id"],
    "questions": reader.questions,
    
    "labels": reader.labels,
    "predictions": reader.predictions,
    
    "labels": results["labels"],
    "confidence": results["confidence"],

    "complexity": df["complexity"],
    "question_class": df["question_class"],
})

results_df.to_csv(f"results/{config['dir']}/checkpoint-{checkpoint}-llm-judge.csv", index = False)