import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from my_tools import *

config = MyConfig.load_json(sys.argv[1])
checkpoint = config.get("checkpoint", MyUtils.get_latest_checkpoint(config['dir']))

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

SYSTEM_PROMPT = (
    "You are a semantic equivalence judge, your task is to evaluate wether the two sentences is the same or different in terms of meaning, aslo you need to give brief reason for your decision"
    "Given two sentences, output STRICT JSON with keys: "
    "label (SAME or DIFFERENT), confidence (0.00-1.00), reason (your decision's reason)"
)

USER_TEMPLATE = (
    "Sentence A: {a}\nSentence B: {b}\n\n"
    "Rules:\n"
    "1) SAME if their meanings are equivalent in everyday context.\n"
    "2) DIFFERENT if meaning changes or facts conflict.\n"
    "3) Give a short reason why you give that answer.\n"
    "Respond with JSON ONLY, no extra text"
)

def build_prompt(a, b):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(a=a, b=b)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

@torch.inference_mode()
def judge(a: str, b: str):
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
        return {"label": "DIFFERENT", "confidence": 0.0}
    
print(judge("i have two apples", "i have two ranges"))

# reader = MyUtils.TestLogger.ResultsReader(
#     dir = config["dir"],
#     checkpoint = checkpoint,
# )

# pbar = tqdm(zip(reader.labels, reader.predictions), total = len(reader.labels))
# results = {
#     "labels": [],
#     "confidence": []
# }
# for l, p in pbar:
#     res = judge(l, p)
#     results["labels"].append(1 if res["label"] == "same" else 0)
#     results["confidence"].append(res["confidence"])
    
#     pbar.set_postfix(
#         accuracy = round(np.mean(results["labels"]), 4),
#         avg_confidence = round(np.mean(results["confidence"]), 4),
#         cur_confidence = round(results["confidence"][-1], 4)
#     )

# df = pd.read_csv("data/test.csv")
# results_df = pd.DataFrame({
#     "img_id": df["img_id"],
#     "questions": reader.questions,
    
#     "labels": reader.labels,
#     "predictions": reader.predictions,
    
#     "labels": results["labels"],
#     "confidence": results["confidence"],

#     "complexity": df["complexity"],
#     "question_class": df["question_class"],
# })

# results_df.to_csv(f"results/{config['dir']}/checkpoint-{checkpoint}-llm-judge.csv", index = False)