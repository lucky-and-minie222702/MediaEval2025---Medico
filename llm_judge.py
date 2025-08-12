import os
import json
from huggingface_hub import InferenceClient
from my_tools import *

config = MyConfig.load_json(sys.argv[1])
checkpoint = config.get("checkpoint", MyUtils.get_latest_checkpoint(config['dir']))

client = InferenceClient(
    api_key = config["api_key"],
    provider = "hf-inference",
)

SYSTEM = (
    "You are a semantic equivalence judge. "
    "Given two sentences, respond with STRICT JSON: "
    '{"label": "SAME" or "DIFFERENT", "confidence": float number 0..1}. '
    "No extra text."
)

def judge(a, b):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
         f"Sentence A: {a}\nSentence B: {b}\nRules:\n"
         "1) SAME if meanings are equivalent for a typical reader.\n"
         "2) DIFFERENT if facts conflict or meaning changes.\n"
         "Respond with JSON ONLY."}
    ]
    out = client.chat.completions.create(
        model = config["model_name"],
        messages = messages,
        temperature = 0.0,
        max_tokens = 64,
        response_format = {"type": "json_object"}
    )
    text = out.choices[0].message.content
    s, e = text.find("{"), text.rfind("}")
    payload = text[s:e+1] if s != -1 and e != -1 else '{"label":"DIFFERENT","confidence":0.0}'
    payload = payload.lower()
    return json.loads(payload)

reader = MyUtils.TestLogger.ResultsReader(
    dir = config["dir"],
    checkpoint = checkpoint,
)

pbar = tqdm(zip(reader.labels, reader.predictions), total = len(reader.labels))
results = {
    "labels": [],
    "confidence": []
}
for l, p in pbar:
    res = judge(l, p)
    results["labels"].append(1 if res["label"] == "same" else 0)
    results["confidence"].append(res["confidence"])
    
    pbar.set_postfix(
        accuracy = round(np.mean(results["labels"]), 4),
        avg_confidence = round(np.mean(results["confidence"]), 4),
        cur_confidence = round(results["confidence"][-1], 4)
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