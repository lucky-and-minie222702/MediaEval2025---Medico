import os
import json
from huggingface_hub import InferenceClient
from my_tools import *
import requests

config = MyConfig.load_json(sys.argv[1])
checkpoint = config.get("checkpoint", MyUtils.get_latest_checkpoint(config['dir']))

SYSTEM = (
    "You are a semantic equivalence judge. "
    "Given two sentences, respond with STRICT JSON:"
    "The JSON must have keys: label ('SAME' or 'DIFFERENT'), confidence (0..1)."
)

USER = (
    "Sentence A: {a}\nSentence B: {b}\n"
    "The rules are:"
    "1) SAME if meanings are equivalent for a typical reader."
    "2) DIFFERENT if facts conflict or meaning changes."
    "Respond with JSON ONLY."
)

def judge(a, b):
    url = "https://api.siliconflow.com/v1/chat/completions"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER.format(a = a, b = b)}
    ]
    payload = {
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "messages": messages,
    }
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    print(response.json())
    exit()

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