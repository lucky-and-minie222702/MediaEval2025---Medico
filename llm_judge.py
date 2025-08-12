import os
import json
from huggingface_hub import InferenceClient
from my_tools import *

config = MyConfig.load_json(sys.argv[1])

client = InferenceClient(
    api_key = config["api_key"],
    provider = "auto",
)

SYSTEM = (
    "You are a semantic equivalence judge. "
    "Given two sentences, respond with STRICT JSON: "
    '{"label": "SAME" or "DIFFERENT", "confidence": float number 0..1}. '
    "No extra text."
)

def judge(a: str, b: str):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
         f"Sentence A: {a}\nSentence B: {b}\nRules:\n"
         "1) SAME if meanings are equivalent for a typical reader.\n"
         "2) DIFFERENT if facts conflict or meaning changes.\n"
         "Respond with JSON ONLY."}
    ]
    out = client.chat.completions.create(
        model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        messages = messages,
        temperature = 0.0,
        max_tokens = 64,
        response_format = {"type": "json_object"}
    )
    text = out.choices[0].message.content
    s, e = text.find("{"), text.rfind("}")
    payload = text[s:e+1] if s != -1 and e != -1 else '{"label":"DIFFERENT","confidence":0.0}'
    print(text)
    return json.loads(payload)

print(judge("The meeting starts at 3 pm.", "The meeting begins at 3:00 in the afternoon."))
