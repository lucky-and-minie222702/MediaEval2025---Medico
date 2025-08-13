import json
from transformers import AutoModelForCausalLM, AutoTokenizer,  BitsAndBytesConfig
from my_tools import *
import re

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
tokenizer.padding_side = "left"

SYSTEM_PROMPT = (
    "You are a semantic equivalence judge."
    "Given two medical sentences, output STRICT JSON with keys: "
    "label (SAME or DIFFERENT), confidence (0.00 - 1.00)"
)

USER_TEMPLATE = (
    "Sentence A: {a}\nSentence B: {b}\n\n"
    "Rules:\n"
    "1) SAME if their meanings are equivalent, do not be strictly with extra information.\n"
    "2) DIFFERENT only if meaning changes or facts conflict.\n"
    "Respond with JSON ONLY, no extra text."
)


def build_prompt(a, b):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(a = a, b = b)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

def build_prompts_batch(batch_pairs):
    return [build_prompt(a, b) for a, b in batch_pairs]


def parse_json_safe(text):
    _json_re = re.compile(r"\{.*\}", re.DOTALL)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _json_re.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {"label": "DIFFERENT", "confidence": 0.0}


@torch.inference_mode()
def judge_batch(pairs):
    prompts = build_prompts_batch(pairs)
    inputs = tokenizer(
        prompts,
        return_tensors = "pt",
        padding = True,
        truncation = False,
    ).to(model.device)

    # Generate batched
    gen_ids = model.generate(
        **inputs,
        max_new_tokens = 64,
    )

    input_lens = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim = 1)
    outs = []
    for i in range(gen_ids.size(0)):
        gen_slice = gen_ids[i, input_lens[i]:]
        text = tokenizer.decode(gen_slice, skip_special_tokens = True).strip()
        outs.append(parse_json_safe(text))
    return outs


reader = MyUtils.TestLogger.ResultsReader(
    dir = config["dir"],
    checkpoint = checkpoint,
)

labels = list(reader.labels)
preds  = list(reader.predictions)
n_samples = len(labels)

batch_size = config["batch_size"]

pbar = tqdm(range(0, n_samples, batch_size), total = (n_samples + batch_size - 1) // batch_size)
results = {"labels": [], "confidence": []}

for start in pbar:
    end = min(start + batch_size, n_samples)
    batch_pairs = list(zip(labels[start:end], preds[start:end]))

    batch_res = judge_batch(batch_pairs)

    for res in batch_res:
        label = res["label"].strip().lower()
        conf = res["confidence"]

        results["labels"].append(1 if label == "same" else 0)
        results["confidence"].append(conf)

    pbar.set_postfix(
        accuracy = round(float(np.mean(results["labels"])), 3),
        avg_confidence = round(float(np.mean(results["confidence"])), 2),
        cur_confidence = round(results["confidence"][-1], 2)
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