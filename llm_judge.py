import json
from transformers import AutoModelForCausalLM, AutoTokenizer,  BitsAndBytesConfig
from my_tools import *
import re

config = MyConfig.load_json(sys.argv[1])
checkpoint = config.get("checkpoint", MyUtils.get_latest_checkpoint(config['dir']))

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code = True,
    device_map="auto",  
    dtype = torch.bfloat16,
    low_cpu_mem_usage = True,
)
tokenizer.padding_side = 'left'

def build_adjudicator_prompt(question, model_response, ground_truth, eval_aspects, complexity, atomic_pairs):
    prompt = f"""
### Context
- Endoscopic Image Question: {question}
- Model’s Generated Response: {model_response}
- Ground-Truth Answer: {ground_truth}
- Evaluation Aspects (Clinical Categories): {eval_aspects}
- Complexity Level: {complexity}
- Original Atomic QA Pairs: {atomic_pairs}

### Output
"""
    return prompt


INSTRUCTION = f"""
You are a medical examiner grading an exam response. 
Your task is to systematically evaluate the model's answer with respect to the specified aspects of clinical reasoning.

### Instructions
1. Compare the model's response against the ground-truth.
2. Assign a binary score:
   - 1 = Correct and complete
   - 0 = Incorrect, incomplete, or not addressed
3. Provide a brief justification for your score.

### Output Format
Return your evaluation strictly as structured JSON with the following format:
{{
    "score": 0 or 1,
    "justification": "<short explanation>"
}}
No extra text.
"""

def build_prompt(question, model_response, ground_truth, eval_aspects, complexity, atomic_pairs):
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": build_adjudicator_prompt(question, model_response, ground_truth, eval_aspects, complexity, atomic_pairs)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)


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
        return None


@torch.inference_mode()
def judge_batch(prompts):
    inputs = tokenizer(
        prompts,
        return_tensors = "pt",
        padding = True,
        truncation = False,
    )
    inputs = {k: v.to("cuda:1") for k, v in inputs.items()}

    gen_ids = model.generate(
        **inputs,
        max_new_tokens = 512,
    )

    outs = []
    for i in range(len(gen_ids)):
        gen_slice = gen_ids[i]
        text = tokenizer.decode(gen_slice, skip_special_tokens = True).strip()
        print(text)
        outs.append(parse_json_safe(text))
    return outs


df = pd.read_csv("data/test.csv")

reader = MyUtils.TestLogger.ResultsReader(
    dir = config["dir"],
    checkpoint = checkpoint,
)

labels = list(reader.labels)
preds  = list(reader.predictions)

n_samples = len(labels)
batch_size = config["batch_size"]

pbar = tqdm(range(0, n_samples, batch_size), total = (n_samples + batch_size - 1) // batch_size)
results = {"score": [], "justification": []}

for start in pbar:
    end = min(start + batch_size, n_samples)
    batch = list(zip(
        df["question"][start:end:], 
        preds[start:end:], 
        labels[start:end:], 
        df["question_class"][start:end:],
        df["complexity"][start:end:],
        df["original"][start:end:],
    ))

    batch_res = judge_batch([build_prompt(*x) for x in batch])

    for res in batch_res:
        label = int(res["score"])

    pbar.set_postfix(
        accuracy = round(float(np.mean(results["score"])), 3),
    )
    

results_df = pd.DataFrame({
    "img_id": df["img_id"],
    "question": reader.questions,
    
    "label": reader.labels,
    "prediction": reader.predictions,
    
    "scores": results["score"],
    "justification": results["justification"],

    "complexity": df["complexity"],
    "question_class": df["question_class"],
})

results_df.to_csv(f"results/{config['dir']}/checkpoint-{checkpoint}-llm-judge.csv", index = False)