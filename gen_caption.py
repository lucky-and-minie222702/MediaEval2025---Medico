import sys
import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from my_tools import *


# load config
config = MyConfig.load_json(sys.argv[1])


# load data
df = pd.read_csv("data/train.csv")
org = df["original"].apply(json.loads)


# question id mapping
question_set = set()
for pairs in org:
    for pair in pairs:
        question_set.add(pair["q"])

question_dict = dict(enumerate(question_set))
question_dict.update({v: k for k, v in question_dict.items()})

def to_ids_list(pairs):
    return sorted([question_dict[p["q"]] for p in pairs])

qid_lists = [to_ids_list(pairs) for pairs in org]
qid_sets  = [frozenset(lst) for lst in qid_lists] 

answers = df["answer"].astype(str).tolist()
N = len(answers)

n_informative = int(config["n_informative"])
n_caption = int(config["n_caption"])
base_seed = int(config.get("seed", 42))


# build captions for one row
def make_one_caption(idx, qid_sets, answers, rng):
    ids = qid_sets[idx]
    out_sentences = []

    for _ in range(n_informative):
        tries = 0
        while True:
            j = int(rng.integers(0, N))
            if j != idx and qid_sets[j].isdisjoint(ids):
                cap = answers[j].strip()
                if not cap.endswith("."):
                    cap += "."
                out_sentences.append(cap)
                ids = qid_sets[j]
                break
            tries += 1
            if tries > 5_000:
                for k in range(N):
                    if k != idx and qid_sets[k].isdisjoint(ids):
                        cap = answers[k].strip()
                        if not cap.endswith("."):
                            cap += "."
                        out_sentences.append(cap)
                        ids = qid_sets[k]
                        break
                else:
                    out_sentences.append("")
                break

    return " ".join(out_sentences).strip()

def process_row(idx, qid_sets, answers, n_caption, base_seed):
    rng = np.random.default_rng((base_seed * 1_000_003) ^ idx)
    return [make_one_caption(idx, qid_sets, answers, rng) for _ in range(n_caption)]


# parallel execution
def main():
    try:
        import os
        workers = int(config.get("num_workers", os.cpu_count() or 1))
    except Exception:
        workers = None

    captions = [None] * N
    fn = partial(
        process_row, 
        qid_sets = qid_sets, 
        answers = answers,
        n_caption = n_caption, 
        base_seed = base_seed)

    with ProcessPoolExecutor(max_worker = workers) as ex:
        futures = {ex.submit(fn, i): i for i in range(N)}
        for f in tqdm(as_completed(futures), total = N, desc = "Generating caption (parallel)"):
            i = futures[f]
            captions[i] = f.result()

    df_out = df.copy()
    df_out["qid"] = qid_lists
    df_out["caption"] = captions
    df_out.to_csv("data/train_with_caption.csv", index=False)


main()
