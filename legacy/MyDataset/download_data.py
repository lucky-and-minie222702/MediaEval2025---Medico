import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
import os


os.makedirs(f"data", exist_ok  =True)
os.makedirs(f"data/images", exist_ok = True)


ds = load_dataset("SimulaMet/Kvasir-VQA-x1")

df = ds['train'].to_pandas()
df.to_csv(f"data/train.csv", index=False)

df = ds['test'].to_pandas()
df.to_csv(f"data/test.csv", index=False)


ds = load_dataset("SimulaMet-HOST/Kvasir-VQA")

df = ds['raw'].select_columns(['source', 'question', 'answer', 'img_id']).to_pandas()
for i, row in df.groupby('img_id').nth(0).iterrows():
    image = ds['raw'][i]['image'].save(f"data/images/{row['img_id']}.jpg")