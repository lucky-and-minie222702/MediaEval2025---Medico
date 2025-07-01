import numpy as np
import cv2
import os
import pandas as pd
from custom_obj import *
from tqdm import tqdm
import joblib


os.makedirs(f"data/converted", exist_ok = True)


file_ids = pd.read_csv("data/train.csv")["img_id"].unique().tolist()
file_paths = [f"data/images/{s}.jpg" for s in file_ids]

images = []

for file_path, file_id in tqdm(zip(file_paths, file_ids)):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = MyImage.change_size(img, (224, 224))
    
    images.append((file_id, img))
    
joblib.dump(images, "data/converted/images.joblib")
