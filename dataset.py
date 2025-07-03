from torchvision import transforms
from torch.utils.data import Dataset
import torch.utils.data as t_data
from PIL import Image
import pandas as pd
from tqdm import tqdm
from custom_obj import *
import os
from os import path
import joblib

# resnet like norm
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]


def get_img_dict(save_path = None):
    file_ids = os.listdir("data/images")
    file_ids = [path.splitext(s)[0] for s in file_ids if s[-4::] == ".jpg"]
    file_paths = [f"data/images/{s}.jpg" for s in file_ids]

    images = {}

    for file_path, file_id in tqdm(zip(file_paths, file_ids), total = len(file_ids), desc = "Load image"):
        img = Image.open(file_path)
        img = MyImage.change_size(img, (224, 224))
        images[file_id] = img
        
    if save_path is not None:
        joblib.dump(images, save_path)
        
    return images


BASE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406], 
        std = [0.229, 0.224, 0.225]
    ),
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.ColorJitter(
        brightness = 0.2,
        contrast = 0.2,
        saturation = 0.1,
    ),
    transforms.RandomAffine(36),
    BASE_TRANSFORM.transforms,
])


def to_text_data(df):
    return df[["img_id", "question", "answer"]].to_numpy().tolist()


def get_ids(df, current_tokenizer: MyText.MyTokenizer = None, init_tokenizer = True, question_max_length = None, answer_max_length = None):
    tokenizer = current_tokenizer


    ques_texts = df["question"].tolist()
    ques_texts = tokenizer.norm_text(ques_texts)
    ques_texts = tokenizer.remove_stopwords(ques_texts)
    sep = len(ques_texts)
    
    ans_texts = df["answer"].tolist()
    ans_texts = tokenizer.norm_text(ans_texts)
    merge_texts = ques_texts + ans_texts
    
    merge_tokens = tokenizer.preprocess(merge_texts)

    if init_tokenizer:
        tokenizer.fit(merge_tokens)
        tokenizer.add_vocab("<EOS>")


    # question
    ques_tokens = merge_tokens[:sep:]
    
    ques_ids = tokenizer.transform(ques_tokens, post_ids = [tokenizer.get_id("<EOS>")])
    ques_ids = tokenizer.pad_or_trunc(
        ques_ids, 
        max_length = question_max_length,
        padding = "pre",
        truncation = "post"
    )
    
    
    # answer
    ans_tokens = merge_tokens[sep::]
    
    ans_ids = tokenizer.transform(ans_tokens, post_ids = [tokenizer.get_id("<EOS>")])
    ans_ids = tokenizer.pad_or_trunc(
        ans_ids, 
        max_length = answer_max_length,
        padding = "post",
        truncation = "post"
    )
    
    if init_tokenizer:
        return ques_ids, ans_ids, tokenizer
    else:
        return ques_ids, ans_ids
    


class MyDataset(Dataset):
    def __init__(self, img_dict, ques_ids, ans_ids, transform = None):
        super().__init__()
        self.img_dict = img_dict
        self.ques_ids = ques_ids
        self.ans_ids = ans_ids
        self.transform = transform
        
    def __len__(self):
        return len(self.ques_ids)
    
    def __getitem__(self, index):
        img = self.img_dict[self.ids[index][0]]
        question = self.ques_ids[index]
        answer = self.ans_ids[index]
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, question, answer