from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from my_tools import *
import os
from os import path
from sklearn.model_selection import train_test_split


def get_img_dict():
    file_ids = os.listdir("data/images")
    file_ids = [path.splitext(s)[0] for s in file_ids if s[-4::] == ".jpg"]
    file_paths = [f"data/images/{s}.jpg" for s in file_ids]

    images = dict({})

    for file_path, file_id in zip(file_paths, file_ids):
        images[file_id] = file_path
        
    return images


BASE_TRANSFORM = transforms.Compose([
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.ColorJitter(
        brightness = 0.1,
        contrast = 0.1,
        saturation = 0.025,
    ),
    transforms.RandomAffine(18),
    *BASE_TRANSFORM.transforms,
])


def norm_text(text):
    out = text.lower()
    
    out = out.replace("?", "")
    out = out.replace("/", " ")
    out = out.replace(".", ",")
    out = out.replace(";", ",")
    
    return out


def preprocess(processor, d, max_length, include_answer = True, mask_answer = -100, img_dict = None, transform = None):
    if img_dict is None:
        img_dict = get_img_dict()

    image = Image.open(img_dict[d["img_id"]])
    image = MyImage.change_size(image, (224, 224))

    if transform is not None:
        image = transform(image)
    
    quest = f"Question: {norm_text(d['question'])}"
    
    inputs = processor(
        images = image,
        text = quest,
        return_tensors = "pt",
        max_length = max_length[0],
        padding = "max_length",
        truncation = True,
    )
    
    if include_answer:
        inputs["labels"] = processor.tokenizer(
            norm_text(d["answer"]), 
            return_tensors = "pt",
            max_length = max_length[1],
            padding = "max_length",
            truncation = True,
        )["input_ids"]
        if mask_answer is not None:
            inputs["labels"][inputs["labels"] == processor.tokenizer.pad_token_id] = mask_answer
        
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    
    return inputs


class MyDataset(Dataset):
    def __init__(self, df, max_question_legnth, max_answer_length, processor, transform = None, mask_answer = -100):
        super().__init__()

        self.max_length = [max_question_legnth, max_answer_length]
        self.processor = processor
        self.transform = transform
        self.data = df.to_dict(orient = 'records')
        self.img_dict = get_img_dict()
        self.mask_answer = mask_answer
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return preprocess(
            self.processor, self.data[index], self.max_length, 
            img_dict = self.img_dict, 
            mask_answer = self.mask_answer,
            transform = self.transform
        )
    
    
def load_data(processor, max_question_length, max_answer_length, train_ratio = None, train_complexities = [1, 2, 3], test_complexities = [1, 2, 3], test_only = False, seed = 22022009):
    def invalid_char(texts):
        not_good = lambda x: sum([ord(c) > 255 for c in x]) > 0
        invalid_idx = [i for i, s in enumerate(texts) if not_good(s)]
        return invalid_idx

    def drop_invalid_char_df(df):
        df.drop(index = invalid_char(df["question"].tolist()), inplace = True)
        df.drop(index = invalid_char(df["answer"].tolist()), inplace = True)
    
    if test_only:
        test_df = pd.read_csv("data/test.csv")
        drop_invalid_char_df(test_df)
        mask = test_df["complexity"].map(lambda x: x in test_complexities)
        test_df = test_df[mask]
        test_ds = MyDataset(test_df, max_question_length, max_answer_length, processor, BASE_TRANSFORM)
        return test_ds

    # load df
    df = pd.read_csv("data/train.csv")
    drop_invalid_char_df(df)
    mask = df["complexity"].map(lambda x: x in train_complexities)
    df = df[mask]

    train_df, val_df = train_test_split(df, train_size = train_ratio, shuffle = True, random_state = seed)

    train_ds = MyDataset(train_df, max_question_length, max_answer_length, processor, TRAIN_TRANSFORM)
    val_ds = MyDataset(val_df, max_question_length, max_answer_length, processor, BASE_TRANSFORM)
    
    return train_ds, val_ds