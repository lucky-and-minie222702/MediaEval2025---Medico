from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
from custom_obj import *
import os
from os import path
import torch

# resnet like norm
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]


def get_img_dict():
    file_ids = os.listdir("data/images")
    file_ids = [path.splitext(s)[0] for s in file_ids if s[-4::] == ".jpg"]
    file_paths = [f"data/images/{s}.jpg" for s in file_ids]

    images = dict({})

    for file_path, file_id in zip(file_paths, file_ids):
        images[file_id] = file_path
        
    return images


BASE_TRANSFORM = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Normalize(
    #     mean = [0.485, 0.456, 0.406], 
    #     std = [0.229, 0.224, 0.225]
    # ),
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.ColorJitter(
        brightness = 0.2,
        contrast = 0.2,
        saturation = 0.1,
    ),
    transforms.RandomAffine(36),
    *BASE_TRANSFORM.transforms,
])


def norm_text(text):
    out = text.lower()
    
    out = out.replace("?", "")
    out = out.replace("/", " ")
    out = out.replace(".", ",")
    out = out.replace(";", ",")
    
    return out


def preprocess(processor, d, max_length, include_answer = True, img_dict = None, transform = None):
    if img_dict is None:
        img_dict = get_img_dict()

    image = Image.open(img_dict[d["img_id"]])
    image = MyImage.change_size(image, (384, 384))

    if transform is not None:
        image = transform(image)
    
    inputs = processor(
        image, norm_text(d["question"]), 
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
        inputs["labels"][inputs["labels"] == processor.tokenizer.pad_token_id] = -100
    
    return {k: v.squeeze(0) for k, v in inputs.items()}


class MyDataset(Dataset):
    def __init__(self, df, max_question_legnth, max_answer_length, processor, transform = None):
        super().__init__()

        self.max_length = [max_question_legnth, max_answer_length]
        self.processor = processor
        self.transform = transform
        self.data = df.to_dict(orient = 'records')
        self.img_dict = get_img_dict()
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return preprocess(self.processor, self.data[index], self.max_length, 
                          img_dict = self.img_dict, transform = self.transform)
    
    
# train: 114_868
# question_max_length = 20,  # 114_729 in train
# answer_max_length = 40,  # 113_865 in train
def load_data(processor, batch_size = 32, max_question_length = 20,  max_answer_length = 40):
    # Tu nhien co tieng Trung
    def invalid_char(texts):
        not_good = lambda x: sum([ord(c) > 255 for c in x]) > 0
        invalid_idx = [i for i, s in enumerate(texts) if not_good(s)]
        return invalid_idx

    def drop_invalid_char_df(df):
        df.drop(index = invalid_char(df["question"].tolist()), inplace = True)
        df.drop(index = invalid_char(df["answer"].tolist()), inplace = True)

    # load df
    train_df = pd.read_csv("data/train.csv")
    drop_invalid_char_df(train_df)
    train_size = int(len(train_df) * 0.9)
    val_df = train_df.iloc[train_size::]
    train_df = train_df.iloc[:train_size:]
    test_df = pd.read_csv("data/test.csv")
    drop_invalid_char_df(test_df)


    train_ds = MyDataset(train_df, max_question_length, max_answer_length, processor, TRAIN_TRANSFORM)
    val_ds = MyDataset(val_df, max_question_length, max_answer_length, processor, BASE_TRANSFORM)
    test_ds = MyDataset(test_df, max_question_length, max_answer_length, processor, BASE_TRANSFORM)
    

    def collate_fn(batch):
        return {
            key: torch.stack([item[key] for item in batch])
            for key in batch[0]
        }


    dl_wrapper = lambda ds, sh: DataLoader(ds, batch_size = batch_size, shuffle = sh, num_workers = 4, persistent_workers = True, pin_memory = True, collate_fn = collate_fn)
    
    train_dl = dl_wrapper(train_ds, True)
    val_dl = dl_wrapper(val_ds, False)
    test_dl = dl_wrapper(test_ds, False)
    
    return train_dl, val_dl, test_dl