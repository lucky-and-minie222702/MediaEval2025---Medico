from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from my_tools import *
import os
from os import path
from sklearn.model_selection import train_test_split


TRAIN_TRANSFORM = transforms.Compose([
    transforms.ColorJitter(
        brightness = 0.2,
        contrast = 0.2,
    ),
    transforms.RandomRotation(12),
])

QUESTION_INSTRUCTION = (
    "You are a medical vision-language assistant. "
    "Answer the question using only evidence-based medical facts and the image, avoiding speculation or anecdotes. "
    "Respond in natural medical language as a doctor would, in one sentence. "
)
QUESTION_PROMPT = QUESTION_INSTRUCTION + "Question: {q}"

CAPTION_INSTRUCTION = (
    "You are a medical vision-language assistant. "
    "Write a caption for the given image. "
    "Respond in natural medical language as a doctor would."
)


def norm_text(text):
    out = text
    out = out.replace("/", " ")
    out = out.replace(";", ",")
    
    return out

def preprocess(
    processor, 
    d, 
    max_length, 
    caption_prompt = False, 
    include_answer = True, 
    mask_answer = -100, 
    img_dict = None, 
    transform = None, 
    all_data = None, 
    n_captions = None):
    if img_dict is None:
        img_dict = MyImage.get_img_dict()

    image = Image.open(img_dict[d["img_id"]]).convert("RGB")
    image = MyImage.change_size(image, (224, 224))

    if transform is not None:
        image = transform(image)

    if caption_prompt:
        quest = CAPTION_INSTRUCTION
    else:
        quest = QUESTION_PROMPT.format(q = norm_text(d['question']))    

    if caption_prompt:
        ids = d["qid"]
        df_ = all_data.reset_index(drop = True)
        ans = []
        for _ in range(n_captions):
            df_ = df_[df_["qid"].apply(lambda x: set(x).isdisjoint(ids))].reset_index(drop = True).index
            idx = np.random.randint(len(df_))
            
            cap = all_data["answer"][idx]
            if cap[-1] != ".":
                cap += "."

            ans.append(cap)
            ids.extend(df_["qid"])

        ans = " ".join(ans)
    else:
        ans = norm_text(d["answer"])

    if ans[-1] != ".":
        ans += "."
    
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
            ans, 
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
    def __init__(
        self, 
        df, 
        max_question_legnth, 
        max_answer_length, 
        processor,
        include_answer = True, 
        caption_prompt = False, 
        n_captions = None, 
        transform = None,
        mask_answer = -100):
        super().__init__()

        self.max_length = [max_question_legnth, max_answer_length]
        self.processor = processor
        self.transform = transform
        
        question_dict = set({})
        org = df["original"].apply(json.loads)
        for pairs in org:
            for pair in pairs:
                question_dict.add(pair["q"])
        question_dict = dict(enumerate(question_dict))
        question_dict.update({v: k for k, v in question_dict.items()})
        to_ids  = lambda o: sorted([question_dict[p["q"]] for p in o])
        df["qid"] = org.apply(to_ids)
        	
        self.raw_data = df
        self.data = df.to_dict(orient = 'records')
        self.question_dict = question_dict
        self.img_dict = MyImage.get_img_dict()
        self.mask_answer = mask_answer
        self.include_answer = include_answer
        self.caption_prompt = caption_prompt
        self.n_captions = n_captions
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return preprocess(
            processor = self.processor, 
            d = self.data[index], 
            caption_prompt = self.caption_prompt,
            include_answer = self.include_answer,
            max_length = self.max_length,
            img_dict = self.img_dict, 
            mask_answer = self.mask_answer,
            transform = self.transform,
            all_data = self.raw_data,
            n_captions = self.n_captions,
        )
    
    
def load_data(
    processor, 
    max_question_length, 
    max_answer_length, 
    train_ratio = None, 
    train_complexities = [1, 2, 3], 
    train_augment = True, 
    test_complexities = [1, 2, 3], 
    test_only = False, 
    caption_prompt = False,
    n_captions = None,
    seed = 27022009):
    np.random.seed(seed)
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
        test_ds = MyDataset(
            test_df, 
            max_question_length, 
            max_answer_length, 
            processor, 
            caption_prompt = caption_prompt, 
            n_captions = n_captions, 
            transform = None)
        return test_ds

    # load df
    df = pd.read_csv("data/train.csv")
    drop_invalid_char_df(df)
    mask = df["complexity"].map(lambda x: x in train_complexities)
    df = df[mask]

    train_df, val_df = train_test_split(df, train_size = train_ratio, shuffle = True, random_state = seed)

    train_ds = MyDataset(
        train_df, 
        max_question_length, 
        max_answer_length, 
        processor, 
        caption_prompt = caption_prompt, 
        n_captions = n_captions, 
        transform = TRAIN_TRANSFORM if train_augment else None)
    val_ds = MyDataset(
        val_df, 
        max_question_length, 
        max_answer_length, 
        processor, 
        caption_prompt = caption_prompt, 
        n_captions = n_captions, 
        transform = None)
    
    return train_ds, val_ds

    
        