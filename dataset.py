from torchvision import transforms
from torch.utils.data import Dataset
import torch.utils.data as t_data
from PIL import Image
import pandas as pd
from tqdm import tqdm
from custom_obj import *

# resnet like norm
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]


def get_img_dict(data = "train"):
    file_ids = pd.read_csv(f"data/{data}.csv")["img_id"].unique().tolist()
    file_paths = [f"data/images/{s}.jpg" for s in file_ids]

    images = {}

    for file_path, file_id in tqdm(zip(file_paths, file_ids), total = len(file_ids)):
        img = Image.open(file_path)
        img = MyImage.change_size(img, (224, 224))
        images[file_id] = img
        
    return images


MY_TRANSFORM = transforms.Compose([
    transforms.ColorJitter(
        brightness = 0.2,
        contrast = 0.2,
        saturation = 0.1,
    ),
    transforms.RandomAffine(36),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406], 
        std = [0.229, 0.224, 0.225]
    ),
])


class MyDataset(Dataset):
    def __init__(self, img_dict, text_data, transform = MY_TRANSFORM):
        super().__init__()
        self.img_dict = img_dict
        self.text_data = text_data
        self.transform = transform
        
    def __len__(self):
        return len(self.question_pair)
    
    def __getitem__(self, index):
        img = self.img_dict[self.text_data[index][0]]
        question = self.text_data[index][1]
        answer = self.text_data[index][2]
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, question, answer