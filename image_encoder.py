import numpy as np
from PIL import Image, ImageOps
import pandas as pd
from my_tools import *
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

df = pd.read_csv(f"data/original.csv")
img_dict = MyImage.get_img_dict()
img_classes = dict({})
for i, c in zip(df["img_id"], df["source"]):
    if i not in img_classes:
        img_classes[i] = c
        
class ImgDataset(Dataset):
    def __init__(self, img_dict, img_classes):
        super().__init__()
        
        np.unique(list(img_classes.values()))
        self.data = list(img_dict.keys())
        self.dict = img_dict
        self.label = img_classes
        self.label_map = {v: k for k, v in dict(enumerate(np.unique(list(img_classes.values())))).items()}
        self.transform = transforms.ToTensor()
        
    def __len__(self):
        return 6500
    
    def __getitem__(self, index):
        d = self.data[index]
        img = Image.open(self.dict[d]).convert("RGB")
        img = MyImage.change_size(img, (224, 224))
        label = self.label[d]
        label = self.label_map[label]
        
        img = self.transform(img)
        label = torch.tensor(label, dtype = torch.long)
        
        return img, label

	
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, padding = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 2),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size = 3, padding = 2),
            nn.ReLU(),
        )
        
        self.blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 2),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 2),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(64, 5)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        
        out = self.blocks(encoded).squeeze([-1, -2])
        out = self.head(out)
        
        return encoded, out
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
train_ds = ImgDataset(img_dict, img_classes)
train_dl = DataLoader(train_ds, batch_size = 32)

model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)


epochs = 5
total_steps = epochs * len(train_dl)
for ep in range(epochs):
    print(f"Epoch {ep+1}/{epochs}")
    ep_loss = []
    pbar = tqdm(train_dl)
    for img, label in pbar:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        encoded, logits = model(img)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        
        ep_loss.append(loss.item())
        pbar.set_postfix(cur_loss = round(ep_loss[-1], 3), avg_loss = round(np.mean(ep_loss), 3))
        

torch.save(model, "results/image_encoder")