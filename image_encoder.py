import numpy as np
from PIL import Image, ImageOps
import pandas as pd
from my_tools import *
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
import random
import copy


df = pd.read_csv(f"data/original.csv")
img_dict = MyImage.get_img_dict()
img_classes = dict({})
for i, c in zip(df["img_id"], df["source"]):
    if i not in img_classes:
        img_classes[i] = c
        
class ImgDataset(Dataset):
    def __init__(self, img_dict, img_classes, data = None):
        super().__init__()
        
        self.dict = img_dict
        self.label = img_classes
        self.label_map = {v: k for k, v in dict(enumerate(np.unique(list(img_classes.values())))).items()}
        self.to_tensor = transforms.ToTensor()
        
        self.data = data
        if data is None:
            self.data = list(img_dict.keys())
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        d = self.data[index]
        img = Image.open(self.dict[d]).convert("RGB")
        img = MyImage.change_size(img, (224, 224))
        label = self.label[d]
        label = self.label_map[label]
        
        img = self.to_tensor(img)
        label = torch.tensor(label, dtype = torch.long)
        
        return img, label

	
class ImgModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.transform = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.SiLU(),
            nn.Conv2d(64, 3, kernel_size = 3, padding = 1),
            nn.Sigmoid(),
        )
        
        self.restore = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.SiLU(),
            nn.Conv2d(64, 3, kernel_size = 3, padding = 1),
            nn.Sigmoid(),
        )
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 2),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 2),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 2),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # for matching
        self.mch_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        # for classifier
        self.cls_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 5)
        )
        
    def forward(self, x, mode):
        B = x.shape[0]

        transformed = self.transform(x)
        assert transformed.shape == x.shape
        
        if mode == "transform":
            return transformed
        
        if mode == "restore":
            restore = self.restore(transformed)
            return restore
        
        encoded = self.encoder(transformed)
        encoded = self.pool(encoded).squeeze([-1, -2])  # B, 64
        
        if mode == "match":
            pairs = encoded.contiguous().view(B // 2, 128)
            return self.mch_head(pairs)
        if mode == "classify":
            return self.cls_head(encoded)

    
def contrastive_logits(x, temperature):
    x = F.normalize(x, dim = -1)
    return (x @ x.t()) / temperature 


def contrastive_loss(logits):
    B = logits.shape[0]
    target = torch.arange(B, device = logits.device)
    loss = F.cross_entropy(logits, target)
    return loss


class ImgTrainer():
    def __init__(self):
        self.model = ImgModel()

        data_keys = list(img_dict.keys())
        random.shuffle(data_keys)
        
        train_keys, val_keys = data_keys[:5500:], data_keys[5500::]
        
        self.train_ds = ImgDataset(img_dict, img_classes, train_keys)
        self.val_ds = ImgDataset(img_dict, img_classes, val_keys)
        
        self.checkpoints = []
        self.logs = []
        self.phase = 0
        
    def train(self, mode, batch_size, epochs, lr):
        self.phase += 1
        
        # freeze transform layer
        if mode == "restore":
            for param in self.model.transform.parameters():
                param.requires_grad = False
        else:
            for param in self.model.transform.parameters():
                param.requires_grad = True
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dl = DataLoader(self.train_ds, batch_size = batch_size, shuffle = True)
        val_dl = DataLoader(self.val_ds, batch_size = batch_size, shuffle = True)
        
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr = lr)
        if mode == "classify":
            criterion = nn.CrossEntropyLoss()
        elif mode == "match":
            criterion = nn.BCEWithLogitsLoss()
        elif mode == "restore":
            criterion = nn.MSELoss()
        
        logs = {
            "train": [],
            "val": []
        }

        for ep in range(epochs):
            print(f"Epochs {ep+1}/{epochs}")
            
            self.model.train()
            losses = [] 
            pbar = tqdm(train_dl, desc = " Train")
            for img, label in pbar:
                img, label = img.to(device), label.to(device)
                optimizer.zero_grad()
                out = self.model(img, mode = mode)
                
                if mode == "classify":
                    pass
                elif mode == "match":
                    label = label.contiguous().view(batch_size // 2, 2)
                    label = (label[:, 0] == label[:, 1]).float().unsqueeze(-1)
                elif mode == "restore":
                    label = img
                    
                loss = criterion(out, label)
                losses.append(loss.item())
                
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix(avg_loss = np.mean(losses), cur_loss = losses[-1])    

            logs["train"].extend(losses)

            
            self.model.eval()
            losses = [] 
            pbar = tqdm(val_dl, desc = " Val  ")
            with torch.no_grad():
                for img, label in pbar:
                    img, label = img.to(device), label.to(device)
                    out = self.model(img, mode = mode)
                    
                    if mode == "classify":
                        pass
                    elif mode == "match":
                        label = label.contiguous().view(batch_size // 2, 2)
                        label = (label[:, 0] == label[:, 1]).float().unsqueeze(-1)
                    elif mode == "restore":
                        label = img
                    
                    loss = criterion(out, label)
                    losses.append(loss.item())
                    
                    pbar.set_postfix(avg_loss = np.mean(losses), cur_loss = losses[-1])
                
            logs["val"].extend(losses)
            
        self.checkpoints.append({
            "name": f"[{self.phase}] {mode}",
            "model": copy.deepcopy(self.model)
        })
        
        self.logs.append({
            "name": f"[{self.phase}] {mode}",
            "log": copy.deepcopy(self.model)
        })
        
trainer = ImgTrainer()

print("\nClassify:")
trainer.train(
    mode = "classify", 
    batch_size = 32, 
    epochs = 10,
    lr = 0.001,
)

print("\nMatch:")
trainer.train(
    mode = "match", 
    batch_size = 100, 
    epochs = 10,
    lr = 0.0001,
)

print("\nRestore:")
trainer.train(
    mode = "restore", 
    batch_size = 32, 
    epochs = 20,
    lr = 0.001,
)

joblib.dump(trainer.logs, "results/image-encoder-logs")
joblib.dump(trainer.checkpoints, "results/image-encoder-checkpoints")