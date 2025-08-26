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
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.SiLU(),
            nn.Conv2d(64, 3, kernel_size = 3, padding = 1),
            nn.Sigmoid(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 2),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 2),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 2),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # for contrastive
        self.ctr_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 64)
        )
        self.emb_norm = lambda emb: F.normalize(emb, dim = -1)
        
        # for matching
        self.mch_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, mode):
        B = x.shape[0]

        encoded = self.encoder(x)
        encoded = torch.round(encoded * 255).float()
        assert encoded.shape == x.shape
        
        if mode == "encode":
            return encoded
        
        decoded = self.decoder(encoded)
        decoded = self.pool(decoded).squeeze([-1, -2])  # B, 64
        
        if mode == "contrastive":
            emb = self.ctr_head(decoded)
            emb = self.emb_norm(emb)
            return emb
        if mode == "matching":
            pairs = decoded.contiguous().view(B // 2, 128)
            return self.mch_head(pairs)

    
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
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dl = DataLoader(self.train_ds, batch_size = batch_size, shuffle = True)
        val_dl = DataLoader(self.val_ds, batch_size = batch_size, shuffle = True)
        
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr = lr)
        criterion = nn.BCEWithLogitsLoss()
        
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
                
                if mode == "contrastive":
                    logits = contrastive_logits(out, temperature = 0.1)
                    loss = contrastive_loss(logits)
                elif mode == "matching":
                    label = label.contiguous().view(batch_size // 2, 2)
                    label = (label[:, 0] == label[:, 1]).float().unsqueeze(-1)
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
                    
                    if mode == "contrastive":
                        logits = contrastive_logits(out, temperature = 0.1)
                        loss = contrastive_loss(logits)
                    elif mode == "matching":
                        label = label.contiguous().view(batch_size // 2, 2)
                        label = (label[:, 0] == label[:, 1]).float().unsqueeze(-1)
                        loss = criterion(out, label)
                    
                    losses.append(loss.item())
                    
                    pbar.set_postfix(avg_loss = np.mean(losses), cur_loss = losses[-1])
                
            logs["val"].extend(losses)
            
        self.checkpoints.append({
            "name": f"[{self.phase}] - {mode}",
            "model": copy.deepcopy(self.model)
        })
        
trainer = ImgTrainer()
print("\nMatching:")
trainer.train(
    mode = "matching", 
    batch_size = 32, 
    epochs = 5,
    lr = 0.0005,
)
print("\nContrastive:")
trainer.train(
    mode = "contrastive", 
    batch_size = 128, 
    epochs = 5,
    lr = 0.001,
)
joblib.dump(trainer.logs, "results/image-encoder-logs")
joblib.dump(trainer.checkpoints, "results/image-encoder-checkpoints")