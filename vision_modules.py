import torch
from torch import nn
import torchvision.models as v_models
    

# using resnet34 backbone (512, 7, 7)
class ImageEncoder(nn.Module):
    def __init__(self, proj_dim, backbone =  nn.Sequential(*list(v_models.resnet34(weights = v_models.ResNet34_Weights.DEFAULT).children())[:-2]), back_bone_dim = 512, dropout = 0.0):
        super().__init__()
        
        self.proj_dim = proj_dim
        
        self.feats = backbone
        
        self.proj = nn.Conv2d(back_bone_dim, proj_dim, kernel_size = 1)
        self.dropout = nn.Dropout1d(dropout)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        B = x.shape[0]
        feats = self.feats(x)
        
        feats = self.proj(feats)  # (B, proj_dim, H, W)
        feats = feats.contiguous().reshape(B, self.proj_dim, -1)  # (B, proj_dim, H * W)
        feats = self.tanh(feats)
        
        feats = self.dropout(feats)
        
        return feats