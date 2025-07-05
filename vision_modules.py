import torch
from torch import nn
import torchvision.models as v_models
    

# using resnet18 backbone (512, 7, 7)
class ImageEncoder(nn.Module):
    def __init__(self, proj_dim, backbone =  nn.Sequential(*list(v_models.resnet18(weights = v_models.ResNet18_Weights.DEFAULT).children())[:-2]), back_bone_dim = 512, dropout = 0.0):
        super().__init__()
        
        self.proj_dim = proj_dim
        
        self.spatial_feats = backbone
        self.pooled_feats = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        self.spatial_proj = nn.Conv2d(back_bone_dim, proj_dim, kernel_size = 1)
        self.spatial_dropout = nn.Dropout2d(dropout)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        B = x.shape[0]
        spatial_feats = self.spatial_feats(x)
        
        spatial_feats = self.spatial_proj(spatial_feats)  # (B, proj_dim, H, W)
        spatial_feats = spatial_feats.contigous().reshape(B, self.proj_dim, -1)  # (B, proj_dim, H * W)
        spatial_feats = self.tanh(spatial_feats)
        
        spatial_feats = self.spatial_dropout(spatial_feats)
        
        return spatial_feats