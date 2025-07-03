import torch
from torch import nn
from torchvision.models import resnet18

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 4):
        super().__init__()
        
        mid_channels = in_channels // reduction_ratio
        
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.SiLU(),
            nn.Linear(mid_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C = x.shape[:2:]
        
        squeeze = self.squeeze(x).view(B, C)
        excitation = self.excitation(squeeze).view(B, C, 1, 1)

        return x * excitation
    

# using resnet18 backbone (512, 7, 7)
class ImageEncoder(nn.Module):
    def __init__(self, proj_dim, dropout = 0.0):
        super().__init__()
        
        self.spatial_feats = nn.Sequential(*list(resnet18(pretrained = False).children())[:-2])
        self.pooled_feats = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        self.spatial_proj = nn.Conv2d(512, proj_dim, kernel_size=1)
        self.spatial_dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        B = x.shape[0]
        spatial_feats = self.spatial_feats(x)
        
        spatial_feats = self.spatial_proj(spatial_feats)  # (B, proj_dim, 7, 7)
        spatial_feats = spatial_feats.contigous().reshape(B, -1, 7 * 7)  # (B, proj_dim, 7 * 7)
        
        spatial_feats = self.spatial_dropout(spatial_feats)
        
        return spatial_feats