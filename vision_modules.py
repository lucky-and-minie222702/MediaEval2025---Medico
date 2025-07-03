import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_cls = nn.Conv2d, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            conv_cls(in_channels, out_channels, **kwargs, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU() 
        )

    def forward(self, x):
        return self.block(x)


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


class EfficientNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio = 4, stride = 1):
        super().__init__()
        
        mid_channels = in_channels * expansion_ratio
        
        self.expand_block = ConvBlock(in_channels, mid_channels, kernel_size = 1)

        self.deptwise_block = ConvBlock(mid_channels, mid_channels, groups = mid_channels, kernel_size = 3, stride = stride, padding = "same" if stride == 1 else 0)
        self.se_block = SEBlock(mid_channels)
        self.pointwise_block = ConvBlock(mid_channels, out_channels, kernel_size = 1)

        self.block = nn.Sequential(
            self.expand_block,
            self.deptwise_block,
            self.se_block,
            self.pointwise_block,
        )
        
        # residual connection projection
        self.res_block = ConvBlock(in_channels, out_channels, kernel_size = 1 if stride == 1 else 3, stride = stride)
        
        self.dropout = nn.Dropout2d(p = 0.2)

    def forward(self, x):
        shortcut = self.res_block(x)
        out = self.block(x)
        out = out + shortcut
        out = self.dropout(out)
        return out