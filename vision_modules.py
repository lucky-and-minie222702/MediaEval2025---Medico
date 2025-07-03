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
        batch_size, channels, height, width = x.shape
        
        squeeze = self.squeeze(x).view(batch_size, channels)
        excitation = self.excitation(squeeze).view(batch_size, channels, 1, 1)

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

    
# encoder should be lambda x: EncoderModel() if multiple encoder are used
class PatchEncoder(nn.Module):
    def __init__(self, in_channels, encoder, grid_size, multiple_encoder = False, dropout = 0.0):
        super().__init__()
        
        self.grid_rows = grid_size[0]
        self.grid_cols = grid_size[1]
        
        self.multipl_encoder = multiple_encoder
        if self.multipl_encoder:
            self.blocks = [
                [encoder for _ in self.grid_cols]
                    for _ in self.grid_rows
            ]
        else:
            self.block = encoder
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, patches):
        for r in self.grid_rows:
            for c in self.grid_cols:
                patch = patches[:, r, c, :, :, :]
                shortcut = patch
                
                if self.multipl_encoder:
                    encoded = self.blocks[r][c](patch)
                else:
                    encoded = self.blocks(patch)

                patch = encoded + shortcut
                patch = self.dropout(patch)
        
        return patches


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size, img_size, pre_encoder_patch = None, dropout = 0.0):
        super().__init__()
        
        self.grid_rows = grid_size[0]
        self.grid_cols = grid_size[1]

        self.patch_H = img_size[0] // self.grid_rows
        self.patch_W = img_size[1] // self.grid_cols
        self.n_patches = self.grid_rows * self.grid_cols

        self.proj = nn.Linear(self.patch_H * self.patch_W * in_channels, out_channels)
        
        self.p_encode = pre_encoder_patch
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, C, H, W = x.shape

        patches = x.unfold(2, self.patch_H, self.patch_H).unfold(3, self.patch_W, self.patch_W)  # (B, in_channels, grid_rows, grid_cols, patch_H, patch_W)
        patches = patches.contiguous().permute(0, 2, 3, 1, 4, 5)  # (B, grid_rows, grid_cols, patch_H, patch_W, in_channels)

        if self.p_encode is not None:
            patches = self.p_encode(patches)

        patches = patches.reshape(B, self.patch_H * self.patch_W, C * self.patch_H * self.patch_W)  # (B, N_patches, in_channels * patch_H * patch_W)

        embeddings = self.proj(patches)  # (B, N_patches, embed_dim)
        embeddings = self.dropout(embeddings)

        return embeddings