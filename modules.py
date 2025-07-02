import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
import torch.utils.data as t_data


class PatchesEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size, img_size):
        super().__init__()
        
        self.grid_rows = grid_size[0]
        self.grid_cols = grid_size[1]

        self.patch_H = img_size[0] // self.grid_rows
        self.patch_W = img_size[1] // self.grid_cols
        self.n_patches = self.grid_rows * self.grid_cols

        self.proj = nn.Linear(self.patch_H * self.patch_W * in_channels, out_channels)
        
    def forward(self, x):
        B, C, H, W = x.shape

        patches = x.unfold(2, self.patch_H, self.patch_H).unfold(3, self.patch_W, self.patch_W)  # (B, in_channels, grid_rows, grid_cols, patch_H, patch_W)
        patches = patches.contiguous().permute(0, 2, 3, 1, 4, 5)  # (B, grid_rows, grid_cols, patch_H, patch_W, in_channels)
        patches = patches.reshape(B, -1, C * self.patch_H * self.patch_W)  # (B, N_patches, in_channels * patch_H * patch_W)

        embeddings = self.proj(patches)  # (B, N_patches, embed_dim)
        return embeddings