from turtle import forward
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
import torch.utils.data as t_data


# design for (3, 224, 224) to (48, 56, 56)
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.blocks = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            
            nn.Conv2d(24, 48, kernel_size = 5, stride = 2, padding = 2, bias = False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        
    def forward(self, x):
        return self.blocks(x)
    
    
class PatchEncoder(nn.Module):
    def __init__(self, in_channels, kernel_size, grid_size):
        super().__init__()
        
        self.grid_rows = grid_size[0]
        self.grid_cols = grid_size[1]
        
        self.blocks = [[nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = kernel_size, padding = "same", bias = False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            
            nn.Conv2d(in_channels, in_channels, kernel_size = kernel_size, padding = "same", bias = False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        ) for _ in self.grid_cols]
            for _ in self.grid_rows]
        
    def forward(self, patches):
        for r in self.grid_rows:
            for c in self.grid_cols:
                patch = patches[:, r, c, :, :, :]
                shortcut = patch
                
                encoded = self.blocks[r][c](patch)
                patch = encoded + shortcut
        
        return patches


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size, img_size, pre_encode_patch = True):
        super().__init__()
        
        self.grid_rows = grid_size[0]
        self.grid_cols = grid_size[1]

        self.patch_H = img_size[0] // self.grid_rows
        self.patch_W = img_size[1] // self.grid_cols
        self.n_patches = self.grid_rows * self.grid_cols

        self.proj = nn.Linear(self.patch_H * self.patch_W * in_channels, out_channels)
        
        self.p_encoder = None
        self.pre_encode_patch = pre_encode_patch 
        if pre_encode_patch:
            self.p_encoder = PatchEncoder(in_channels, kernel_size = 3)
        
    def forward(self, x):
        B, C, H, W = x.shape

        patches = x.unfold(2, self.patch_H, self.patch_H).unfold(3, self.patch_W, self.patch_W)  # (B, in_channels, grid_rows, grid_cols, patch_H, patch_W)
        patches = patches.contiguous().permute(0, 2, 3, 1, 4, 5)  # (B, grid_rows, grid_cols, patch_H, patch_W, in_channels)

        if self.pre_encode_patch:
            patches = self.p_encoder(patches)

        patches = patches.reshape(B, self.patch_H * self.patch_W, C * self.patch_H * self.patch_W)  # (B, N_patches, in_channels * patch_H * patch_W)

        embeddings = self.proj(patches)  # (B, N_patches, embed_dim)

        return embeddings
    
    
class WordEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, max_length, padding_idx, dropout = 0.0):
        super().__init__()
        
        self.word_embed = nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx = padding_idx
        )
        
        self.max_length = max_length
        self.pos_embed = nn.Embedding(
            self.max_length,
            embedding_dim,
        )
        
        self.dropout = nn.Dropout(dropout)    
        
    def forward(self, x):
        B, L = x.shape[1]

        pos = torch.arange(0, L, device = x.device)  # (L,)
        pos = pos.unsqueeze(0)  # (1, L)
        pos = pos.expand((B, -1))  # (B, L)

        p_e = self.pos_embed(pos)
        w_e = self.word_embed(x)
        
        x = p_e + w_e
        x = self.dropout(x)

        return w_e + p_e
    
    
class WordEncoder(nn.Module):
    def __init__(self, in_channels, proj_channels, out_channels, num_layers, dropout = 0.0):
        super().__init__()
        
        self.lstm = nn.LSTM(
            proj_channels,
            out_channels,
            num_layers = num_layers,
            dropout = dropout,
            batch_first = True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.ngram1_batchnorm = nn.BatchNorm1d(proj_channels)
        self.ngram2_batchnorm = nn.BatchNorm1d(proj_channels)
        self.ngram3_batchnorm = nn.BatchNorm1d(proj_channels)
        
        self.ngram1 = nn.Conv1d(
            in_channels,
            proj_channels,
            kernel_size = 1,
            bias = False,
            padding = (0, 0)
        )
        
        self.ngram2 = nn.Conv1d(
            in_channels,
            proj_channels,
            kernel_size = 2,
            bias = False,
            padding = (1, 0)
        )
        
        self.ngram3 = nn.Conv1d(
            in_channels,
            proj_channels,
            kernel_size = 3,
            bias = False,
            padding = (1, 1)
        )
        
        self.silu = nn.SiLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x, hidden = None, cell = None):
        # (B, in_channels, seq_len)
        ngram1 = self.ngram2(x)
        ngram1 = self.ngram1_batchnorm(ngram1)
        ngram1 = self.tanh(ngram1)
        ngram1 = self.dropout(ngram1)
        
        ngram2 = self.ngram2(x)
        ngram2 = self.ngram1_batchnorm(ngram2)
        ngram2 = self.tanh(ngram2)
        ngram2 = self.dropout(ngram2)
        
        ngram3 = self.ngram3(x)
        ngram3 = self.ngram1_batchnorm(ngram3)
        ngram3 = self.tanh(ngram3)
        ngram3 = self.dropout(ngram3)
        
        sentence = torch.stack([ngram1, ngram2, ngram3], dim = 1)  # (B, 3, in_channels, seq_len)
        sentence = torch.max(sentence, dim = 1)  # (B, in_channels, seq_len)
        sentence, _, _ = self.lstm(sentence, hidden, cell)  # (B, out_channels, seq_len)
        sentence = self.dropout(sentence)
        
        return sentence