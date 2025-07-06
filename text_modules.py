from torch import nn
import torch
import torch.nn.functional as F


# postional embedding made for left padded
class WordEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, max_length, padding_idx):
        super().__init__()
        
        self.word_embed = nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx = padding_idx
        )
        
        self.max_length = max_length
        self.pos_embed = nn.Embedding(
            self.max_length + 1,
            embedding_dim,
            padding_idx = 0,
        )
        
    def forward(self, x):
        B, L = x.shape  # (B, text_len)

        Ls = torch.count_nonzero(x, dim = 1)
        pos = torch.zeros((B, L), dtype = torch.long, device = x.device)

        for i, length in enumerate(Ls):
            if length != 0:
                pos[i, -length:] = torch.arange(1, length + 1)

        p_e = self.pos_embed(pos)
        w_e = self.word_embed(x)
        
        out = p_e + w_e

        return out
    
    
class NgramEncoder(nn.Module):
    def __init__(self, word_embedding_dim, proj_dim, dropout = 0.0):
        super().__init__()
        
        self.dropout = nn.Dropout1d(dropout)
        
        self.ngram1_batchnorm = nn.BatchNorm1d(proj_dim)
        self.ngram2_batchnorm = nn.BatchNorm1d(proj_dim)
        self.ngram3_batchnorm = nn.BatchNorm1d(proj_dim)
        
        self.ngram1 = nn.Conv1d(
            word_embedding_dim,
            proj_dim,
            kernel_size = 1,
            bias = False,
        )
        
        self.ngram2 = nn.Conv1d(
            word_embedding_dim,
            proj_dim,
            kernel_size = 2,
            bias = False,
        )
        
        self.ngram3 = nn.Conv1d(
            word_embedding_dim,
            proj_dim,
            kernel_size = 3,
            bias = False,
        )
        
        self.silu = nn.SiLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # (B, text_len, word_embedding_dim)
        W = x.contiguous().permute(0, 2, 1)  # (B, word_embedding_dim, text_len)
        
        ngram1 = self.ngram1(W)
        ngram1 = self.ngram1_batchnorm(ngram1)
        
        ngram2 = self.ngram2(F.pad(W, (1, 0)))
        ngram2 = self.ngram2_batchnorm(ngram2)
        
        ngram3 = self.ngram3(F.pad(W, (2, 0)))
        ngram3 = self.ngram3_batchnorm(ngram3)
        
        ngram_feats = torch.stack([ngram1, ngram2, ngram3], dim = 1)  # (B, 3, in_channels, text_len)
        ngram_feats = torch.max(ngram_feats, dim = 1).values  # (B, in_channels, text_len)
        ngram_feats = self.tanh(ngram_feats)
        ngram_feats = self.dropout(ngram_feats)
        
        return ngram_feats
    