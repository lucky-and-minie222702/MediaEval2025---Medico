from torch import nn
import torch


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
            self.max_length,
            embedding_dim,
        )
        
    def forward(self, x):
        B, L = x.shape[1]

        pos = torch.zeros((B, L), dtype = torch.long, device = x.device)

        for i, length in enumerate(L):
            pos[i, -length:] = torch.arange(1, length + 1)

        p_e = self.pos_embed(pos)
        w_e = self.word_embed(x)
        
        x = p_e + w_e

        return w_e + p_e
    
    
class NgramEncoder(nn.Module):
    def __init__(self, in_channels, proj_dim, dropout = 0.0):
        super().__init__()
        
        self.dropout = nn.Dropout1d(dropout)
        
        self.ngram1_batchnorm = nn.BatchNorm1d(proj_dim)
        self.ngram2_batchnorm = nn.BatchNorm1d(proj_dim)
        self.ngram3_batchnorm = nn.BatchNorm1d(proj_dim)
        
        self.ngram1 = nn.Conv1d(
            in_channels,
            proj_dim,
            kernel_size = 1,
            bias = False,
            padding = (0, 0)
        )
        
        self.ngram2 = nn.Conv1d(
            in_channels,
            proj_dim,
            kernel_size = 2,
            bias = False,
            padding = (1, 0)
        )
        
        self.ngram3 = nn.Conv1d(
            in_channels,
            proj_dim,
            kernel_size = 3,
            bias = False,
            padding = (1, 1)
        )
        
        self.silu = nn.SiLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x, hidden = None):
        # (B, in_channels, text_len)
        ngram1 = self.ngram2(x)
        ngram1 = self.ngram1_batchnorm(ngram1)
        
        ngram2 = self.ngram2(x)
        ngram2 = self.ngram1_batchnorm(ngram2)
        
        ngram3 = self.ngram3(x)
        ngram3 = self.ngram1_batchnorm(ngram3)
        
        ngram_feats = torch.stack([ngram1, ngram2, ngram3], dim = 1)  # (B, 3, in_channels, text_len)
        ngram_feats = torch.max(ngram_feats, dim = 1)  # (B, in_channels, text_len)
        ngram_feats = self.tanh(ngram_feats)
        ngram_feats = self.dropout(ngram_feats)
        
        return ngram_feats
    