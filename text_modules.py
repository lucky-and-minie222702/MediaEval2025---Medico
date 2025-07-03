from torch import nn
import torch


# postional embedding made for left padded
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

        pos = torch.zeros((B, L), dtype = torch.long, device = x.device)

        for i, length in enumerate(L):
            pos[i, -length:] = torch.arange(1, length + 1)

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
    