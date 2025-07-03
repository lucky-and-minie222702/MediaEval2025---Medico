from vision_modules import *
from text_modules import *


class ImageWordCoAttention(nn.Module):
    def __init__(self, vocab_size, max_length, padding_idx, embedding_dim, out_channels, feed_forward_dim, num_lstm_layers, num_att_heads, pre_encoder_patch = None, dropout = 0.0):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(
            3, embedding_dim,
            pre_encoder_patch = pre_encoder_patch, 
            dropout = dropout,
        )
        
        self.word_embed = WordEmbedding(
            vocab_size, embedding_dim, 
            max_length = max_length, 
            padding_idx = padding_idx, 
            dropout = dropout
        )
        
        self.word_encode = WordEncoder(
            embedding_dim, embedding_dim, embedding_dim,
            num_layers = num_lstm_layers
        )
        
        self.attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads = num_att_heads,
            batch_first = True,
            dropout = dropout
        )
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.SiLU(),
            nn.Linear(feed_forward_dim, embedding_dim, bias = False)
        )
        
    def forward(self, image, words, word_padding_mask):
        img_embed = self.patch_embed(image)
        word_embed = self.word_embed(words)
        
        ngrams, sentence = self.word_encode(word_embed)
        # ngrams: (B, embedding_dim, seq_len)
        # sentence: (B, embedding_dim)
        
        cross_attention = self.attention(
            query = ngrams,
            key = img_embed,
            query = img_embed,
            
            need_weights = False,
        )
        
        cross_attention = cross_attention * word_padding_mask.unsqueeze(-1)
        cross_attention = self.norm1(cross_attention + ngrams)  # (B, embedding_dim, word_len)
        
        out = self.feed_forward(cross_attention)
        out = self.norm2(out + cross_attention)   # (B, embedding_dim, word_len)
        
        return cross_attention, sentence
    
    
class AnswerGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        
        
