from vision_modules import *
from text_modules import *

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


class TextImageEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, feed_forward_dim, num_att_heads, dropout = 0.0, attention_dropout = 0.0):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads = num_att_heads,
            batch_first = True,
            dropout = attention_dropout
        )
        
        self.norm1 = nn.LayerNorm(embedding_dim)  # after attention
        self.norm2 = nn.LayerNorm(embedding_dim)  # after feed forward
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, embedding_dim, bias = False)
        )
        
    def forward(self, ngram_feats, img_spatial_feats, word_padding_masks = None):
        cross_attention_feats, _ = self.attention(
            query = ngram_feats,
            key = img_spatial_feats,
            value = img_spatial_feats,
            need_weights = False,
        )  # (B, text_len, embedding_dim)
        
        if word_padding_masks is not None:
            # word_padding_masks: (B, text_len)
            cross_attention_feats = cross_attention_feats * word_padding_masks.unsqueeze(-1)

        cross_attention_feats = self.norm1(cross_attention_feats + ngram_feats)  # (B, embedding_dim, text_len)
        
        out = self.feed_forward(cross_attention_feats)
        out = self.norm2(out + cross_attention_feats)   # (B, text_len, embedding_dim)

        return out


# encoder layer should be lambda: TextImageEncoderLayer(...)
class TextImageEncoder(nn.Module):
    def __init__(self, vocab_size, max_answer_length, padding_idx, embedding_dim, word_embedding_dim, encoder_spawn, num_layers = 1, dropout = 0.0):
        super().__init__()

        self.img_encode = ImageEncoder(embedding_dim)
        
        self.word_embed = WordEmbedding(
            vocab_size, word_embedding_dim, 
            max_length = max_answer_length, 
            padding_idx = padding_idx, 
        )
        
        self.ngram_encode = NgramEncoder(
            word_embedding_dim, embedding_dim,
            dropout = dropout,
        )
        
        self.encoders = nn.ModuleList([encoder_spawn() for _ in range(num_layers)])
        
    def forward(self, image, words, word_padding_masks = None):
        word_embed = self.word_embed(words)
        ngram_feats = self.ngram_encode(word_embed)  # (B, embedding_dim, text_len)
        
        img_spatial_feats = self.img_encode(image)  # (B, embedding_dim, H * W)

        ngram_feats = ngram_feats.contiguous().permute(0, 2, 1)  # (B, text_len, embedding_dim)
        img_spatial_feats = img_spatial_feats.contiguous().permute(0, 2, 1)  # (B, H * W, embedding_dim)
        
        for t, encoder in enumerate(self.encoders):
            print(t, ngram_feats.device, img_spatial_feats.device)
            ngram_feats = encoder(ngram_feats, img_spatial_feats, word_padding_masks)
            
        return ngram_feats
    
    
class AnswerLSTMDecoder(nn.Module):
    def __init__(self, in_channels, embedding_dim, num_classes, classifier_dim, word_embedding, ngram_encoder, num_gru_layers = 1, dropout = 0.0, lstm_dropout = 0.0):
        super().__init__()
        
        self.word_embed = word_embedding
        self.ngram_encoder = ngram_encoder
        
        self.lstm = nn.LSTM(
            in_channels,
            embedding_dim,
            num_layers = num_gru_layers,
            dropout = lstm_dropout,
            batch_first = True,
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, classifier_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes),
        )
        
        self.dropout = nn.Dropout1d(dropout)
        
    def forward(self, cross_attention_feats, max_answer_length, answers = None, teacher_forcing_ratio = 0.5):
        bos_feats, (hidden, cell) = self.lstm(cross_attention_feats)
        # bos_token: (B, text_len, embedding_dim)
        bos_feats = bos_feats[::, -1, ::]  # (B, embedding_dim)
        
        # init first output
        input_token = self.classifier(bos_feats)  # (B, vocab_size)
        input_token = torch.argmax(input_token, dim = 1, keepdim = True) # (B,)
        
        outputs = []

        for t in range(max_answer_length):
            token_embed = self.word_embed(input_token)  # (B, 1, embedding_dim)
            
            token_embed = token_embed.contiguous().permute(0, 2, 1)  # (B, embedding_dim, 1)
            token_embed = self.ngram_encoder.ngram1(token_embed)
            token_embed = self.ngram_encoder.ngram1_batchnorm(token_embed)
            token_embed = self.dropout(token_embed)
            token_embed = token_embed.contiguous().permute(0, 2, 1)  # (B, 1, embedding_dim)
            
            output, (hidden, cell) = self.lstm(token_embed, (hidden, cell)) 
            # output: (B, 1, embedding_dim)

            output = output.squeeze(1)  # (B, embedding_dim)
            logits = self.classifier(output)  # (B, vocab_size)
            
            outputs.append(logits)
            
            if answers is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # teacher forcing
                input_token = answers[::, t]  # (B,)
                input_token = input_token.unsqueeze(1)  # (B, 1)
            else:
                input_token = torch.argmax(outputs[-1], dim = 1, keepdim = True) # (B, 1)
                
        outputs = torch.stack(outputs, dim = 1)  # (B, max_answer_length, vocab_size)
        return outputs