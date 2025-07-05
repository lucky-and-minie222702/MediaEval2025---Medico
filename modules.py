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


class TextImageEncoder(nn.Module):
    def __init__(self, vocab_size, max_length, padding_idx, embedding_dim, feed_forward_dim, num_att_heads, word_embedding_dim = None,dropout = 0.0, attention_dropout = 0.0):
        super().__init__()
        
        self.img_encode = ImageEncoder(embedding_dim if word_embedding_dim is None else word_embedding_dim)
        
        self.word_embed = WordEmbedding(
            vocab_size, embedding_dim, 
            max_length = max_length, 
            padding_idx = padding_idx, 
        )
        
        self.ngram_encode = ngramEncoder(
            embedding_dim, embedding_dim,
            dropout = dropout,
        )
        
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
            nn.Linear(feed_forward_dim, embedding_dim, bias = False)
        )
        
    def forward(self, image, words, word_padding_masks = None):
        word_embed = self.word_embed(words)
        ngram_feats = self.ngram_encode(word_embed)  # (B, embedding_dim, text_len)
        
        img_spatial_feats = self.img_encode(image)  # (B, embedding_dim, H * W)
        
        cross_attention_feats = self.attention(
            query = ngram_feats,
            key = img_spatial_feats,
            value = img_spatial_feats,
            need_weights = False,
        )  # (B, embedding_dim, text_len)
        
        if word_padding_masks is not None:
            # word_padding_masks: (B, text_len)
            cross_attention_feats = cross_attention_feats * word_padding_masks.unsqueeze(-1)

        cross_attention_feats = self.norm1(cross_attention_feats + ngram_feats)  # (B, embedding_dim, text_len)
        
        out = self.feed_forward(cross_attention_feats)
        out = self.norm2(out + cross_attention_feats)   # (B, embedding_dim, text_len)

        return out

    
class AnswerLSTMDecoder(nn.Module):
    def __init__(self, in_channels, embedding_dim, num_classes, classifier_dim, word_embedding, num_lstm_layers = 1, dropout = 0.0, lstm_dropout = 0.0):
        super().__init__()
        
        self.word_embed = word_embedding
        
        self.lstm = nn.LSTM(
            in_channels,
            embedding_dim,
            num_layers = num_lstm_layers,
            dropout = lstm_dropout,
            batch_first = True,
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, classifier_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes),
        )
        
    def forward(self, cross_attention_feats, max_length, answer_tokens = None, teacher_forcing_ratio = 0.5):
        bos_feats, (hidden, cell) = self.lstm(cross_attention_feats)
        # bos_token: (B, embedding_dim, text_len) = beginning of sequence
        # hidden: (B, embedding_dim)
        # cell: (B, embedding_dim)
        
        # init first output
        input_token = self.classifier(bos_feats)  # (B, vocab_size)
        input_token = torch.argmax(input_token, dim = 1, keepdim = True) # (B, 1)
        
        outputs = []

        for t in range(max_length):
            token_emb = self.word_embed(input_token)  # (B, 1, embed_dim)
            
            output, (hidden, cell) = self.lstm(token_emb, hidden, cell) 
            # output: (B, 1, embedding_dim)
            # hidden: (B, embedding_dim)
            # cell: (B, embedding_dim)

            output = output.squeeze(1)  # (B, embedding_dim)
            logits = self.classifier()  # (B, vocab_size)
            
            outputs.append(logits.unsqueeze(1))  # (B,)
            
            if answer_tokens is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # teacher forcing
                input_token = answer_tokens[::, t]  # (B,)
                input_token = input_token.unsqueeze(1)  # (B, 1)
            else:
                input_token = logits.argmax(dim = 1, keepdim = True)  # (B, 1)
                
        outputs = torch.cat(outputs, dim = 1)  # (B, max_length)
        return outputs