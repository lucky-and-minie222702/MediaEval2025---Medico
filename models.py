from vision_modules import *
from text_modules import *
from modules import *


class MyLSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        self.encoder = TextImageEncoder(
            vocab_size = vocab_size,
            max_length = 35,
            padding_idx = 0,
            embedding_dim = 512,
            word_embedding_dim = 64,
            dropout = 0.1,
            attention_dropout = 0.1
        )
        
        self.decoder = AnswerLSTMDecoder(
            in_channels = 512,
            embedding_dim = 128,
            num_classes = vocab_size,
            classifier_dim = 512,
            num_lstm_layers = 1,
            word_embedding = self.encoder.word_embed,
            dropout = 0.1,
            lstm_dropout = 0.1
        )
        
    def forward(self, image, words, max_length, word_padding_masks = None, answer_tokens = None, teacher_forcing_ratio = 0.5):
        encoded = self.encoder(image, words, word_padding_masks)
        
        decoded = self.decoder(encoded, max_length, answer_tokens, teacher_forcing_ratio)
        
        return decoded
        