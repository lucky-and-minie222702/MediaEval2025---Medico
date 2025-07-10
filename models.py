from vision_modules import *
from text_modules import *
from modules import *


class MyModel(nn.Module):
    def __init__(self, vocab_size, device):
        super().__init__()
        
        encoder_spawn = lambda: TextImageEncoderLayer(
            embedding_dim = 512,
            feed_forward_dim = 512,
            num_att_heads = 4,
            dropout = 0.1,
        ).to(device)
        
        self.encoder = TextImageEncoder(
            vocab_size = vocab_size,
            max_answer_length = 50,
            padding_idx = 0,
            embedding_dim = 512,
            word_embedding_dim = 128,
            encoder_spawn = encoder_spawn,
            num_layers = 2,
            dropout = 0.1,
        )
        
        self.decoder = AnswerLSTMDecoder(
            in_channels = 512,
            embedding_dim = 64,
            num_classes = vocab_size,
            classifier_dim = 512,
            num_gru_layers = 1,
            word_embedding = self.encoder.word_embed,
            ngram_encoder = self.encoder.ngram_encode,
            dropout = 0.1,
        )
        
    def forward(self, image, questions, max_answer_length, question_padding_masks = None, answers = None, teacher_forcing_ratio = 0.5):
        encoded = self.encoder(image, questions, question_padding_masks)
        
        decoded = self.decoder(encoded, max_answer_length, answers, teacher_forcing_ratio)
        
        return decoded
        