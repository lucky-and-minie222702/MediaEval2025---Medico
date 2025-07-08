from vision_modules import *
from text_modules import *
from modules import *


class MyGRUModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        self.encoder = TextImageEncoder(
            vocab_size = vocab_size,
            max_answer_length = 50,
            padding_idx = 0,
            embedding_dim = 256,
            feed_forward_dim = 256,
            word_embedding_dim = 64,
            num_att_heads = 8,
            dropout = 0.1,
            # attention_dropout = 0.1,
        )
        
        self.decoder = AnswerGRUDecoder(
            in_channels = 256,
            embedding_dim = 64,
            num_classes = vocab_size,
            classifier_dim = 256,
            num_gru_layers = 1,
            word_embedding = self.encoder.word_embed,
            ngram_encoder = self.encoder.ngram_encode,
            dropout = 0.1,
            # gru_dropout = 0.1
        )
        
    def forward(self, image, questions, max_answer_length, question_padding_masks = None, answers = None, teacher_forcing_ratio = 0.5):
        encoded = self.encoder(image, questions, question_padding_masks)
        
        decoded = self.decoder(encoded, max_answer_length, answers, teacher_forcing_ratio)
        
        return decoded
        