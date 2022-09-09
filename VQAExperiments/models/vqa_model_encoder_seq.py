import torch
import torch.nn as nn
import pytorch_lightning as pl


class EncoderSeq(pl.LightningModule):
    def __init__(self, text_embedding_size):
        super(EncoderSeq, self).__init__()
        self.input_size = text_embedding_size
        self.hidden_size = text_embedding_size // 4
        self.num_layers = 1
        self.gru = nn.GRU(self.input_size, self.hidden_size, num_layers=self.num_layers)

    def forward(self, question_embedding_word, hidden):
        question_embedding_word = question_embedding_word.unsqueeze(0)  # [1, batch_size, 768]
        output, hidden = self.gru(question_embedding_word, hidden)  # hidden = [num_layers, batch_size, 384]
        # output = [1, batch_size, 384]
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=torch.device('cuda:0'))
