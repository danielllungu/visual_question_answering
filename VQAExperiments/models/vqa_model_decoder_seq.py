import torch
import torch.nn as nn
import pytorch_lightning as pl


class DecoderSeq(pl.LightningModule):
    def __init__(self, hidden_size, image_embedding_size, vocab_size):
        super(DecoderSeq, self).__init__()
        self.hidden_size = hidden_size
        self.image_embedding_size = image_embedding_size
        self.image_word_size = self.hidden_size * 2 + self.image_embedding_size
        self.gru = nn.GRU(self.image_word_size, hidden_size, num_layers=1)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, text_embedding, image_embedding, hidden):  # [b, 768], [b, 2048], [num_l=1, b, 384]
        output = torch.cat([text_embedding, image_embedding], dim=1)  # [b, 2816]
        output = self.relu(output)
        output, hidden = self.gru(output.unsqueeze(0), hidden)  # output [1, b, 384], hidden [num_l = 1, b, 384]
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)
