import torch
import torch.nn as nn
import pytorch_lightning as pl


class VqaModelBertV2(pl.LightningModule):
    def __init__(self, text_embedding_size=300, image_embedding_size=2048):
        super(VqaModelBertV2, self).__init__()
        self.input_lstm = text_embedding_size
        self.hidden_lstm = 512
        self.num_layers = 2

        self.LSTM = nn.LSTM(input_size=self.input_lstm, hidden_size=self.hidden_lstm, num_layers=self.num_layers)

        self.fc = nn.Linear(self.hidden_lstm + (self.num_layers * self.hidden_lstm) + image_embedding_size, 2048)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, self.input_lstm)

        self.layer_norm = nn.LayerNorm(self.input_lstm)
        self.gelu = nn.GELU()

    def forward(self, image_embedding, question_embedding):  # img [batch, 2048], question [batch, seq, text_emb_size]
        question_embedding = torch.permute(question_embedding, (1, 0, 2))
        out, (hn, cn) = self.LSTM(question_embedding)

        hn = torch.permute(hn, (1, 0, 2))
        hn = torch.reshape(hn, (image_embedding.shape[0], self.num_layers*self.hidden_lstm))
        hn_img_out = torch.cat([hn, image_embedding, out[-1, :, :]], dim=1)

        output = self.fc(hn_img_out)
        output = self.fc1(output)
        output = self.fc2(output)

        output = self.gelu(output)
        output = self.layer_norm(output)

        return output

