import torch
import torch.nn as nn
import pytorch_lightning as pl


# image embedding size = 2048 , question embedding size = matrix (?, 300)
class VQAModel(pl.LightningModule):
    def __init__(self, embedding_dimension, projection_dimension):
        super().__init__()
        self.projection = nn.Linear(embedding_dimension, projection_dimension)
        self.fc = nn.Linear(600, 1024)
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, projection_dimension)
        self.layer_norm = nn.LayerNorm(projection_dimension)
        self.gelu = nn.GELU()

    def forward(self, image_embedding, question_embedding):
        projection = self.projection(image_embedding)
        out = torch.cat([projection, question_embedding], dim=1)
        x = self.fc(out)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.gelu(x)
        x = self.layer_norm(x)

        return x
