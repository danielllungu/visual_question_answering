import pytorch_lightning as pl
from models.VQAModelV2 import VqaModelBertV2


class VQANetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.batch_size = 1000
        self.batch_size = 5
        self.net = VqaModelBertV2(text_embedding_size=300, image_embedding_size=768)
        self.train_data = None
        self.val_data = None
        self.TRAIN_LEN = 307836
        self.VALIDATION_LEN = 151314

    def forward(self, img_emb, question_emb):
        embedding = self.net(img_emb, question_emb)
        return embedding

