import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from service import utils
import pytorch_lightning.loggers as loggers
from datasets.vqa_dataset_v2 import VQADataset
from models.vqa_model_v1 import VQAModel


class VQANetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = VQAModel(2048, 300)
        self.train_data = None
        self.val_data = None
        self.TRAIN_LEN = 307836
        self.VALIDATION_LEN = 151314
        self.fasttext_model = utils.load_fasttext_pretrained_model()

    def forward(self, img_emb, question_emb):
        embedding = self.net(img_emb, question_emb)
        return embedding

    def prepare_data(self):
        self.train_data = VQADataset(self.TRAIN_LEN, train=True)
        self.val_data = VQADataset(self.VALIDATION_LEN, train=False)

    def train_dataloader(self):
        return utils.get_loader(self.train_data, batch_size=30000, shuffle=True, num_workers=3)

    def val_dataloader(self):
        return utils.get_loader(self.val_data, batch_size=30000, shuffle=False, num_workers=3)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        img_emb, q_emb, ans_emb = train_batch
        image_embedding = img_emb.squeeze()
        answer_hat = self.net(image_embedding, q_emb)
        loss = F.mse_loss(answer_hat, ans_emb)

        cosine_similarity = utils.get_cosine_similarity(ans_emb, answer_hat)
        self.log('train_loss', loss)
        self.log("cosine_similarity_average_batch_train", cosine_similarity)

        loss_dictionary = {
            "loss": loss,
            "cosine_similarity": cosine_similarity.detach()
        }

        return loss_dictionary

    def training_epoch_end(self, outputs):

        lst_loss_tensor = torch.tensor([elem["loss"] for elem in outputs], dtype=torch.float32, device=torch.device('cuda:0'))
        lst_cosine_tensor = torch.tensor([elem["cosine_similarity"] for elem in outputs], dtype=torch.float32, device=torch.device('cuda:0'))

        avg_loss = torch.mean(lst_loss_tensor)
        avg_cosine_similarity = torch.mean(lst_cosine_tensor)

        self.log("Epoch training loss", avg_loss)
        self.log("Cosine similarity average per epoch train", avg_cosine_similarity)

    def validation_step(self, val_batch, batch_idx):
        # img_emb [batch, 2048] , q_emb matrix (batch, seq, 1024), a_emb (batch, seq, 1024)

        img_emb, q_emb, ans_emb = val_batch
        image_embedding = img_emb.squeeze()
        answer_hat = self.net(image_embedding, q_emb)
        val_loss = F.mse_loss(answer_hat, ans_emb)
        self.log('val_loss', val_loss)

        cosine_similarity_val = utils.get_cosine_similarity(ans_emb, answer_hat)

        val_dict = {
            "val_loss": val_loss,
            "cosine_similarity_val": cosine_similarity_val.detach()
        }
        return val_dict

    def validation_epoch_end(self, outputs):
        lst_loss_tensor = torch.tensor([elem["val_loss"] for elem in outputs], dtype=torch.float32,
                                       device=torch.device('cuda:0'))
        lst_cosine_tensor = torch.tensor([elem["cosine_similarity_val"] for elem in outputs], dtype=torch.float32,
                                         device=torch.device('cuda:0'))

        avg_val_loss = torch.mean(lst_loss_tensor)
        avg_val_cosine_similarity = torch.mean(lst_cosine_tensor)

        self.log("Epoch validation loss", avg_val_loss)
        self.log("Cosine similarity average per epoch validation", avg_val_cosine_similarity)


if __name__ == '__main__':
    model = VQANetwork()
    log_path = r"logs"
    experiment_name = "VQA_exp_FastText"
    tb_logger = loggers.TensorBoardLogger(log_path, experiment_name)
    save_dir = log_path + "\\" + experiment_name
    trainer = pl.Trainer(gpus=1, max_epochs=18, logger=tb_logger,
                         weights_save_path=save_dir + "\weights", log_every_n_steps=5)
    trainer.fit(model)

