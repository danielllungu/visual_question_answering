from torch.utils.data import Dataset
import torch
from service import utils


class VqaDataset(Dataset):  # dataset v5 Swin Transformer Tiny - FastText MS COCO V2
    def __init__(self, length, train):
        self.length = length
        self.train = train

    def __getitem__(self, idx):
        if self.train:
            image_emb, question_emb, answer_emb = utils.get_train_index_swin(idx)
        else:
            image_emb, question_emb, answer_emb = utils.get_validation_index_swin(idx)

        question_embedding = utils.get_padding_question(question_emb)
        answer_embedding = utils.get_embedding_average_fasttext(answer_emb)

        return image_emb.type(torch.FloatTensor), question_embedding.type(torch.FloatTensor), answer_embedding.type(torch.FloatTensor)

    def __len__(self):
        return self.length



