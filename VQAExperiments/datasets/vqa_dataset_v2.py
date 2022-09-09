from torch.utils.data import Dataset
import torch
from service import utils


class VQADataset(Dataset):  # dataset v2 ResNet50 - FastText MS COCO V1
    def __init__(self, length, train):
        self.length = length
        self.train = train

    def __getitem__(self, idx):
        if self.train:
            image_emb, question_emb, answer_emb = utils.get_train_index_ms_coco_v1(idx)
        else:
            image_emb, question_emb, answer_emb = utils.get_validation_index_ms_coco_v1(idx)
        question_embedding = utils.get_tensor_matrix_average(question_emb)
        answer_embedding = utils.get_tensor_matrix_average(answer_emb)
        return torch.tensor(image_emb).float(), torch.tensor(question_embedding).float(), torch.tensor(answer_embedding).float()

    def __len__(self):
        return self.length


