import numpy as np
import torch
import compress_fasttext
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.VQAmodel import VQANetwork as VqaNet
from models.image_features import ImageFeatures
from models.run_ssl_resnet import VQANetwork as VqaSSL
from models.run_regnety import VQANetwork as VqaRegnety
from models.run_resnest50d import VQANetwork as VqaResnest
from models.run_swin_tiny import VQANetwork as VqaSwin
from models.run_convnext_tiny import VQANetwork as VqaConvNext
from models.img2vec import Img2Vec as img2vec
from config import CFG


def get_text_embedding_fasttext(question, model):
    """
        :param question: the question string
        :param model: initialized word2vec model
        :return: question embedding list of lists
        """
    question = question.lower()
    new_q = "".join(char for char in question if
                    char != "?" and char != "\"" and char != "," and char != "." and char != "_" and char != "(" and char != ")" and char != ":")

    q = new_q.split()
    embeddings = []
    for word in q:
        embedding = model[word]
        embeddings.append(embedding)

    text_embedding_matrix = np.array(embeddings)
    return torch.tensor(text_embedding_matrix)


def get_text_embedding_matrix(question, model):
    """
        :param question: the question string
        :param model: initialized word2vec model
        :return: question embedding list of lists
        """
    question = question.lower()
    new_q = "".join(char for char in question if
                    char != "?" and char != "\"" and char != "," and char != "." and char != "_" and char != "(" and char != ")" and char != ":")

    q = new_q.split()
    embeddings = []
    for word in q:
        embedding = model[word]
        embeddings.append(embedding)

    text_embedding_matrix = np.array(embeddings)
    return text_embedding_matrix


def get_padding_question(question_ndarray):
    max_len_seq_question = 23
    question = torch.tensor(question_ndarray, dtype=torch.float32)
    zeros = torch.zeros((max_len_seq_question - question.shape[0], 300))
    out = torch.cat([question, zeros], dim=0)

    return out


def load_fasttext_pretrained_model():
    compressed_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
        'https://github.com/avidale/compress-fasttext/releases/download/v0.0.4/cc.en.300.compressed.bin')
    # https://github.com/avidale/compress-fasttext/releases/download/gensim-4-draft/ft_cc.en.300_freqprune_400K_100K_pq_300.bin
    return compressed_model


def load_fasttext_from_pc():
    compress_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
        r"compressed_fasttext_pretrained\cc.en.300.compressed.bin"
    )

    return compress_model


def get_loader(images_dataset, batch_size, shuffle, num_workers):
    train_loader = DataLoader(images_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader


def get_cosine_similarity(ans, predicted_ans):

    cosine_similarity_value = F.cosine_similarity(ans, predicted_ans, dim=1)

    return torch.mean(cosine_similarity_value)


def get_trained_vqa_model(checkpoint):
    vqa_model = VqaNet.load_from_checkpoint(checkpoint)

    return vqa_model


def get_image2feat(model_name):
    img2feat_ssl = ImageFeatures('ssl_resnet50', do_norm=False)
    img2feat_regnety = ImageFeatures('regnety_040', do_norm=False)
    img2feat_resnest = ImageFeatures('resnest50d', do_norm=False)
    img2feat_swin = ImageFeatures(is_transformer=True, do_norm=False)
    img2feat_convnext = ImageFeatures('convnext_tiny', do_norm=False)
    img2feat_resnet50 = img2vec(model='resnet50', layer_output_size=2048)

    checkpoint_ssl_resnet50 = CFG.ssl_resnet50_checkpoint
    checkpoint_regnety = CFG.regnety_checkpoint
    checkpoint_resnest50d = CFG.resnest_checkpoint
    checkpoint_swin = CFG.swin_checkpoint
    checkpoint_convnext = CFG.convnext_checkpoint
    checkpoint_resnet50 = CFG.resnet50_checkpoint

    vqa_ssl = VqaSSL.load_from_checkpoint(checkpoint_ssl_resnet50)
    vqa_regnety = VqaRegnety.load_from_checkpoint(checkpoint_regnety)
    vqa_resnest = VqaResnest.load_from_checkpoint(checkpoint_resnest50d)
    vqa_swin = VqaSwin.load_from_checkpoint(checkpoint_swin)
    vqa_convnext = VqaConvNext.load_from_checkpoint(checkpoint_convnext)
    vqa_resnet50 = VqaNet.load_from_checkpoint(checkpoint_resnet50)

    if model_name == 'ssl':
        return img2feat_ssl, vqa_ssl
    elif model_name == 'regnety':
        return img2feat_regnety, vqa_regnety
    elif model_name == 'resnest':
        return img2feat_resnest, vqa_resnest
    elif model_name == 'swin':
        return img2feat_swin, vqa_swin
    elif model_name == 'convnext':
        return img2feat_convnext, vqa_convnext
    else:
        return img2feat_resnet50, vqa_resnet50
