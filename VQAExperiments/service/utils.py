import pickle
import numpy as np
import json
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import gensim
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
from service.config import CFG
from PIL import Image
import compress_fasttext
import csv
import torch.nn.functional as F


def _add_zeros_for_path(number):
    nr = str(number)
    s = ""
    for i in range(12-len(nr)):
        s = "0"+s
    s = s+nr

    return s

# TODO : pretrained embs --> accuracy output top 5 most similar ?= answer, percentage
# TODO : feature map, le salvez FLOAT 16 (min max len)


def top5_accuracy(predicted_answer, ground_truth, model):
    predicted_answer_sim = model.wv.most_similar(positive=[predicted_answer], topn=5)
    common_words = 0
    for elem_ans in predicted_answer_sim:
        for elem_predicted in predicted_answer_sim:
            if elem_ans[0] == elem_predicted[0]:
                common_words = common_words + 1

    percentage = common_words / 10


def get_embeddings_train():
    """
    Takes all the saved Embeddings Bert Vocab of images, questions and answers for training
    :return: dictionary of all images - questions Embeddings Bert Vocab pairs with keys "image" and "question"
             list of answers having same indexes for the dictionary pair (image, question)
    """
    all_images_questions = []
    list_answers = []

    # real train
    a_path = CFG.REAL_TRAIN_ANNOTATIONS_PATH

    with open(a_path, "r") as file_a:
        json_file_a = file_a.read()
    annotations = json.loads(json_file_a)

    path_images_embs = CFG.PATH_EMBEDDINGS_REAL_IMAGES_TRAIN
    path_questions_embs = CFG.PATH_EMBEDDINGS_REAL_QUESTIONS_TRAIN
    path_answers_embs = CFG.PATH_EMBEDDINGS_REAL_ANSWERS_TRAIN
    path = CFG.REAL_TRAIN_IMAGES_PATH
    print("Reading real train...")
    idx = 0
    for element_a in annotations["annotations"][:20000]:
        PIL_image = Image.open(path + "/COCO_train2014_" + _add_zeros_for_path(element_a["image_id"]) + ".jpg")
        np_im = np.array(PIL_image)
        if len(np_im.shape) == 3:
            if np_im.shape[2] == 3:
                idx = idx + 1
                if idx % 20000 == 0:
                    print(idx, " files read.")
                path_to_image = path_images_embs + "\\tensor_" + str(element_a["image_id"]) + ".pt"
                path_to_question = path_questions_embs + "\\img_" + str(element_a["image_id"])
                path_to_answer = path_answers_embs + "\\question_" + str(element_a["question_id"]) + "\\answer_emb.pt"

                image_emb = torch.load(path_to_image)
                i_emb = np.array(image_emb)
                pth = path_to_question + "\\question_" + str(element_a["question_id"]) + ".pt"

                question_emb = torch.load(pth)
                q_emb = np.array(question_emb)

                answer_emb = torch.load(path_to_answer)
                a_emb = np.array(answer_emb)

                all_images_questions.append({"image": i_emb, "question": q_emb})
                list_answers.append(a_emb)

    # abstract train
    a_path = CFG.ABSTRACT_TRAIN_ANNOTATIONS_PATH

    with open(a_path, "r") as file_a:
        json_file_a = file_a.read()
    annotations = json.loads(json_file_a)

    path_images_embs = CFG.PATH_EMBEDDINGS_ABSTRACT_IMAGES_TRAIN
    path_questions_embs = CFG.PATH_EMBEDDINGS_ABSTRACT_QUESTIONS_TRAIN
    path_answers_embs = CFG.PATH_EMBEDDINGS_ABSTRACT_ANSWERS_TRAIN
    path = CFG.ABSTRACT_TRAIN_IMAGES_PATH

    print("Reading abstract train...")
    idx = 0
    for element_a in annotations["annotations"][:20000]:
        PIL_image = Image.open(path + r"\abstract_v002_train2015_" + _add_zeros_for_path(
            element_a["image_id"]) + ".png")
        rgb_image = PIL_image.convert('RGB')
        np_im = np.array(rgb_image)
        if len(np_im.shape) == 3:
            if np_im.shape[2] == 3:
                idx = idx + 1
                if idx % 20000 == 0:
                    print(idx, "files read.")

                path_to_image = path_images_embs + "\\tensor_" + str(element_a["image_id"]) + ".pt"
                path_to_question = path_questions_embs + "\\img_" + str(element_a["image_id"])
                path_to_answer = path_answers_embs + "\\question_" + str(element_a["question_id"]) + "\\answer_emb.pt"

                image_emb = torch.load(path_to_image)
                i_emb = np.array(image_emb)
                pth = path_to_question + "\\question_" + str(element_a["question_id"]) + ".pt"

                question_emb = torch.load(pth)
                q_emb = np.array(question_emb)

                answer_emb = torch.load(path_to_answer)
                a_emb = np.array(answer_emb)

                all_images_questions.append({"image": i_emb, "question": q_emb})
                list_answers.append(a_emb)

    return all_images_questions, list_answers


def get_embeddings_validation():
    """
    Takes all the saved Embeddings Bert Vocab of images, questions and answers for validation
    :return: dictionary of all images - questions Embeddings Bert Vocab pairs with keys "image" and "question"
             list of answers having same indexes for the dictionary pair (image, question)
    """
    all_images_questions = []
    list_answers = []

    # real validation
    a_path = CFG.REAL_VAL_ANNOTATIONS_PATH

    with open(a_path, "r") as file_a:
        json_file_a = file_a.read()
    annotations = json.loads(json_file_a)

    path_images_embs = CFG.PATH_EMBEDDINGS_REAL_IMAGES_VALIDATION
    path_questions_embs = CFG.PATH_EMBEDDINGS_REAL_QUESTIONS_VALIDATION
    path_answers_embs = CFG.PATH_EMBEDDINGS_REAL_ANSWERS_VALIDATION
    path = CFG.REAL_VAL_IMAGES_PATH

    print("Reading real validation...")
    idx = 0
    for element_a in annotations["annotations"][:10000]:
        PIL_image = Image.open(path + "/COCO_val2014_" + _add_zeros_for_path(element_a["image_id"]) + ".jpg")
        np_im = np.array(PIL_image)
        if len(np_im.shape) == 3:
            if np_im.shape[2] == 3:
                idx = idx + 1
                if idx % 10000 == 0:
                    print(idx, "files read.")
                path_to_image = path_images_embs + "\\tensor_" + str(element_a["image_id"]) + ".pt"
                path_to_question = path_questions_embs + "\\img_" + str(element_a["image_id"])
                path_to_answer = path_answers_embs + "\\question_" + str(element_a["question_id"]) + "\\answer_emb.pt"

                image_emb = torch.load(path_to_image)
                i_emb = np.array(image_emb)
                pth = path_to_question + "\\question_" + str(element_a["question_id"]) + ".pt"

                question_emb = torch.load(pth)
                q_emb = np.array(question_emb)

                answer_emb = torch.load(path_to_answer)
                a_emb = np.array(answer_emb)

                all_images_questions.append({"image": i_emb, "question": q_emb})
                list_answers.append(a_emb)

    # abstract validation
    a_path = CFG.ABSTRACT_VAL_ANNOTATIONS_PATH

    with open(a_path, "r") as file_a:
        json_file_a = file_a.read()
    annotations = json.loads(json_file_a)

    path_images_embs = CFG.PATH_EMBEDDINGS_ABSTRACT_IMAGES_VALIDATION
    path_questions_embs = CFG.PATH_EMBEDDINGS_ABSTRACT_QUESTIONS_VALIDATION
    path_answers_embs = CFG.PATH_EMBEDDINGS_ABSTRACT_ANSWERS_VALIDATION
    path = CFG.ABSTRACT_VAL_IMAGES_PATH

    print("Reading abstract validation...")
    idx = 0
    for element_a in annotations["annotations"][:10000]:
        PIL_image = Image.open(path + r"\abstract_v002_val2015_" + _add_zeros_for_path(
            element_a["image_id"]) + ".png")
        rgb_image = PIL_image.convert('RGB')
        np_im = np.array(rgb_image)
        if len(np_im.shape) == 3:
            if np_im.shape[2] == 3:
                idx = idx + 1
                if idx % 10000 == 0:
                    print(idx, "files read.")
                path_to_image = path_images_embs + "\\tensor_" + str(element_a["image_id"]) + ".pt"
                path_to_question = path_questions_embs + "\\img_" + str(element_a["image_id"])
                path_to_answer = path_answers_embs + "\\question_" + str(element_a["question_id"]) + "\\answer_emb.pt"

                image_emb = torch.load(path_to_image)
                i_emb = np.array(image_emb)
                pth = path_to_question + "\\question_" + str(element_a["question_id"]) + ".pt"

                question_emb = torch.load(pth)
                q_emb = np.array(question_emb)

                answer_emb = torch.load(path_to_answer)
                a_emb = np.array(answer_emb)

                all_images_questions.append({"image": i_emb, "question": q_emb})
                list_answers.append(a_emb)

    return all_images_questions, list_answers


def get_loader(images_dataset, batch_size, shuffle, num_workers):
    train_loader = DataLoader(images_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader


def initialize_model():
    """
    Function that initializes the word2vec model's vocabulary.
    Iterates through all the questions and answers in dataset.
    :return: word2vec model
    """

    # real images train
    mc_path = CFG.REAL_TRAIN_MultipleChoice_PATH
    f = open(mc_path, "r")
    json_file = f.read()
    mc = json.loads(json_file)

    f.close()

    a_path = CFG.REAL_TRAIN_ANNOTATIONS_PATH
    file_a = open(a_path, "r")
    json_file_a = file_a.read()
    a = json.loads(json_file_a)

    file_a.close()

    list_questions=[]
    list_multiple_choices = []
    list_answers = []

    for element in mc["questions"]:
        q = element["question"]
        multiple_choices = element["multiple_choices"]
        for elem in multiple_choices:
            elm = elem.lower()
            elem_lst = elm.split()
            list_multiple_choices.append(elem_lst)

        q = q.lower()
        question = "".join(char for char in q if char != "?")

        lst = question.split()
        list_questions.append(lst)

    for elem in a["annotations"]:
        answer = elem["multiple_choice_answer"]
        answer = answer.lower()
        new_q = "".join(char for char in answer if char != "?")
        answer_lst = new_q.split()
        list_answers.append(answer_lst)

    data = list_questions + list_multiple_choices + list_answers

    # real images validation
    mc_path = CFG.REAL_VAL_MultipleChoice_PATH
    f = open(mc_path, "r")
    json_file = f.read()
    mc = json.loads(json_file)

    f.close()

    a_path = CFG.REAL_VAL_ANNOTATIONS_PATH
    file_a = open(a_path, "r")
    json_file_a = file_a.read()
    a = json.loads(json_file_a)

    file_a.close()

    list_questions = []
    list_multiple_choices = []
    list_answers = []

    for element in mc["questions"]:
        q = element["question"]
        multiple_choices = element["multiple_choices"]
        for elem in multiple_choices:
            elm = elem.lower()
            elem_lst = elm.split()
            list_multiple_choices.append(elem_lst)

        q = q.lower()
        question = "".join(char for char in q if char != "?")

        lst = question.split()
        list_questions.append(lst)

    for elem in a["annotations"]:
        answer = elem["multiple_choice_answer"]
        answer = answer.lower()
        new_q = "".join(char for char in answer if char != "?")
        answer_lst = new_q.split()
        list_answers.append(answer_lst)

    data = data + list_questions + list_multiple_choices + list_answers

    # abstract images train
    mc_path = CFG.ABSTRACT_TRAIN_MultipleChoice_PATH
    f = open(mc_path, "r")
    json_file = f.read()
    mc = json.loads(json_file)

    f.close()

    a_path = CFG.ABSTRACT_TRAIN_ANNOTATIONS_PATH
    file_a = open(a_path, "r")
    json_file_a = file_a.read()
    a = json.loads(json_file_a)

    file_a.close()

    list_questions = []
    list_multiple_choices = []
    list_answers = []

    for element in mc["questions"]:
        q = element["question"]
        multiple_choices = element["multiple_choices"]
        for elem in multiple_choices:
            elm = elem.lower()
            elem_lst = elm.split()
            list_multiple_choices.append(elem_lst)

        q = q.lower()
        question = "".join(char for char in q if char != "?")

        lst = question.split()
        list_questions.append(lst)

    for elem in a["annotations"]:
        answer = elem["multiple_choice_answer"]
        answer = answer.lower()
        new_q = "".join(char for char in answer if char != "?")
        answer_lst = new_q.split()
        list_answers.append(answer_lst)

    data = data + list_questions + list_multiple_choices + list_answers

    # abstract images validation
    mc_path = CFG.ABSTRACT_VAL_MultipleChoice_PATH
    f = open(mc_path, "r")
    json_file = f.read()
    mc = json.loads(json_file)

    f.close()

    a_path = CFG.ABSTRACT_VAL_ANNOTATIONS_PATH
    file_a = open(a_path, "r")
    json_file_a = file_a.read()
    a = json.loads(json_file_a)

    file_a.close()

    list_questions = []
    list_multiple_choices = []
    list_answers = []

    for element in mc["questions"]:
        q = element["question"]
        multiple_choices = element["multiple_choices"]
        for elem in multiple_choices:
            elm = elem.lower()
            elem_lst = elm.split()
            list_multiple_choices.append(elem_lst)

        q = q.lower()
        question = "".join(char for char in q if char != "?")

        lst = question.split()
        list_questions.append(lst)

    for elem in a["annotations"]:
        answer = elem["multiple_choice_answer"]
        answer = answer.lower()
        new_q = "".join(char for char in answer if char != "?")
        answer_lst = new_q.split()
        list_answers.append(answer_lst)

    data = data + list_questions + list_multiple_choices + list_answers
    model = Word2Vec(data, min_count = 1, vector_size=1024, window = 5, sg=0)

    return model


def get_text_embedding(question, model):
    question = question.lower()
    new_q = ""
    for char in question:
        if char != "?":
            new_q = new_q + char

    q = new_q.split()
    embeddings = []
    for word in q:
        embedding = model.wv[word]
        embeddings.append(embedding)

    question_embedding = []
    for index_elm in range(0, len(embeddings[0])):
        s = 0
        for idx in range(0, len(embeddings)):
            s = s + embeddings[idx][index_elm]
        d = s / len(embeddings)
        question_embedding.append(d)

    return question_embedding


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


def get_avg_of_tensors(tensor1, tensor2):
    a = tensor1.detach().cpu().numpy()
    b = tensor2.detach().cpu().numpy()
    x = np.mean(np.array([a, b]), axis=0)
    mean = torch.tensor(x)

    return mean


def get_tensor_matrix_average(text_embedding):
    #a = text.detach().cpu().numpy()
    sum_rows = np.sum(text_embedding, axis=0)
    mean = sum_rows/2
    #mean_tensor = torch.tensor(mean)

    return mean


def get_tensors_average(embedding):
    return torch.mean(embedding, dim=1)


def get_bert_embedding_sentence_or_word_not_in_dict(text, bert_model, tokenizer):
    tokenized_text = tokenizer.tokenize(text)
    words_embeddings = []
    for sub_word in tokenized_text:
        word_emb = get_bert_embedding(sub_word, bert_model, tokenizer)
        if word_emb.shape[1] > 1:
            word_emb = get_tensors_average(word_emb).unsqueeze(0)
        words_embeddings.append(word_emb)

    return words_embeddings


def get_bert_tokenizer_and_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states=True,
                                      )
    return tokenizer, model


def get_bert_embedding(word, bert_model, tokenizer):
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_text = tokenizer.tokenize(word)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model = bert_model

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # hidden_states = outputs[2]
        last_hidden_state = outputs[0]

    return last_hidden_state


def get_question_input_padding_bert(question_embeddings_list):
    max_len_question = 27
    question = torch.cat(question_embeddings_list, dim=1)
    question = question.squeeze(0)
    zeros = torch.zeros((max_len_question - question.shape[0], 768))
    question = torch.cat([question, zeros], dim=0)

    return question


def get_answer_padding_bert(answer_embeddings_list):
    max_len_answer = 19
    answer = torch.cat(answer_embeddings_list, dim=1)
    answer = answer.squeeze(0)
    zeros = torch.zeros((max_len_answer - answer.shape[0], 768))
    answer = torch.cat([answer, zeros], dim=0)

    return answer


def get_answer_average_bert(answer_list):
    answer = torch.cat(answer_list, dim=1)
    answer = answer.squeeze(0)
    return torch.mean(answer, dim=0)


def get_seq2seq_question(question_words_list, sep_token_embedding):
    question = torch.cat(question_words_list, dim=1)
    question = question.squeeze(0)
    max_len_question = 27

    sep_token_embedding = sep_token_embedding.squeeze(0)  # [1, 768]
    question = torch.cat([question, sep_token_embedding], dim=0)

    zeros = torch.zeros((max_len_question - question.shape[0], 768))
    question = torch.cat([question, zeros], dim=0)
    # [seq = 27, 768]
    return question


def get_seq2seq_answer(answer_words_list, sep_token_embedding):
    answer = torch.cat(answer_words_list, dim=1)
    answer = answer.squeeze(0)
    max_len_answer = 19

    sep_token_embedding = sep_token_embedding.squeeze(0)  # [1, 768]
    answer = torch.cat([answer, sep_token_embedding], dim=0)

    zeros = torch.zeros((max_len_answer - answer.shape[0], 768))
    answer = torch.cat([answer, zeros], dim=0)
    # [seq = 19, 768]
    return answer


def get_sep_token_embedding():
    """
    :return: end of string token embedding [SEP]
    """
    bert_dict = get_vocabulary_dict_bert()
    for elem in bert_dict.items():
        if elem[0] == "[SEP]":
            sep_emb = elem[1]
            return sep_emb


def get_CLS_TOKEN(bert_dict):
    cuda0 = torch.device('cuda:0')
    for elem in bert_dict.items():
        if elem[0] == "[CLS]":
            cls = elem[1]
            return cls.to(cuda0)


def get_cls_token_embedding(batch_size):
    """
        :return: start of string token embedding [CLS]
    """
    bert_dict = get_vocabulary_dict_bert()
    for elem in bert_dict.items():
        if elem[0] == "[CLS]":
            cat = elem[1]
            for index in range(0, batch_size-1):
                cat = torch.cat([cat, elem[1]], dim=1)
            cls_emb = cat.detach().cpu().numpy()
            return torch.tensor(cls_emb, device=torch.device('cuda:0'))


def get_vocab_size_bert():
    """

    :return: Bert Vocabulary Size
    """
    bert_dict = get_vocabulary_dict_bert()
    return len(bert_dict.items())


def get_word2index_bert(bert_dict):
    """

    :param bert_dict: Bert Vocab Dict: key: word, value: embedding
    :return: word2index Dict : key: word, value: index
    """
    word2index = {}

    index = 0
    for elem in bert_dict.items():
        word2index[elem[0]] = torch.tensor([index])
        index = index + 1

    return word2index


def get_word2index_index2word_bert(bert_dict):
    """

    :param bert_dict: Bert Vocab Dict: key: word, value: embedding
    :return: word2index Dict : key: word, value: index
    """
    word2index = {}
    index2word = {}
    index = 0
    for elem in bert_dict.items():
        word2index[elem[0]] = torch.tensor([index])
        index2word[index] = elem[0]
        index = index + 1

    return word2index, index2word


def trim_batch_question_bert(question_embedding_batch):
    """
    Gets the padding of the longest sequence in batch
    :param question_embedding_batch: batch of question embeddings shape: [batch_size, seq = 27, 768]
    :return: batch question embedding shape [batch_size, seq = longest in batch, 768]
    """
    max_question_len = question_embedding_batch.shape[1]

    zeros = torch.zeros(768, device=torch.device("cuda:0"))
    question_final_index = max_question_len
    for index in range(0, max_question_len):
        all_zero = True
        for question_word_embedding in question_embedding_batch[:, index]:
            if not torch.equal(question_word_embedding, zeros):
                all_zero = False

        if all_zero:
            question_final_index = index
            break

    return question_embedding_batch[:, :question_final_index, :]


def trim_batch_answer_bert(answer_embedding_batch):
    """
    Gets the padding of the longest sequence in batch
    :param answer_embedding_batch: batch of answer embeddings shape: [batch_size, seq = 19, 768]
    :return: batch answer embedding shape [batch_size, seq = longest in batch, 768]
    """
    max_answer_len = answer_embedding_batch.shape[1]

    zeros = torch.zeros(768, device=torch.device("cuda:0"))
    question_final_index = max_answer_len
    for index in range(0, max_answer_len):
        all_zero = True
        for answer_word_embedding in answer_embedding_batch[:, index]:
            if not torch.equal(answer_word_embedding, zeros):
                all_zero = False

        if all_zero:
            question_final_index = index
            break

    return answer_embedding_batch[:, :question_final_index, :]


def get_embedding2index_bert(word2index, bert_dict, embedding):
    """

    :param word2index: word 2 index Bert Dict
    :param bert_dict: Bert Vocab Dict
    :param embedding: text embedding shape [batch_size, 768]
    :return: index of the embedding in vocab
    """
    words2find = []
    indexes = []
    emb = torch.tensor(embedding.detach().cpu().numpy())

    for word_embedding in emb:
        word_emb = word_embedding.unsqueeze(0).unsqueeze(0)
        for elem in bert_dict.items():
            if torch.equal(elem[1], word_emb):
                words2find.append(elem[0])
                break

    for list_elem in words2find:
        indexes.append(word2index[list_elem])

    return torch.tensor(indexes, device=torch.device('cuda:0'))


def get_matrix_average(emb_text):
    sum_rows = np.sum(emb_text, axis=0)
    mean = sum_rows / 2

    return mean


def get_embedding_average_fasttext(embedding):
    text_emb = torch.tensor(embedding)
    return torch.mean(text_emb, dim=0)


def get_answers_dictionary():
    list_answers = []

    # real train
    a_path = CFG.REAL_TRAIN_ANNOTATIONS_PATH

    with open(a_path, "r") as file_a:
        json_file_a = file_a.read()
    annotations = json.loads(json_file_a)

    path_answers_embs = CFG.PATH_EMBEDDINGS_REAL_ANSWERS_TRAIN
    path = CFG.REAL_TRAIN_IMAGES_PATH

    for element_a in annotations["annotations"][:1]:
        PIL_image = Image.open(path + "/COCO_train2014_" + _add_zeros_for_path(element_a["image_id"]) + ".jpg")
        np_im = np.array(PIL_image)
        if len(np_im.shape) == 3:
            if np_im.shape[2] == 3:
                path_to_answer = path_answers_embs + "\\question_" + str(element_a["question_id"]) + "\\answer_emb.pt"
                answer_emb = torch.load(path_to_answer)
                a_emb = np.array(answer_emb)

                list_answers.append(a_emb)

    # abstract train
    a_path = CFG.ABSTRACT_TRAIN_ANNOTATIONS_PATH

    with open(a_path, "r") as file_a:
        json_file_a = file_a.read()
    annotations = json.loads(json_file_a)

    path_answers_embs = CFG.PATH_EMBEDDINGS_ABSTRACT_ANSWERS_TRAIN
    path = CFG.ABSTRACT_TRAIN_IMAGES_PATH

    for element_a in annotations["annotations"][:1]:
        PIL_image = Image.open(path + r"\abstract_v002_train2015_" + _add_zeros_for_path(
            element_a["image_id"]) + ".png")
        rgb_image = PIL_image.convert('RGB')
        np_im = np.array(rgb_image)
        if len(np_im.shape) == 3:
            if np_im.shape[2] == 3:
                path_to_answer = path_answers_embs + "\\question_" + str(element_a["question_id"]) + "\\answer_emb.pt"
                answer_emb = torch.load(path_to_answer)
                a_emb = np.array(answer_emb)
                list_answers.append(a_emb)

    list_answers_validation = []

    # real validation
    a_path = CFG.REAL_VAL_ANNOTATIONS_PATH

    with open(a_path, "r") as file_a:
        json_file_a = file_a.read()
    annotations = json.loads(json_file_a)

    path_answers_embs = CFG.PATH_EMBEDDINGS_REAL_ANSWERS_VALIDATION
    path = CFG.REAL_VAL_IMAGES_PATH

    for element_a in annotations["annotations"][:1]:
        PIL_image = Image.open(path + "/COCO_val2014_" + _add_zeros_for_path(element_a["image_id"]) + ".jpg")
        np_im = np.array(PIL_image)
        if len(np_im.shape) == 3:
            if np_im.shape[2] == 3:
                path_to_answer = path_answers_embs + "\\question_" + str(element_a["question_id"]) + "\\answer_emb.pt"
                answer_emb = torch.load(path_to_answer)
                a_emb = np.array(answer_emb)
                list_answers_validation.append(a_emb)

    # abstract validation
    a_path = CFG.ABSTRACT_VAL_ANNOTATIONS_PATH

    with open(a_path, "r") as file_a:
        json_file_a = file_a.read()
    annotations = json.loads(json_file_a)

    path_answers_embs = CFG.PATH_EMBEDDINGS_ABSTRACT_ANSWERS_VALIDATION
    path = CFG.ABSTRACT_VAL_IMAGES_PATH

    for element_a in annotations["annotations"][:1]:
        PIL_image = Image.open(path + r"\abstract_v002_val2015_" + _add_zeros_for_path(
            element_a["image_id"]) + ".png")
        rgb_image = PIL_image.convert('RGB')
        np_im = np.array(rgb_image)
        if len(np_im.shape) == 3:
            if np_im.shape[2] == 3:
                path_to_answer = path_answers_embs + "\\question_" + str(element_a["question_id"]) + "\\answer_emb.pt"
                answer_emb = torch.load(path_to_answer)
                a_emb = np.array(answer_emb)
                list_answers_validation.append(a_emb)

    # answers_dictionary_list = []
    # for answer_train in list_answers:
    #     response = []
    #     mean = get_matrix_average(answer_train)
    #     for emb_word in answer_train:
    #         response.append(model.wv.most_similar(positive=[emb_word], topn=1))
    #     entire_response = ""
    #     for elem in response:
    #         entire_response = entire_response + " " + elem[0][0]
    #     print(entire_response)

    return list_answers, list_answers_validation


def get_cosine_similarity_Bert(embedding1, embedding2):
    cosine_similarity_value = F.cosine_similarity(embedding1, embedding2, dim=0)

    return cosine_similarity_value


def get_most_similar_word_bert(word_emb, bert_dict):
    """

    :param word_emb: word embedding: shape [1, 768]
    :return: string most similar word
    """
    for elem in bert_dict.items():
        if torch.equal(elem[1], word_emb.unsqueeze(0)):
            return elem[0]

    return ""


def get_most_similar_sentence(text_emb, dict_embeds):
    """

    :param text_emb: List of tensors - Embeddings Bert Vocab of every word or subword
    :param dict_embeds: dictionary vocabulary
    :return: most similar words list
    """

    most_sim_word = ""
    most_sim_words = []
    for word_emb in text_emb:
        most_sim_cosine = torch.tensor(0.)
        for elem in dict_embeds.items():
            cos_sim = get_cosine_similarity_Bert(word_emb[0][0], elem[1][0][0])
            if cos_sim > most_sim_cosine:
                most_sim_word = elem[0]
                most_sim_cosine = cos_sim

        most_sim_words.append(most_sim_word)

    return most_sim_words


def get_most_similar_answer(answer_emb, dict_embeds):
    """

    :param answer_emb: List of tensors - Embeddings Bert Vocab of every word or subword
    :param dict_embeds: dictionary vocabulary
    :return: most similar words list
    """

    most_sim_word = ""
    most_sim_words = []
    for word_emb in answer_emb:
        most_sim_cosine = torch.tensor(0.)
        for elem in dict_embeds.items():
            cos_sim = get_cosine_similarity_Bert(word_emb, elem[1][0][0])
            if cos_sim > most_sim_cosine:
                most_sim_word = elem[0]
                most_sim_cosine = cos_sim

        most_sim_words.append(most_sim_word)

    return most_sim_words


def get_top5_most_similar_bert(answer_emb, dict_embeds):
    """

    :param answer_emb: List of tensors - Embeddings Bert Vocab of every word or subword
    :param dict_embeds: dictionary vocabulary
    :return: most similar words list
    """

    most_sim_word = ""
    top5 = []
    for word_emb in answer_emb:
        for index in range(0,5):
            most_sim_cosine = torch.tensor(0.)
            for elem in dict_embeds.items():
                cos_sim = get_cosine_similarity_Bert(word_emb, elem[1][0][0])
                if cos_sim > most_sim_cosine and [elem[0], cos_sim] not in top5:
                    most_sim_word = elem[0]
                    most_sim_cosine = cos_sim

            top5.append([most_sim_word, most_sim_cosine])

    return top5


def get_vocabulary_dict_bert():
    with open('Embeddings Bert Vocab/embeds_updated1.pickle', 'rb') as handle:
        dict_embeds = pickle.load(handle)

    return dict_embeds


def verify_and_get_full_sentence(emb_list):
    text = ""
    for sub_word in emb_list:
        if sub_word in ['.', ',', ';', '\'', '\"', '[', ']', '(', ')', '{', '}', '?', '!']:
            text = text + sub_word
        else:
            if len(sub_word) > 1:
                if sub_word[0] + sub_word[1] != "##":
                    if text == "":
                        text = text + sub_word
                    else:
                        text = text + " " + sub_word
                else:
                    if text == "":
                        text = text + sub_word
                    else:
                        text = text + sub_word[2:]
            else:
                if text != "":
                    if text[-1] == "\'" and text[-2] != 's':
                        text = text + sub_word
                    else:
                        text = text + " " + sub_word

                else:
                    text = text + sub_word

    return text


def get_train_batch_index(index):
    # real train
    a_path = CFG.REAL_TRAIN_ANNOTATIONS_PATH

    with open(a_path, "r") as file_a:
        json_file_a = file_a.read()
    annotations = json.loads(json_file_a)

    path_images_embs = CFG.PATH_EMBEDDINGS_REAL_IMAGES_TRAIN
    path_questions_embs = CFG.PATH_EMBEDDINGS_REAL_QUESTIONS_TRAIN
    path_answers_embs = CFG.PATH_EMBEDDINGS_REAL_ANSWERS_TRAIN
    path = CFG.REAL_TRAIN_IMAGES_PATH

    idx = 0
    for element_a in annotations["annotations"]:
        PIL_image = Image.open(path + "/COCO_train2014_" + _add_zeros_for_path(element_a["image_id"]) + ".jpg")
        np_im = np.array(PIL_image)
        if len(np_im.shape) == 3:
            if np_im.shape[2] == 3:
                path_to_image = path_images_embs + "\\tensor_" + str(element_a["image_id"]) + ".pt"
                path_to_question = path_questions_embs + "\\img_" + str(element_a["image_id"])
                path_to_answer = path_answers_embs + "\\question_" + str(element_a["question_id"]) + "\\answer_emb.pt"

                image_emb = torch.load(path_to_image)
                i_emb = np.array(image_emb)
                pth = path_to_question + "\\question_" + str(element_a["question_id"]) + ".pt"

                question_emb = torch.load(pth)
                q_emb = np.array(question_emb)

                answer_emb = torch.load(path_to_answer)
                a_emb = np.array(answer_emb)

                if idx == index:
                    return i_emb, q_emb, a_emb

                idx = idx + 1

    # abstract train
    a_path = CFG.ABSTRACT_TRAIN_ANNOTATIONS_PATH

    with open(a_path, "r") as file_a:
        json_file_a = file_a.read()
    annotations = json.loads(json_file_a)

    path_images_embs = CFG.PATH_EMBEDDINGS_ABSTRACT_IMAGES_TRAIN
    path_questions_embs = CFG.PATH_EMBEDDINGS_ABSTRACT_QUESTIONS_TRAIN
    path_answers_embs = CFG.PATH_EMBEDDINGS_ABSTRACT_ANSWERS_TRAIN
    path = CFG.ABSTRACT_TRAIN_IMAGES_PATH

    for element_a in annotations["annotations"]:
        PIL_image = Image.open(path + r"\abstract_v002_train2015_" + _add_zeros_for_path(
            element_a["image_id"]) + ".png")
        rgb_image = PIL_image.convert('RGB')
        np_im = np.array(rgb_image)
        if len(np_im.shape) == 3:
            if np_im.shape[2] == 3:
                path_to_image = path_images_embs + "\\tensor_" + str(element_a["image_id"]) + ".pt"
                path_to_question = path_questions_embs + "\\img_" + str(element_a["image_id"])
                path_to_answer = path_answers_embs + "\\question_" + str(element_a["question_id"]) + "\\answer_emb.pt"

                image_emb = torch.load(path_to_image)
                i_emb = np.array(image_emb)
                pth = path_to_question + "\\question_" + str(element_a["question_id"]) + ".pt"

                question_emb = torch.load(pth)
                q_emb = np.array(question_emb)

                answer_emb = torch.load(path_to_answer)
                a_emb = np.array(answer_emb)

                if idx == index:
                    return i_emb, q_emb, a_emb

                idx = idx + 1


def get_validation_batch_index(index):
    # real validation
    a_path = CFG.REAL_VAL_ANNOTATIONS_PATH

    with open(a_path, "r") as file_a:
        json_file_a = file_a.read()
    annotations = json.loads(json_file_a)

    path_images_embs = CFG.PATH_EMBEDDINGS_REAL_IMAGES_VALIDATION
    path_questions_embs = CFG.PATH_EMBEDDINGS_REAL_QUESTIONS_VALIDATION
    path_answers_embs = CFG.PATH_EMBEDDINGS_REAL_ANSWERS_VALIDATION
    path = CFG.REAL_VAL_IMAGES_PATH

    idx = 0
    for element_a in annotations["annotations"]:
        PIL_image = Image.open(path + "/COCO_val2014_" + _add_zeros_for_path(element_a["image_id"]) + ".jpg")
        np_im = np.array(PIL_image)
        if len(np_im.shape) == 3:
            if np_im.shape[2] == 3:
                path_to_image = path_images_embs + "\\tensor_" + str(element_a["image_id"]) + ".pt"
                path_to_question = path_questions_embs + "\\img_" + str(element_a["image_id"])
                path_to_answer = path_answers_embs + "\\question_" + str(element_a["question_id"]) + "\\answer_emb.pt"

                image_emb = torch.load(path_to_image)
                i_emb = np.array(image_emb)
                pth = path_to_question + "\\question_" + str(element_a["question_id"]) + ".pt"

                question_emb = torch.load(pth)
                q_emb = np.array(question_emb)

                answer_emb = torch.load(path_to_answer)
                a_emb = np.array(answer_emb)

                if idx == index:
                    return i_emb, q_emb, a_emb

                idx = idx + 1

    # abstract validation
    a_path = CFG.ABSTRACT_VAL_ANNOTATIONS_PATH

    with open(a_path, "r") as file_a:
        json_file_a = file_a.read()
    annotations = json.loads(json_file_a)

    path_images_embs = CFG.PATH_EMBEDDINGS_ABSTRACT_IMAGES_VALIDATION
    path_questions_embs = CFG.PATH_EMBEDDINGS_ABSTRACT_QUESTIONS_VALIDATION
    path_answers_embs = CFG.PATH_EMBEDDINGS_ABSTRACT_ANSWERS_VALIDATION
    path = CFG.ABSTRACT_VAL_IMAGES_PATH

    for element_a in annotations["annotations"]:
        PIL_image = Image.open(path + r"\abstract_v002_val2015_" + _add_zeros_for_path(
            element_a["image_id"]) + ".png")
        rgb_image = PIL_image.convert('RGB')
        np_im = np.array(rgb_image)
        if len(np_im.shape) == 3:
            if np_im.shape[2] == 3:

                path_to_image = path_images_embs + "\\tensor_" + str(element_a["image_id"]) + ".pt"
                path_to_question = path_questions_embs + "\\img_" + str(element_a["image_id"])
                path_to_answer = path_answers_embs + "\\question_" + str(element_a["question_id"]) + "\\answer_emb.pt"

                image_emb = torch.load(path_to_image)
                i_emb = np.array(image_emb)
                pth = path_to_question + "\\question_" + str(element_a["question_id"]) + ".pt"

                question_emb = torch.load(pth)
                q_emb = np.array(question_emb)

                answer_emb = torch.load(path_to_answer)
                a_emb = np.array(answer_emb)

                if idx == index:
                    return i_emb, q_emb, a_emb
                idx = idx + 1


def get_len_train_data():
    # real train
    a_path = CFG.REAL_TRAIN_ANNOTATIONS_PATH

    with open(a_path, "r") as file_a:
        json_file_a = file_a.read()
    annotations = json.loads(json_file_a)
    path = CFG.REAL_TRAIN_IMAGES_PATH

    idx = 0
    print("Calculating length of train data... ")
    for element_a in annotations["annotations"]:
        PIL_image = Image.open(path + "/COCO_train2014_" + _add_zeros_for_path(element_a["image_id"]) + ".jpg")
        np_im = np.array(PIL_image)
        if len(np_im.shape) == 3:
            if np_im.shape[2] == 3:
                idx = idx + 1
                if idx % 20000 == 0:
                    print(idx, "files read.")

    # abstract train
    a_path = CFG.ABSTRACT_TRAIN_ANNOTATIONS_PATH

    with open(a_path, "r") as file_a:
        json_file_a = file_a.read()
    annotations = json.loads(json_file_a)
    path = CFG.ABSTRACT_TRAIN_IMAGES_PATH

    for element_a in annotations["annotations"]:
        PIL_image = Image.open(path + r"\abstract_v002_train2015_" + _add_zeros_for_path(
            element_a["image_id"]) + ".png")
        rgb_image = PIL_image.convert('RGB')
        np_im = np.array(rgb_image)
        if len(np_im.shape) == 3:
            if np_im.shape[2] == 3:
                idx = idx + 1
                if idx % 20000 == 0:
                    print(idx, "files read.")

    print(idx, "total train files.")

    return idx


def get_len_validation_data():
    # real validation
    a_path = CFG.REAL_VAL_ANNOTATIONS_PATH

    with open(a_path, "r") as file_a:
        json_file_a = file_a.read()
    annotations = json.loads(json_file_a)
    path = CFG.REAL_VAL_IMAGES_PATH

    idx = 0
    print("Calculating length of validation data... ")
    for element_a in annotations["annotations"]:
        PIL_image = Image.open(path + "/COCO_val2014_" + _add_zeros_for_path(element_a["image_id"]) + ".jpg")
        np_im = np.array(PIL_image)
        if len(np_im.shape) == 3:
            if np_im.shape[2] == 3:
                idx = idx + 1
                if idx % 10000 == 0:
                    print(idx, "files read.")

    # abstract validation
    a_path = CFG.ABSTRACT_VAL_ANNOTATIONS_PATH

    with open(a_path, "r") as file_a:
        json_file_a = file_a.read()
    annotations = json.loads(json_file_a)
    path = CFG.ABSTRACT_VAL_IMAGES_PATH

    for element_a in annotations["annotations"]:
        PIL_image = Image.open(path + r"\abstract_v002_val2015_" + _add_zeros_for_path(
            element_a["image_id"]) + ".png")
        rgb_image = PIL_image.convert('RGB')
        np_im = np.array(rgb_image)
        if len(np_im.shape) == 3:
            if np_im.shape[2] == 3:
                idx = idx + 1
                if idx % 10000 == 0:
                    print(idx, "files read.")

    print(idx, "total validation files.")

    return idx


def get_train_index_word2vec(index):
    path_to_image = r"C:\Datasets\Pretrained FastText Embeddings\train\image_" + str(index) + ".pt"
    path_to_question = r"C:\Datasets\Pretrained FastText Embeddings\train\question_" + str(index) + ".pt"
    path_to_answer = r"C:\Datasets\Pretrained FastText Embeddings\train\answer_" + str(index) + ".pt"

    image_emb = torch.load(path_to_image)

    question_emb = torch.load(path_to_question)
    q_emb = np.array(question_emb)

    answer_emb = torch.load(path_to_answer)
    a_emb = np.array(answer_emb)

    return image_emb, q_emb, a_emb


def get_validation_index_word2vec(index):
    path_to_image = r"C:\Datasets\Pretrained FastText Embeddings\validation\image_" + str(index) + ".pt"
    path_to_question = r"C:\Datasets\Pretrained FastText Embeddings\validation\question_" + str(index) + ".pt"
    path_to_answer = r"C:\Datasets\Pretrained FastText Embeddings\validation\answer_" + str(index) + ".pt"

    image_emb = torch.load(path_to_image)

    question_emb = torch.load(path_to_question)
    q_emb = np.array(question_emb)

    answer_emb = torch.load(path_to_answer)
    a_emb = np.array(answer_emb)

    return image_emb, q_emb, a_emb


def get_train_index_ms_coco_v2(index):
    path_to_image = r"C:\Datasets\Pretrained FastText Embeddings\train\image_" + str(index) + ".pt"
    path_to_question = r"C:\Datasets\Pretrained FastText Embeddings\train\question_" + str(index) + ".pt"
    path_to_answer = r"C:\Datasets\Pretrained FastText Embeddings\train\answer_" + str(index) + ".pt"

    image_emb = torch.load(path_to_image)
    # i_emb = np.array(image_emb)

    question_emb = torch.load(path_to_question)
    q_emb = np.array(question_emb)

    answer_emb = torch.load(path_to_answer)
    a_emb = np.array(answer_emb)

    return image_emb, q_emb, a_emb


def get_validation_index_ms_coco_v2(index):
    path_to_image = r"C:\Datasets\Pretrained FastText Embeddings\validation\image_" + str(index) + ".pt"
    path_to_question = r"C:\Datasets\Pretrained FastText Embeddings\validation\question_" + str(index) + ".pt"
    path_to_answer = r"C:\Datasets\Pretrained FastText Embeddings\validation\answer_" + str(index) + ".pt"

    image_emb = torch.load(path_to_image)
    # i_emb = np.array(image_emb)

    question_emb = torch.load(path_to_question)
    q_emb = np.array(question_emb)

    answer_emb = torch.load(path_to_answer)
    a_emb = np.array(answer_emb)

    return image_emb, q_emb, a_emb


def get_train_index_ms_coco_v1(index):
    path_to_image = r"C:\Datasets\Pretrained FastText Embeddings\train\image_" + str(index) + ".pt"
    path_to_question = r"C:\Datasets\Pretrained FastText Embeddings\train\question_" + str(index) + ".pt"
    path_to_answer = r"C:\Datasets\Pretrained FastText Embeddings\train\answer_" + str(index) + ".pt"

    image_emb = torch.load(path_to_image)
    # i_emb = np.array(image_emb)

    question_emb = torch.load(path_to_question)
    q_emb = np.array(question_emb)

    answer_emb = torch.load(path_to_answer)
    a_emb = np.array(answer_emb)

    return image_emb, q_emb, a_emb


def get_validation_index_ms_coco_v1(index):
    path_to_image = r"C:\Datasets\Pretrained FastText Embeddings\validation\image_" + str(index) + ".pt"
    path_to_question = r"C:\Datasets\Pretrained FastText Embeddings\validation\question_" + str(index) + ".pt"
    path_to_answer = r"C:\Datasets\Pretrained FastText Embeddings\validation\answer_" + str(index) + ".pt"

    image_emb = torch.load(path_to_image)
    # i_emb = np.array(image_emb)

    question_emb = torch.load(path_to_question)
    q_emb = np.array(question_emb)

    answer_emb = torch.load(path_to_answer)
    a_emb = np.array(answer_emb)

    return image_emb, q_emb, a_emb


def get_train_index_bert(index):
    path_to_image = r"C:\Datasets\Pretrained FastText Embeddings\train\image_" + str(index) + ".pt"
    path_to_question = r"C:\Datasets\Pretrained BERT Embeddings\train\question_" + str(index) + ".pt"
    path_to_answer = r"C:\Datasets\Pretrained BERT Embeddings\train\answer_" + str(index) + ".pt"

    image_embedding = torch.load(path_to_image)
    question_embeddings_list = torch.load(path_to_question)
    answer_embedding_list = torch.load(path_to_answer)

    return image_embedding, question_embeddings_list, answer_embedding_list


def get_validation_index_bert(index):
    path_to_image = r"C:\Datasets\Pretrained FastText Embeddings\validation\image_" + str(index) + ".pt"
    path_to_question = r"C:\Datasets\Pretrained BERT Embeddings\validation\question_" + str(index) + ".pt"
    path_to_answer = r"C:\Datasets\Pretrained BERT Embeddings\validation\answer_" + str(index) + ".pt"

    image_embedding = torch.load(path_to_image)
    question_embeddings_list = torch.load(path_to_question)
    answer_embedding_list = torch.load(path_to_answer)

    return image_embedding, question_embeddings_list, answer_embedding_list


def load_word2vec_model():
    model = Word2Vec.load(r"WORD2VEC_MODEL\word2vec_model.model")

    return model


def load_google_word2vec_model():
    print("Loading model...")
    model = gensim.models.KeyedVectors.load_word2vec_format("Google_word2vec_model/GoogleNews-vectors-negative300.bin", binary=True)
    print("Word2Vec model loaded.")
    return model


def load_fasttext_pretrained_model():
    compressed_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
        'https://github.com/avidale/compress-fasttext/releases/download/v0.0.4/cc.en.300.compressed.bin')
    # https://github.com/avidale/compress-fasttext/releases/download/gensim-4-draft/ft_cc.en.300_freqprune_400K_100K_pq_300.bin
    return compressed_model


def get_accuracy_one_by_one(predicted_answer, answer, model):
    pred_ans = predicted_answer.detach().cpu().numpy()
    ans = answer.detach().cpu().numpy()
    pred_a = model.similar_by_vector(pred_ans, topn=1)
    a = model.similar_by_vector(ans, topn=1)

    if pred_a[0][0] == a[0][0]:
        return 1

    return 0


def get_cosine_similarity(ans, predicted_ans):
    # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # output = cos(ans, predicted_ans)

    cosine_similarity_value = F.cosine_similarity(ans, predicted_ans, dim=1)

    return torch.mean(cosine_similarity_value)


def get_accuracy_batch(predicted_answer_batch, answer_batch, fasttext_model):
    result = []
    for index in range(answer_batch.shape[0]):
        answer = answer_batch[index]
        predicted_answer = predicted_answer_batch[index]
        result.append(torch.tensor(get_accuracy_one_by_one(predicted_answer, answer, fasttext_model), dtype=torch.float32,
                                       device=torch.device('cuda:0')))
        # result.append(get_accuracy_one_by_one(predicted_answer, answer, fasttext_model))

    accuracy = torch.tensor(sum(result) / answer_batch.shape[0], dtype=torch.float32, device=torch.device('cuda:0'))
    return accuracy


def save_accuracy(lst_answers):
    # torch.save(lst_answers, r"C:\Datasets\Accuracy\FastText Experiment\v1_" + str(epoch) + ".pt")

    header = ['ANSWER', 'PREDICTED_ANSWER']
    data = lst_answers

    with open(r"C:\Datasets\Accuracy\FastText Experiment\v1.csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        writer.writerow(data)


def verify_dataset_vocab(word2vec_model):
    unique_words = []
    # real images train
    print("Verifying real train...")
    mc_path = CFG.REAL_TRAIN_MultipleChoice_PATH
    f = open(mc_path, "r")
    json_file = f.read()
    mc = json.loads(json_file)

    f.close()

    a_path = CFG.REAL_TRAIN_ANNOTATIONS_PATH
    file_a = open(a_path, "r")
    json_file_a = file_a.read()
    a = json.loads(json_file_a)

    file_a.close()

    for element in mc["questions"]:
        q = element["question"]

        q = q.lower()
        question = "".join(char for char in q if char != "?" and char!="\"" and char!="," and char!="." and char!="_" and char!="(" and char!=")" and char!=":")

        lst_words_question = question.split()

        for word_question in lst_words_question:
            final_word = word_question.replace(" ", "")
            if final_word not in word2vec_model.index_to_key:
                if final_word not in unique_words:
                    unique_words.append(word_question)

    for elem in a["annotations"]:
        answer = elem["multiple_choice_answer"]
        answer = answer.lower()
        new_q = "".join(char for char in answer if char != "?" and char!="\"" and char!="," and char!="." and char!="_" and char!="(" and char!=")" and char!=":")
        answer_lst = new_q.split()

        for word_answer in answer_lst:
            final_word = word_answer.replace(" ", "")
            if final_word not in word2vec_model.index_to_key:
                if final_word not in unique_words:
                    unique_words.append(word_answer)

    # real images validation
    print("Verifying real validation...")
    mc_path = CFG.REAL_VAL_MultipleChoice_PATH
    f = open(mc_path, "r")
    json_file = f.read()
    mc = json.loads(json_file)

    f.close()

    a_path = CFG.REAL_VAL_ANNOTATIONS_PATH
    file_a = open(a_path, "r")
    json_file_a = file_a.read()
    a = json.loads(json_file_a)

    file_a.close()

    for element in mc["questions"]:
        q = element["question"]
        q = q.lower()
        question = "".join(char for char in q if char != "?" and char!="\"" and char!="," and char!="." and char!="_" and char!="(" and char!=")" and char!=":")

        lst_words_question1 = question.split()

        for word_question in lst_words_question1:
            final_word = word_question.replace(" ", "")
            if final_word not in word2vec_model.index_to_key:
                if final_word not in unique_words:
                    unique_words.append(word_question)

    for elem in a["annotations"]:
        answer = elem["multiple_choice_answer"]
        answer = answer.lower()
        new_q = "".join(char for char in answer if char != "?" and char!="\"" and char!="," and char!="." and char!="_" and char!="(" and char!=")" and char!=":")
        answer_lst = new_q.split()
        for word_answer in answer_lst:
            final_word = word_answer.replace(" ", "")
            if final_word not in word2vec_model.index_to_key:
                if final_word not in unique_words:
                    unique_words.append(word_answer)

    # abstract images train
    print("Verifying abstract train...")
    mc_path = CFG.ABSTRACT_TRAIN_MultipleChoice_PATH
    f = open(mc_path, "r")
    json_file = f.read()
    mc = json.loads(json_file)

    f.close()

    a_path = CFG.ABSTRACT_TRAIN_ANNOTATIONS_PATH
    file_a = open(a_path, "r")
    json_file_a = file_a.read()
    a = json.loads(json_file_a)

    file_a.close()

    for element in mc["questions"]:
        q = element["question"]

        q = q.lower()
        question = "".join(char for char in q if char != "?" and char!="\"" and char!="," and char!="." and char!="_" and char!="(" and char!=")" and char!=":")

        lst_words_question = question.split()

        for word_question in lst_words_question:
            final_word = word_question.replace(" ", "")
            if final_word not in word2vec_model.index_to_key:
                if final_word not in unique_words:
                    unique_words.append(word_question)

    for elem in a["annotations"]:
        answer = elem["multiple_choice_answer"]
        answer = answer.lower()
        new_q = "".join(char for char in answer if char != "?" and char!="\"" and char!="," and char!="." and char!="_" and char!="(" and char!=")" and char!=":")
        answer_lst = new_q.split()

        for word_answer in answer_lst:
            final_word = word_answer.replace(" ", "")
            if final_word not in word2vec_model.index_to_key:
                if final_word not in unique_words:
                    unique_words.append(word_answer)

    # abstract images validation
    print("Verifying abstract validation...")
    mc_path = CFG.ABSTRACT_VAL_MultipleChoice_PATH
    f = open(mc_path, "r")
    json_file = f.read()
    mc = json.loads(json_file)

    f.close()

    a_path = CFG.ABSTRACT_VAL_ANNOTATIONS_PATH
    file_a = open(a_path, "r")
    json_file_a = file_a.read()
    a = json.loads(json_file_a)

    file_a.close()

    for element in mc["questions"]:
        q = element["question"]

        q = q.lower()
        question = "".join(char for char in q if char != "?" and char!="\"" and char!="," and char!="." and char!="_" and char!="(" and char!=")" and char!=":")

        lst_words_question = question.split()

        for word_question in lst_words_question:
            final_word = word_question.replace(" ", "")
            if final_word not in word2vec_model.index_to_key:
                if final_word not in unique_words:
                    unique_words.append(word_question)

    for elem in a["annotations"]:
        answer = elem["multiple_choice_answer"]
        answer = answer.lower()
        new_q = "".join(char for char in answer if char != "?" and char!="\"" and char!="," and char!="." and char!="_" and char!="(" and char!=")" and char!=":")
        answer_lst = new_q.split()

        for word_answer in answer_lst:
            final_word = word_answer.replace(" ", "")
            if final_word not in word2vec_model.index_to_key:
                if final_word not in unique_words:
                    unique_words.append(word_answer)

    print(len(unique_words))
    torch.save(unique_words, r"WORDS NOT IN VOCABULARY\unique_words_com.pt")
    print("saved")


def get_max_sequence_questions_answers():
    TRAIN_LEN = 307836
    VALIDATION_LEN = 151314
    questions_max_len = 0
    answers_max_len = 0

    for index in range(0, TRAIN_LEN):
        _, question_embedding, answer_embedding = get_train_index_ms_coco_v1(index)
        if question_embedding.shape[0] > questions_max_len:
            questions_max_len = question_embedding.shape[0]
            print("Max length modified questions :", questions_max_len)

        if answer_embedding.shape[0] > answers_max_len:
            answers_max_len = answer_embedding.shape[0]
            print("Max length modified answers :", answers_max_len)

    for index in range(0, VALIDATION_LEN):
        _, question_embedding, answer_embedding = get_validation_index_ms_coco_v1(index)
        if question_embedding.shape[0] > questions_max_len:
            questions_max_len = question_embedding.shape[0]
            print("Max length modified questions :", questions_max_len)

        if answer_embedding.shape[0] > answers_max_len:
            answers_max_len = answer_embedding.shape[0]
            print("Max length modified answers :", answers_max_len)

    torch.save(questions_max_len, r"C:\Datasets\Pretrained FastText Embeddings\max_lengths\questions_max_len.pt")
    torch.save(answers_max_len, r"C:\Datasets\Pretrained FastText Embeddings\max_lengths\answers_max_len.pt")

    # 23 questions 14 answers


def get_padding_question(question_ndarray):
    max_len_seq_question = 23
    question = torch.tensor(question_ndarray, dtype=torch.float32)
    zeros = torch.zeros((max_len_seq_question - question.shape[0], 300))
    out = torch.cat([question, zeros], dim=0)
    return out


def get_padding_question_fasttext(question):
    max_len_seq_question = 23
    zeros = torch.zeros((max_len_seq_question - question.shape[0], 300))
    out = torch.cat([question, zeros], dim=0)
    return out


def get_padding_answer(answer_ndarray):
    max_len_seq_answer = 14
    answer = torch.tensor(answer_ndarray)
    zeros = torch.zeros((max_len_seq_answer - answer.shape[0], 300))
    out = torch.cat([answer, zeros], dim=0)

    return out


def show_final_features(model, image_path):
    print(model.default_cfg)
    image = Image.open(image_path)
    image = torch.as_tensor(np.array(image, dtype=np.float32)).transpose(2, 0)[None]
    out = model.forward_features(image)

    plt.imshow(out[0].transpose(0, 2).sum(-1).detach().numpy())
    plt.show()


def show_feature_maps(model, image_path):
    print(model.feature_info.module_name())
    print(model.feature_info.reduction())
    print(model.feature_info.channels())

    image = Image.open(image_path)
    image = torch.as_tensor(np.array(image, dtype=np.float32)).transpose(2, 0)[None]

    out = model(image)
    print(len(out))
    print("FEATURE MAPS shapes")
    for o in out:
        print(o.shape)

    for o in out:
        plt.imshow(o[0].transpose(0, 2).sum(-1).detach().numpy())
        plt.show()


def get_feature_maps(model, image_path):
    image = Image.open(image_path)
    image = torch.as_tensor(np.array(image, dtype=np.float32)).transpose(2, 0)[None]

    out = model(image)

    outputs = []
    for o in out:
        outputs.append(o)

    return outputs


def get_train_index_regnety(index):
    source = CFG.PATH_TRAIN_REGNETY_MS_COCO_V2_IMAGES
    source_language = CFG.PATH_TRAIN_REGNETY_MS_COCO_V2_TEXT

    path_to_image = source + str(index) + ".pt"
    path_to_question = source_language + str(index) + ".pt"
    path_to_answer = source_language + str(index) + ".pt"

    image_emb = torch.load(path_to_image)
    question_emb = torch.load(path_to_question)
    q_emb = np.array(question_emb)

    answer_emb = torch.load(path_to_answer)
    a_emb = np.array(answer_emb)

    return image_emb, q_emb, a_emb


def get_validation_index_regnety(index):
    source = CFG.PATH_VAL_REGNETY_MS_COCO_V2_IMAGES
    source_language = CFG.PATH_VAL_REGNETY_MS_COCO_V2_TEXT

    path_to_image = source + str(index) + ".pt"
    path_to_question = source_language + str(index) + ".pt"
    path_to_answer = source_language + str(index) + ".pt"

    image_emb = torch.load(path_to_image)
    question_emb = torch.load(path_to_question)
    q_emb = np.array(question_emb)

    answer_emb = torch.load(path_to_answer)
    a_emb = np.array(answer_emb)

    return image_emb, q_emb, a_emb


def get_train_index_ssl(index):
    source = CFG.PATH_TRAIN_SSL_RESNET50_MS_COCO_V2_IMAGES
    source_language = CFG.PATH_TRAIN_SSL_RESNET50_MS_COCO_V2_TEXT

    path_to_image = source + str(index) + ".pt"
    path_to_question = source_language + str(index) + ".pt"
    path_to_answer = source_language + str(index) + ".pt"

    image_emb = torch.load(path_to_image)
    question_emb = torch.load(path_to_question)
    q_emb = np.array(question_emb)

    answer_emb = torch.load(path_to_answer)
    a_emb = np.array(answer_emb)

    return image_emb, q_emb, a_emb


def get_validation_index_ssl(index):
    source = CFG.PATH_VAL_SSL_RESNET50_MS_COCO_V2_IMAGES
    source_language = CFG.PATH_VAL_SSL_RESNET50_MS_COCO_V2_TEXT

    path_to_image = source + str(index) + ".pt"
    path_to_question = source_language + str(index) + ".pt"
    path_to_answer = source_language + str(index) + ".pt"

    image_emb = torch.load(path_to_image)
    question_emb = torch.load(path_to_question)
    q_emb = np.array(question_emb)

    answer_emb = torch.load(path_to_answer)
    a_emb = np.array(answer_emb)

    return image_emb, q_emb, a_emb


def get_train_index_swin(index):
    source = CFG.PATH_TRAIN_SWIN_TINY_MS_COCO_V2_IMAGES
    source_language = CFG.PATH_TRAIN_SWIN_TINY_MS_COCO_V2_TEXT

    path_to_image = source + str(index) + ".pt"
    path_to_question = source_language + str(index) + ".pt"
    path_to_answer = source_language + str(index) + ".pt"

    image_emb = torch.load(path_to_image)
    question_emb = torch.load(path_to_question)
    q_emb = np.array(question_emb)

    answer_emb = torch.load(path_to_answer)
    a_emb = np.array(answer_emb)

    return image_emb, q_emb, a_emb


def get_validation_index_swin(index):
    source = CFG.PATH_VAL_SWIN_TINY_MS_COCO_V2_IMAGES
    source_language = CFG.PATH_VAL_SWIN_TINY_MS_COCO_V2_TEXT

    path_to_image = source + str(index) + ".pt"
    path_to_question = source_language + str(index) + ".pt"
    path_to_answer = source_language + str(index) + ".pt"

    image_emb = torch.load(path_to_image)
    question_emb = torch.load(path_to_question)
    q_emb = np.array(question_emb)

    answer_emb = torch.load(path_to_answer)
    a_emb = np.array(answer_emb)

    return image_emb, q_emb, a_emb


def get_train_index_convnext(index):
    source = CFG.PATH_TRAIN_CONVNEXT_TINY_MS_COCO_V2_IMAGES
    source_language = CFG.PATH_TRAIN_CONVNEXT_TINY_MS_COCO_V2_TEXT

    path_to_image = source + str(index) + ".pt"
    path_to_question = source_language + str(index) + ".pt"
    path_to_answer = source_language + str(index) + ".pt"

    image_emb = torch.load(path_to_image)
    question_emb = torch.load(path_to_question)
    q_emb = np.array(question_emb)

    answer_emb = torch.load(path_to_answer)
    a_emb = np.array(answer_emb)

    return image_emb, q_emb, a_emb


def get_validation_index_convnext(index):
    source = CFG.PATH_VAL_CONVNEXT_TINY_MS_COCO_V2_IMAGES
    source_language = CFG.PATH_VAL_CONVNEXT_TINY_MS_COCO_V2_TEXT

    path_to_image = source + str(index) + ".pt"
    path_to_question = source_language + str(index) + ".pt"
    path_to_answer = source_language + str(index) + ".pt"

    image_emb = torch.load(path_to_image)
    question_emb = torch.load(path_to_question)
    q_emb = np.array(question_emb)

    answer_emb = torch.load(path_to_answer)
    a_emb = np.array(answer_emb)

    return image_emb, q_emb, a_emb


def get_train_index_resnest(index):
    source = CFG.PATH_TRAIN_RESNEST50D_MS_COCO_V2_IMAGES
    source_language = CFG.PATH_TRAIN_RESNEST50D_MS_COCO_V2_TEXT

    path_to_image = source + str(index) + ".pt"
    path_to_question = source_language + str(index) + ".pt"
    path_to_answer = source_language + str(index) + ".pt"

    image_emb = torch.load(path_to_image)
    question_emb = torch.load(path_to_question)
    q_emb = np.array(question_emb)

    answer_emb = torch.load(path_to_answer)
    a_emb = np.array(answer_emb)

    return image_emb, q_emb, a_emb


def get_validation_index_resnest(index):
    source = CFG.PATH_VAL_RESNEST50D_MS_COCO_V2_IMAGES
    source_language = CFG.PATH_VAL_RESNEST50D_MS_COCO_V2_TEXT

    path_to_image = source + str(index) + ".pt"
    path_to_question = source_language + str(index) + ".pt"
    path_to_answer = source_language + str(index) + ".pt"

    image_emb = torch.load(path_to_image)
    question_emb = torch.load(path_to_question)
    q_emb = np.array(question_emb)

    answer_emb = torch.load(path_to_answer)
    a_emb = np.array(answer_emb)

    return image_emb, q_emb, a_emb