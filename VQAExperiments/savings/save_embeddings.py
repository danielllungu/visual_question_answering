import compress_fasttext
import torch
from service import utils
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import pickle


def _add_zeros_for_path(number):
    """

    :param number: int image id
    :return: string image id in dataset
    """
    nr = str(number)
    s = ""
    for i in range(12-len(nr)):
        s = "0"+s
    s = s+nr

    return s


def get_text_embedding_matrix(question, model):
    """
    :param question: the question string
    :param model: pretrained NLP model
    :return: question embedding list of lists
    """
    question = question.lower()
    new_q = "".join(char for char in question if char != "?" and char!="\"" and char!="," and char!="." and char!="_" and char!="(" and char!=")" and char!=":")

    q = new_q.split()
    embeddings = []
    for word in q:
        embedding = model[word]
        embeddings.append(embedding)

    text_embedding_matrix = np.array(embeddings)
    return text_embedding_matrix


def load_fasttext_pretrained_model():
    """

    :return: Fasttext pretrained model
    """
    compressed_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
        'https://github.com/avidale/compress-fasttext/releases/download/v0.0.4/cc.en.300.compressed.bin')
    # https://github.com/avidale/compress-fasttext/releases/download/gensim-4-draft/ft_cc.en.300_freqprune_400K_100K_pq_300.bin
    return compressed_model


def get_bert_embedding(word, bert_model, tokenizer):
    """

    :param word: string word
    :param bert_model: Bert Model
    :param tokenizer: Bert tokenizer
    :return: Bert text embedding
    """
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


def get_cosine_similarity(embedding1, embedding2):
    """

    :param embedding1: tensor word embedding
    :param embedding2: tensor word embedding
    :return: cosine similarity between embedding1 and embedding2
    """
    cosine_similarity_value = F.cosine_similarity(embedding1, embedding2, dim=0)

    return cosine_similarity_value


def get_topn_most_similar_words_bert(topn, word_emb, dict_embeds):
    """

    :param topn: number of words to return
    :param word_emb: tensor word embedding Bert
    :param dict_embeds: Bert dict vocabulary
    :return:
    """
    most_sim_words = []
    most_sim_word = ""
    for index in range(0, topn):
        most_sim_cosine = torch.tensor(0.)
        for elem in dict_embeds.items():
            cos_sim = get_cosine_similarity(word_emb[0][0], elem[1][0][0])
            if cos_sim > most_sim_cosine and elem[0] not in most_sim_words:
                most_sim_word = elem[0]
                most_sim_cosine = cos_sim

        most_sim_words.append(most_sim_word)

    return most_sim_words


def get_most_similar_sentence(text_emb, dict_embeds):
    """
    :param text_emb: List of tensors - Embeddings Bert Vocab of every word or subword
    :param dict_embeds: dictionary vocabulary
    :return: most similar word
    """

    most_sim_word = ""
    most_sim_words = []
    for word_emb in text_emb:
        most_sim_cosine = torch.tensor(0.)
        for elem in dict_embeds.items():
            cos_sim = get_cosine_similarity(word_emb[0][0], elem[1][0][0])
            if cos_sim > most_sim_cosine:
                most_sim_word = elem[0]
                most_sim_cosine = cos_sim

        most_sim_words.append(most_sim_word)

    return most_sim_words


def get_tensors_average(embedding):
    """
    :param embedding: tensor word embedding
    :return: average of tensor on dimension 1
    """
    return torch.mean(embedding, dim=1)


def get_bert_embedding_sentence_or_word_not_in_dict(text, bert_model, tokenizer):
    """

    :param text: string
    :param bert_model: Bert Model
    :param tokenizer: Bert Tokenizer
    :return: Bert word embeddings
    """
    tokenized_text = tokenizer.tokenize(text)
    words_embeddings = []
    for sub_word in tokenized_text:
        word_emb = get_bert_embedding(sub_word, bert_model, tokenizer)
        if word_emb.shape[1] > 1:
            word_emb = get_tensors_average(word_emb).unsqueeze(0)
        words_embeddings.append(word_emb)

    return words_embeddings


def get_vocabulary_dict_bert():
    """

    :return: Dict Vocabulary Bert
    """
    with open('Embeddings Bert Vocab/embeds_updated1.pickle', 'rb') as handle:
        dict_embeds = pickle.load(handle)

    return dict_embeds


def get_bert_tokenizer_and_model():
    """

    :return: Bert Tokenizer and Bert Model
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states=True,
                                      )
    return tokenizer, model


def save_bert_embeddings_v2_only_text():  # All text (including black and white) Bert DONE real train
    """
    Script to save Bert embeddings MS COCO v2
    :return: None
    """
    tokenizer, bert_model = get_bert_tokenizer_and_model()
    real_train = torch.load(r"dataset\COCO V2 Bert Embeddings Text\dataset images questions answers\real_train.pt")
    # path_to_save_train_ssd = r"E:\Dataset\Embeddings\Bert\Train"
    path_to_save_train = r"dataset\COCO V2 Bert Embeddings Text\Train"
    index = 0
    for element in real_train:
        if index > 137300:
            question_embedding = get_bert_embedding_sentence_or_word_not_in_dict(element["question"], bert_model,
                                                                                 tokenizer)
            answer_embedding = get_bert_embedding_sentence_or_word_not_in_dict(element["answer"], bert_model, tokenizer)

            torch.save(question_embedding, path_to_save_train + r"\question_" + str(index) + ".pt")
            torch.save(answer_embedding, path_to_save_train + r"\answer_" + str(index) + ".pt")

            if index % 20000 == 0:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(current_time, " saved ", index)

        index = index + 1


def save_bert_embeddings_v2_text(data_type, tokenizer, bert_model):
    """
    Script to save Bert embeddings
    :param data_type: string: train or validation
    :param tokenizer: Bert tokenizer
    :param bert_model: Bert Language Model
    :return: None
    """
    if data_type == "real val":
        real_val = torch.load(r"dataset\COCO V2 Bert Embeddings Text\dataset images questions answers\real_val.pt")
        path_to_save_val = r"dataset\COCO V2 Bert Embeddings Text\REAL VALIDATION"
        index = 0
        for element in real_val:
            question_embedding = get_bert_embedding_sentence_or_word_not_in_dict(element["question"], bert_model,
                                                                                 tokenizer)
            answer_embedding = get_bert_embedding_sentence_or_word_not_in_dict(element["answer"], bert_model, tokenizer)

            torch.save(question_embedding, path_to_save_val + r"\question_" + str(index) + ".pt")
            torch.save(answer_embedding, path_to_save_val + r"\answer_" + str(index) + ".pt")

            if index % 20000 == 0:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(current_time, " saved ", index)

            index = index + 1

    elif data_type == "abstract train":
        abstract_train = torch.load(r"dataset\COCO V2 Bert Embeddings Text\dataset images questions answers\abstract_train.pt")
        path_to_save_train = r"dataset\COCO V2 Bert Embeddings Text\ABSTRACT TRAIN"
        index = 0
        for element in abstract_train:
            question_embedding = get_bert_embedding_sentence_or_word_not_in_dict(element["question"], bert_model,
                                                                                 tokenizer)
            answer_embedding = get_bert_embedding_sentence_or_word_not_in_dict(element["answer"], bert_model, tokenizer)

            torch.save(question_embedding, path_to_save_train + r"\question_" + str(index) + ".pt")
            torch.save(answer_embedding, path_to_save_train + r"\answer_" + str(index) + ".pt")

            if index % 20000 == 0:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(current_time, " saved ", index)

            index = index + 1

    elif data_type == "abstract val":
        abstract_val = torch.load(r"dataset\COCO V2 Bert Embeddings Text\dataset images questions answers\abstract_validation.pt")
        path_to_save_val = r"dataset\COCO V2 Bert Embeddings Text\ABSTRACT VALIDATION"
        index = 0
        for element in abstract_val:
            question_embedding = get_bert_embedding_sentence_or_word_not_in_dict(element["question"], bert_model,
                                                                                 tokenizer)
            answer_embedding = get_bert_embedding_sentence_or_word_not_in_dict(element["answer"], bert_model, tokenizer)

            torch.save(question_embedding, path_to_save_val + r"\question_" + str(index) + ".pt")
            torch.save(answer_embedding, path_to_save_val + r"\answer_" + str(index) + ".pt")

            if index % 20000 == 0:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(current_time, " saved ", index)

            index = index + 1


def tensor_in_list_of_tensors(tensor, list_tensors):
    """

    :param tensor: tensor to verify
    :param list_tensors: list of tensors
    :return: True if tensor is in list of tensors
    """
    for element in list_tensors:
        if torch.equal(tensor, element):
            return True

    return False


def get_text_embedding_bert(word, tokenizer, bert_embeddings):
    """

    :param word: string word text
    :param tokenizer: Bert tokenizer
    :param bert_embeddings: dict Bert embeddings
    :return:
    """
    tokenized_text = tokenizer.tokenize(word)

    embedding = []
    for sub_word in tokenized_text:
        for elem in bert_embeddings:
            if elem['word'] == sub_word:
                embedding.append(elem['embedding'])

    return embedding


def get_text_from_embedding_bert(list_of_embeddings, bert_embeddings):
    """

    :param list_of_embeddings: embeddings list
    :param bert_embeddings: dict Bert embeddings
    :return:
    """
    text = []
    for word_embedding in list_of_embeddings:
        most_sim_cosine = torch.tensor(0.)
        w = ''
        for elem in bert_embeddings:
            cos_sim = get_cosine_similarity(word_embedding, elem['embedding'])
            if cos_sim > most_sim_cosine:
                most_sim_cosine = cos_sim
                w = elem['word']
        text.append(w)

    return text


def get_topn_text_from_embedding_bert(topn, list_of_embeddings, bert_embeddings):
    """

    :param topn: number of words to return
    :param list_of_embeddings: embeddings list
    :param bert_embeddings: dict bert embeddings
    :return:
    """
    all_most_similar = []
    for word_embedding in list_of_embeddings:
        w = ''
        most_similar = []
        for index in range(0, topn):
            most_sim_cosine = torch.tensor(0.)
            for elem in bert_embeddings:
                cos_sim = get_cosine_similarity(word_embedding, elem['embedding'])
                if cos_sim > most_sim_cosine and elem['word'] not in most_similar:
                    most_sim_cosine = cos_sim
                    w = elem['word']

            most_similar.append(w)

        all_most_similar.append(most_similar)

    return all_most_similar


def get_bert_vocab_embeddings():
    """
    :return: dict Bert embeddings
    """
    bert_embeds = torch.load(r"dataset\COCO V2 Bert Embeddings Text\bert_embeddings_updated.pt")

    return bert_embeds


def save_bert_embeddings(data_type, tokenizer, bert_embeddings):
    """

    :param data_type: string: train or val
    :param tokenizer: Bert Tokenizer
    :param bert_embeddings: dict bert embeddings
    :return:
    """
    source = r"dataset\COCO V2"
    if data_type == "train":
        print("Saving train data")
        data1 = torch.load(source + r"\dataset images questions answers\real_train.pt")
        data2 = torch.load(source + r"\dataset images questions answers\abstract_train.pt")
        path_to_save = source + r"\Language BERT\TRAIN"

    else:
        print("Saving validation data")
        data1 = torch.load(source + r"\dataset images questions answers\real_val.pt")
        data2 = torch.load(source + r"\dataset images questions answers\abstract_validation.pt")
        path_to_save = source + r"\Language BERT\VALIDATION"

    index = 0
    print("real")
    for elem in data1:
        question_embedding = get_text_embedding_bert(elem['question'], tokenizer, bert_embeddings)
        answer_embedding = get_text_embedding_bert(elem['answer'], tokenizer, bert_embeddings)

        torch.save(question_embedding, path_to_save + r"\question_" + str(index) + ".pt")
        torch.save(answer_embedding, path_to_save + r"\answer_" + str(index) + ".pt")
        index = index + 1

        if index % 20000 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time, " read ", index)

    print("abstract")
    for elem in data2:
        question_embedding = get_text_embedding_bert(elem['question'], tokenizer, bert_embeddings)
        answer_embedding = get_text_embedding_bert(elem['answer'], tokenizer, bert_embeddings)

        torch.save(question_embedding, path_to_save + r"\question_" + str(index) + ".pt")
        torch.save(answer_embedding, path_to_save + r"\answer_" + str(index) + ".pt")
        index = index + 1

        if index % 20000 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time, " read ", index)


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


def save_fasttext_embeddings_v2(model, data_type):
    source = r"dataset\COCO V2"
    if data_type == "train":
        data1 = torch.load(source + r"\dataset images questions answers\real_train.pt")
        data2 = torch.load(source + r"\dataset images questions answers\abstract_train.pt")
        path_to_save = r"dataset\Fasttext V2\TRAIN"

    else:
        data1 = torch.load(source + r"\dataset images questions answers\real_val.pt")
        data2 = torch.load(source + r"\dataset images questions answers\abstract_validation.pt")
        path_to_save = r"dataset\Fasttext V2\VALIDATION"

    index = 0
    print("real")
    for elem in data1:
        if data_type == "train":
            if index > 100000:
                question_embedding = get_text_embedding_fasttext(elem['question'], model)
                answer_embedding = get_text_embedding_fasttext(elem['answer'], model)

                torch.save(question_embedding, path_to_save + r"\question_" + str(index) + ".pt")
                torch.save(answer_embedding, path_to_save + r"\answer_" + str(index) + ".pt")

        else:
            question_embedding = get_text_embedding_fasttext(elem['question'], model)
            answer_embedding = get_text_embedding_fasttext(elem['answer'], model)

            torch.save(question_embedding, path_to_save + r"\question_" + str(index) + ".pt")
            torch.save(answer_embedding, path_to_save + r"\answer_" + str(index) + ".pt")

        index = index + 1

        if index % 20000 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time, " saved ", index)

    print("abstract")
    for elem in data2:
        question_embedding = get_text_embedding_fasttext(elem['question'], model)
        answer_embedding = get_text_embedding_fasttext(elem['answer'], model)

        torch.save(question_embedding, path_to_save + r"\question_" + str(index) + ".pt")
        torch.save(answer_embedding, path_to_save + r"\answer_" + str(index) + ".pt")

        index = index + 1

        if index % 20000 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time, " saved ", index)


if __name__ == '__main__':
    fasttext_model = utils.load_fasttext_pretrained_model()
    save_fasttext_embeddings_v2(fasttext_model, "train")
    save_fasttext_embeddings_v2(fasttext_model, "validation")
