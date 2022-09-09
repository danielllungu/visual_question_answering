import json
import unittest
import numpy as np
from PIL import Image
from service import utils
from service.config import CFG


class TestEmbeddingsBert(unittest.TestCase):

    def test_embeddings_vocab(self):
        tokenizer, bert_model = utils.get_bert_tokenizer_and_model()
        dict_vocab = utils.get_vocabulary_dict_bert()
        mc_path1 = CFG.REAL_TRAIN_MultipleChoice_PATH
        real_images_train_path = CFG.REAL_TRAIN_IMAGES_PATH
        with open(mc_path1, "r") as file1:
            json_file1 = file1.read()
            mc = json.loads(json_file1)

        a_path1 = CFG.REAL_TRAIN_ANNOTATIONS_PATH
        with open(a_path1, "r") as file2:
            json_file2 = file2.read()
            annotations = json.loads(json_file2)

        index = 0
        for element in mc["questions"]:
            question_ID = str(element["question_id"])
            PIL_image = Image.open(real_images_train_path + r"\COCO_train2014_" +
                                   utils._add_zeros_for_path(element["image_id"]) + ".jpg")
            np_im = np.array(PIL_image)

            if len(np_im.shape) == 3:
                if np_im.shape[2] == 3:
                    for elem in annotations["annotations"]:
                        question_ID_annotation = str(elem["question_id"])
                        if question_ID_annotation == question_ID:
                            answer_embedding = utils.get_bert_embedding_sentence_or_word_not_in_dict(
                                elem["multiple_choice_answer"], bert_model, tokenizer)
                            decoded_answer = utils.get_most_similar_sentence(answer_embedding, dict_vocab)
                            answer_text = utils.verify_and_get_full_sentence(decoded_answer)
                            self.assertEqual(elem["multiple_choice_answer"].lower(), answer_text)

                            break

                    question_embedding = utils.get_bert_embedding_sentence_or_word_not_in_dict(element["question"],
                                                                                         bert_model, tokenizer)
                    decoded_question = utils.get_most_similar_sentence(question_embedding, dict_vocab)
                    question_text = utils.verify_and_get_full_sentence(decoded_question)
                    self.assertEqual(element["question"].lower(), question_text)

                    index = index + 1

                    if index == 10:
                        break


if __name__ == '__main__':
    unittest.main()
