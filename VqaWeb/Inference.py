from service import utils
from PIL import Image


def do_inference(visual_model, vqa_model, fasttext_model, img2features, image_path, question):
    pil_image = Image.open(image_path)
    pil_image = pil_image.convert('RGB')

    if visual_model != 'swin':
        if visual_model != 'resnet50':
            features = img2features.get_features_img2vec(pil_image).squeeze(2).squeeze(2)
        else:
            features = img2features.get_vec(pil_image, tensor=True).squeeze(2).squeeze(2)

    else:
        features = img2features.get_features_img2vec(pil_image)  # Swin Tiny

    embedding = utils.get_text_embedding_matrix(question, fasttext_model)
    embedding = utils.get_padding_question(embedding).unsqueeze(0)

    answer = vqa_model(features, embedding).squeeze().detach().cpu().numpy()
    top5_answers = fasttext_model.similar_by_vector(answer, topn=5)
    return top5_answers

