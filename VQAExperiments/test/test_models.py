from service import utils
from models.image_features import ImageFeatures
from PIL import Image


def inference(vqa_model, fasttext_model, img2features, image_path, question, swin=False):
    pil_image = Image.open(image_path)
    if swin:
        features = img2features.get_features_img2vec(pil_image)  # Swin Tiny
    else:
        features = img2features.get_features_img2vec(pil_image).squeeze(2).squeeze(2)

    embedding = utils.get_text_embedding_fasttext(question, fasttext_model)

    print(features.shape)
    print(embedding.shape)
    embedding = utils.get_padding_question(embedding).unsqueeze(0)
    # vqa_model = vqa_model.double()
    answer = vqa_model(features, embedding).squeeze().detach().cpu().numpy()
    print(answer.shape)
    top5_answers = fasttext_model.similar_by_vector(answer, topn=5)

    print(top5_answers)


def inference_swin_tiny(path, question_text):
    from run.run_exp_v5_swin_to_fasttext import VQANetwork
    img2feat = ImageFeatures(is_transformer=True, do_norm=False)
    vqa = VQANetwork.load_from_checkpoint(r"logs/comparing_models/VQA_exp_Swin_tiny/weights/VQA_exp_Swin_tiny/version_0/checkpoints/epoch=69-step=8610.ckpt")
    fasttext = utils.load_fasttext_pretrained_model()
    inference(vqa, fasttext, img2feat, path, question_text, swin=True)


def inference_ssl_resnet50(path, question_text):
    from run.run_exp_v8_ssl_resnet50_to_fasttext import VQANetwork
    img2feat = ImageFeatures(model_name='ssl_resnet50', do_norm=False)
    vqa = VQANetwork.load_from_checkpoint(
        r"logs\comparing_models\VQA_exp_SSL_ResNet50\weights\VQA_exp_SSL_ResNet50\version_1\checkpoints\epoch=49-step=12300.ckpt")
    fasttext = utils.load_fasttext_pretrained_model()
    inference(vqa, fasttext, img2feat, path, question_text)


def inference_regnety(path, question_text):
    from run.run_exp_v6_regnety_to_fasttext import VQANetwork
    img2feat = ImageFeatures(model_name='regnety_040', do_norm=False)
    vqa = VQANetwork.load_from_checkpoint(
        r"logs\comparing_models\VQA_exp_RegNetY\weights\VQA_exp_RegNetY\version_2\checkpoints\epoch=69-step=17220.ckpt")
    fasttext = utils.load_fasttext_pretrained_model()
    inference(vqa, fasttext, img2feat, path, question_text)


def inference_resnest50d(path, question_text):
    from run.run_exp_v9_resnest_to_fasttext import VQANetwork
    img2feat = ImageFeatures(model_name='resnest50d', do_norm=False)
    vqa = VQANetwork.load_from_checkpoint(
        r"logs\comparing_models\VQA_exp_ResNest50d\weights\VQA_exp_ResNest50d\version_0\checkpoints\epoch=79-step=19680.ckpt")
    fasttext = utils.load_fasttext_pretrained_model()
    inference(vqa, fasttext, img2feat, path, question_text)


if __name__ == '__main__':
    source = r"C:\Users\lungu\OneDrive\Desktop\demo\grey truck.jpg"
    inference_resnest50d(source, "What is this?")
    inference_ssl_resnet50(source, "What is this?")
