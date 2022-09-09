import torch
from models.img2vec import Img2Vec as img2vec
from PIL import Image
from models.image_features import ImageFeatures
from service import utils
from run.run_exp_v8_ssl_resnet50_to_fasttext import VQANetwork as VqaSSL
from run.run_exp_v6_regnety_to_fasttext import VQANetwork as VqaRegnety
from run.run_exp_v9_resnest_to_fasttext import VQANetwork as VqaResnest
from run.run_exp_v5_swin_to_fasttext import VQANetwork as VqaSwin
from run.run_exp_v7_convnext_to_fasttext import VQANetwork as VqaConvNext
from run.run_exp_v4_resnet50_to_fasttext import VQANetwork as VqaResNet50LastVersion
import timm
import torchvision.transforms as transforms


def test_features():
    image_path = r"C:\Users\lungu\OneDrive\Desktop\demo\boat1.jpg"
    image = Image.open(image_path)
    # ssl_resnet50 convnext_tiny

    img_features = ImageFeatures(model_name='ssl_resnet50', do_norm=True)
    print("\nssl_resnet50 TIMM NORM True")
    with torch.no_grad():
        outputs = img_features.get_features_img2vec(image)
        print("\ntimm shape adaptive avg pool", outputs.shape)
        print("OUT mean TIMM", torch.mean(outputs))
        print("OUT sum TIMM", torch.sum(outputs))
        print("OUT min TIMM", torch.min(outputs))
        print("OUT max TIMM", torch.max(outputs))
        print("STD", torch.std(outputs, dim=1))
        print("\n")

    img_features = ImageFeatures(model_name='ssl_resnet50', do_norm=False)
    print("\nssl_resnet50 TIMM NORM False")
    with torch.no_grad():
        outputs = img_features.get_features_img2vec(image)
        print("\ntimm shape adaptive avg pool", outputs.shape)
        print("OUT mean TIMM", torch.mean(outputs))
        print("OUT sum TIMM", torch.sum(outputs))
        print("OUT min TIMM", torch.min(outputs))
        print("OUT max TIMM", torch.max(outputs))
        print("STD", torch.std(outputs, dim=1))
        print("\n")

    img_features = ImageFeatures(model_name='regnety_040', do_norm=True)
    print("\nregnety_040 TIMM NORM True")
    with torch.no_grad():
        outputs = img_features.get_features_img2vec(image)
        print("\ntimm shape adaptive avg pool", outputs.shape)
        print("OUT mean TIMM", torch.mean(outputs))
        print("OUT sum TIMM", torch.sum(outputs))
        print("OUT min TIMM", torch.min(outputs))
        print("OUT max TIMM", torch.max(outputs))
        print("STD", torch.std(outputs, dim=1))
        print("\n")

    img_features = ImageFeatures(model_name='regnety_040', do_norm=False)
    print("\nregnety_040 TIMM NORM False")
    with torch.no_grad():
        outputs = img_features.get_features_img2vec(image)
        print("\ntimm shape adaptive avg pool", outputs.shape)
        print("OUT mean TIMM", torch.mean(outputs))
        print("OUT sum TIMM", torch.sum(outputs))
        print("OUT min TIMM", torch.min(outputs))
        print("OUT max TIMM", torch.max(outputs))
        print("STD", torch.std(outputs, dim=1))
        print("\n")

    # TO UPDATE
    img_features = ImageFeatures(model_name='convnext_tiny', do_norm=True)
    print("\nconvnext_tiny TIMM NORM True")
    with torch.no_grad():
        outputs = img_features.get_features_img2vec(image)
        print("\ntimm shape adaptive avg pool", outputs.shape)
        print("OUT mean TIMM", torch.mean(outputs))
        print("OUT sum TIMM", torch.sum(outputs))
        print("OUT min TIMM", torch.min(outputs))
        print("OUT max TIMM", torch.max(outputs))
        print("STD", torch.std(outputs, dim=1))
        print("\n")

    img_features = ImageFeatures(model_name='convnext_tiny', do_norm=False)
    print("\nconvnext_tiny TIMM NORM False")
    with torch.no_grad():
        outputs = img_features.get_features_img2vec(image)
        print("\ntimm shape adaptive avg pool", outputs.shape)
        print("OUT mean TIMM", torch.mean(outputs))
        print("OUT sum TIMM", torch.sum(outputs))
        print("OUT min TIMM", torch.min(outputs))
        print("OUT max TIMM", torch.max(outputs))
        print("STD", torch.std(outputs, dim=1))
        print("\n")

    img_features = ImageFeatures(model_name='resnest50d', do_norm=True)
    print("\nresnest50d TIMM NORM True")
    with torch.no_grad():
        outputs = img_features.get_features_img2vec(image)
        print("\ntimm shape adaptive avg pool", outputs.shape)
        print("OUT mean TIMM", torch.mean(outputs))
        print("OUT sum TIMM", torch.sum(outputs))
        print("OUT min TIMM", torch.min(outputs))
        print("OUT max TIMM", torch.max(outputs))
        print("STD", torch.std(outputs, dim=1))
        print("\n")

    img_features = ImageFeatures(model_name='resnest50d', do_norm=False)
    print("\nresnest50d TIMM NORM False")
    with torch.no_grad():
        outputs = img_features.get_features_img2vec(image)
        print("\ntimm shape adaptive avg pool", outputs.shape)
        print("OUT mean TIMM", torch.mean(outputs))
        print("OUT sum TIMM", torch.sum(outputs))
        print("OUT min TIMM", torch.min(outputs))
        print("OUT max TIMM", torch.max(outputs))
        print("STD", torch.std(outputs))
        print("\n")

    img_features = ImageFeatures(do_norm=True, is_transformer=True)
    print("\ntransformer TIMM NORM True")
    with torch.no_grad():
        outputs = img_features.get_features_img2vec(image)
        print("\ntimm shape adaptive avg pool", outputs.shape)
        print("OUT mean TIMM", torch.mean(outputs))
        print("OUT sum TIMM", torch.sum(outputs))
        print("OUT min TIMM", torch.min(outputs))
        print("OUT max TIMM", torch.max(outputs))
        print("STD", torch.std(outputs, dim=1))
        print("\n")

    img_features = ImageFeatures(do_norm=False, is_transformer=True)
    print("\ntransformer TIMM NORM False")
    with torch.no_grad():
        outputs = img_features.get_features_img2vec(image)
        print("\ntimm shape adaptive avg pool", outputs.shape)
        print("OUT mean TIMM", torch.mean(outputs))
        print("OUT sum TIMM", torch.sum(outputs))
        print("OUT min TIMM", torch.min(outputs))
        print("OUT max TIMM", torch.max(outputs))
        print("STD", torch.std(outputs, dim=1))
        print("\n")


def verifying_dimensions_timm():
    model_regnety = timm.create_model('regnety_040', pretrained=True).eval()
    model_resnest = timm.create_model('resnest50d', pretrained=True).eval()
    model_convnext = timm.create_model('convnext_tiny', pretrained=True).eval()
    swin_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)

    img1 = Image.open(r'C:\Users\lungu\OneDrive\Desktop\demo\car.jpg')
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((224, 224))
    img1 = to_tensor(resize(img1))
    regnety = model_regnety(img1.unsqueeze(0))
    print('regnety', regnety.shape)
    resnest = model_resnest(img1.unsqueeze(0))
    print('resnest', resnest.shape)
    convnext = model_convnext(img1.unsqueeze(0))
    print('convnext', convnext.shape)
    swin = swin_model(img1.unsqueeze(0))
    print('swin', swin.shape)


def test_images_questions():
    fasttext_model = utils.load_fasttext_pretrained_model()
    source_path = r"C:\Users\lungu\OneDrive\Desktop\demo\test_accuracy"
    labels = ['airplane/plane/aircraft', 'banana', 'baseball', 'beach/beaches', 'boat', 'cat', 'coffee', 'elephants',
              'football/soccer', 'juices/fruits', 'apple', 'helicopter/heli', 'lake/mountain', 'motorcycle',
              'mountains/snow', 'night/sky', 'boat/ocean', 'penguins', 'people/faces/smiling', 'running/run',
              'plane/buildings/sky/airplane/aircraft', 'volleyball', 'water/glass', 'eating/restaurant/woman/women/dinner/wine/pizza',
              'kids/balloons/children', 'cherries/cherry/fruits', 'laptop/desk/bureau/computer', 'ping-pong/ball/table/game/tennis', 'laptop',
              'class/classroom/room/professor/person/man', 'ski/skiing/snow/mountain', 'hawk/bird/eagle', 'bear/bears',
              'parachute/mountain', 'tiger/animal/cat', 'lake/clouds', 'ball/basket/basketball', 'road/street', 'sunset/person/mountain/riding',
              'skateboard/board/skate', 'crayons/colors', 'horse/beach/sky/sea/ocean']
    checkpoint_ssl_resnet50 = r"D:\facultate\An 3\LICENTA\VQAProjectV2\logs\comparing_models\VQA_exp_SSL_ResNet50\weights\VQA_exp_SSL_ResNet50\version_1\checkpoints\epoch=49-step=12300.ckpt"
    checkpoint_regnety = r"D:\facultate\An 3\LICENTA\VQAProjectV2\logs\comparing_models\VQA_exp_RegNetY\weights\VQA_exp_RegNetY\version_2\checkpoints\epoch=69-step=17220.ckpt"
    checkpoint_resnest50d = r"D:\facultate\An 3\LICENTA\VQAProjectV2\logs\comparing_models\VQA_exp_ResNest50d\weights\VQA_exp_ResNest50d\version_0\checkpoints\epoch=79-step=19680.ckpt"
    checkpoint_swin = r"D:\facultate\An 3\LICENTA\VQAProjectV2\logs\comparing_models\VQA_exp_Swin_tiny\weights\VQA_exp_Swin_tiny\version_0\checkpoints\epoch=69-step=8610.ckpt"
    checkpoint_convnext = r"D:\facultate\An 3\LICENTA\VQAProjectV2\logs\comparing_models\VQA_exp_ConvNext_Tiny\weights\VQA_exp_ConvNext_Tiny\version_0\checkpoints\epoch=79-step=19680.ckpt"
    checkpoint_resnet50_last_version =  r"D:\facultate\An 3\LICENTA\VqaWeb\checkpoint\version8\weights\version_8\checkpoints\epoch=99-step=24600.ckpt"

    vqa_ssl = VqaSSL.load_from_checkpoint(checkpoint_ssl_resnet50)
    vqa_regnety = VqaRegnety.load_from_checkpoint(checkpoint_regnety)
    vqa_resnest = VqaResnest.load_from_checkpoint(checkpoint_resnest50d)
    vqa_swin = VqaSwin.load_from_checkpoint(checkpoint_swin)
    vqa_convnext = VqaConvNext.load_from_checkpoint(checkpoint_convnext)
    vqa_resnet50 = VqaResNet50LastVersion.load_from_checkpoint(checkpoint_resnet50_last_version)

    img2feat_ssl = ImageFeatures('ssl_resnet50', do_norm=False)
    img2feat_regnety = ImageFeatures('regnety_040', do_norm=False)
    img2feat_resnest = ImageFeatures('resnest50d', do_norm=False)
    img2feat_swin = ImageFeatures(is_transformer=True, do_norm=False)
    img2feat_convnext = ImageFeatures('convnext_tiny', do_norm=False)
    image_to_vec_last_version = img2vec(model='resnet50', layer_output_size=2048)

    top5_answers_swin = []
    top5_answers_ssl = []
    top5_answers_resnest = []
    top5_answers_regenty = []
    top5_answers_convnext = []
    top5_answers_resnet50_last_version = []

    sum1_swin = 0
    sum1_ssl = 0
    sum1_resnest = 0
    sum1_regnety = 0
    sum1_convnext = 0
    sum1_resnet50_last_version = 0

    sum5_resnest = 0
    sum5_ssl = 0
    sum5_regnety = 0
    sum5_swin = 0
    sum5_convnext = 0
    sum5_resnet50_last_version = 0

    question = "What is this?"
    for index in range(1, len(labels)):
        image = source_path + "\\" + str(index) + ".jpg"
        top5_answers_swin.append(inference(vqa_swin, fasttext_model, img2feat_swin, image, question, swin_tiny=True))
        top5_answers_regenty.append(inference(vqa_regnety, fasttext_model, img2feat_regnety, image, question))
        top5_answers_ssl.append(inference(vqa_ssl, fasttext_model, img2feat_ssl, image, question))
        top5_answers_resnest.append(inference(vqa_resnest, fasttext_model, img2feat_resnest, image, question))
        top5_answers_convnext.append(inference(vqa_convnext, fasttext_model, img2feat_convnext, image, question))
        top5_answers_resnet50_last_version.append(inference_resnet50_last_version(image, question, fasttext_model, vqa_resnet50, image_to_vec_last_version))

    for index in range(1, len(labels)):
        # RESNEST
        top1 = top5_answers_resnest[index-1][0][0]
        top5 = []
        for element in top5_answers_resnest[index-1]:
            top5.append(element[0])

        if top1 in labels[index-1]:
            sum1_resnest = sum1_resnest + 1
        else:
            print("RESNEST wrong top1[[", top1, ']]image index=', index)

        found_in_top5_resnest = False
        for word in top5:
            if word in labels[index-1]:
                found_in_top5_resnest = True

        if found_in_top5_resnest:
            sum5_resnest = sum5_resnest + 1

        # SSL RESNET50
        top1 = top5_answers_ssl[index-1][0][0]
        top5 = []
        for element in top5_answers_ssl[index-1]:
            top5.append(element[0])

        if top1 in labels[index-1]:
            sum1_ssl = sum1_ssl + 1
        else:
            print("SSL wrong top1[[", top1, ']]image index=', index)

        found_in_top5_ssl = False
        for word in top5:
            if word in labels[index - 1]:
                found_in_top5_ssl = True

        if found_in_top5_ssl:
            sum5_ssl = sum5_ssl + 1

        # REGNETY
        top1 = top5_answers_regenty[index-1][0][0]
        top5 = []
        for element in top5_answers_regenty[index-1]:
            top5.append(element[0])

        if top1 in labels[index-1]:
            sum1_regnety = sum1_regnety + 1
        else:
            print("REGNETY wrong top1[[", top1, ']]image index=', index)

        found_in_top5_regnety = False
        for word in top5:
            if word in labels[index - 1]:
                found_in_top5_regnety = True

        if found_in_top5_regnety:
            sum5_regnety = sum5_regnety + 1

        # SWIN TINY
        top1 = top5_answers_swin[index-1][0][0]
        top5 = []
        for element in top5_answers_swin[index-1]:
            top5.append(element[0])

        if top1 in labels[index-1]:
            sum1_swin = sum1_swin + 1
        else:
            print("Swin Tiny wrong top1[[", top1, ']]image index=', index)

        found_in_top5_swin = False
        for word in top5:
            if word in labels[index - 1]:
                found_in_top5_swin = True

        if found_in_top5_swin:
            sum5_swin = sum5_swin + 1

        # CONVNEXT
        top1 = top5_answers_convnext[index - 1][0][0]
        top5 = []
        for element in top5_answers_convnext[index - 1]:
            top5.append(element[0])

        if top1 in labels[index - 1]:
            sum1_convnext = sum1_convnext + 1
        else:
            print("CONVNEXT wrong top1[[", top1, ']]image index=', index)

        found_in_top5_convnext = False
        for word in top5:
            if word in labels[index - 1]:
                found_in_top5_convnext = True

        if found_in_top5_convnext:
            sum5_convnext = sum5_convnext + 1

        # RESNET50 LAST VERSION
        top1 = top5_answers_resnet50_last_version[index - 1][0][0]
        top5 = []
        for element in top5_answers_resnet50_last_version[index - 1]:
            top5.append(element[0])

        if top1 in labels[index - 1]:
            sum1_resnet50_last_version = sum1_resnet50_last_version + 1
        else:
            print("RESNET50 LAST VERSION wrong top1 [[", top1, ']] image index=', index)

        found_in_top5_resnet50 = False
        for word in top5:
            if word in labels[index - 1]:
                found_in_top5_resnet50 = True

        if found_in_top5_resnet50:
            sum5_resnet50_last_version = sum5_resnet50_last_version + 1

        print("\n")

    print("\n\n")
    print("TOP 1 ACCURACY SWIN", sum1_swin / len(labels) * 100, "%")
    print("TOP 1 ACCURACY REGNETY", sum1_regnety / len(labels) * 100, "%")
    print("TOP 1 ACCURACY RESNEST", sum1_resnest / len(labels) * 100, "%")
    print("TOP 1 ACCURACY SSL", sum1_ssl / len(labels) * 100, "%")
    print("TOP 1 ACCURACY CONVNEXT", sum1_convnext / len(labels) * 100, "%")
    print("TOP 1 ACCURACY RESNET50 LAST VERSION", sum1_resnet50_last_version / len(labels) * 100, "%")
    print("\n\n")
    print("TOP 5 ACCURACY SWIN", sum5_swin / len(labels) * 100, "%")
    print("TOP 5 ACCURACY REGNETY", sum5_regnety / len(labels) * 100, "%")
    print("TOP 5 ACCURACY RESNEST", sum5_resnest / len(labels) * 100, "%")
    print("TOP 5 ACCURACY SSL", sum5_ssl / len(labels) * 100, "%")
    print("TOP 5 ACCURACY CONVNEXT", sum5_convnext / len(labels) * 100, "%")
    print("TOP 5 ACCURACY RESNET50 LAST VERSION", sum5_resnet50_last_version / len(labels) * 100, "%")


def inference(vqa_model, fasttext_model, img2features, image_path, question, swin_tiny=False):
    pil_image = Image.open(image_path)
    pil_image = pil_image.convert('RGB')

    if not swin_tiny:
        features = img2features.get_features_img2vec(pil_image).squeeze(2).squeeze(2)
    else:
        features = img2features.get_features_img2vec(pil_image)  # Swin Tiny

    embedding = utils.get_text_embedding_fasttext(question, fasttext_model)

    embedding = utils.get_padding_question(embedding).unsqueeze(0)
    # vqa_model = vqa_model.double()
    answer = vqa_model(features, embedding).squeeze().detach().cpu().numpy()
    top5_answers = fasttext_model.similar_by_vector(answer, topn=5)

    return top5_answers


def inference_resnet50_last_version(path_to_image, question, fasttext_model, vqa_model, image_to_vec):
    PIL_image = Image.open(path_to_image).convert("RGB")

    embedding_image_tensor = image_to_vec.get_vec(PIL_image, tensor=True)

    question_embedding = utils.get_text_embedding_matrix(question, fasttext_model)
    q = utils.get_padding_question(question_embedding)

    image_embedding = torch.permute(embedding_image_tensor, (0, 2, 3, 1))
    q_emb = q.unsqueeze(0)
    img_emb = image_embedding.squeeze().unsqueeze(0)
    answer = vqa_model(img_emb, q_emb)
    ans = answer.detach().cpu().numpy()
    word_answer = fasttext_model.similar_by_vector(ans[0], topn=10)
    top5 = []
    index = 0
    for elem in word_answer:
        a = elem[0]
        if a.islower() or a.isnumeric():
            top5.append(elem)
            index = index + 1
        if index == 5:
            break
    return top5


if __name__ == '__main__':
    test_images_questions()
