import time
from flask import Flask, render_template, request, url_for
from service import utils
from Inference import do_inference
import requests
# from models.VQAmodel import VQANetwork as VqaNet

app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "D:/facultate/An 3/LICENTA/VqaWeb/static/images"
pair = {"image": "", "question": ""}

fasttext_model = utils.load_fasttext_from_pc()
visual_model_name = 'resnet50'
img2feat, vqa_model = utils.get_image2feat('resnet50')


@app.route('/', methods=['GET', 'POST'])
def home():
    ext = ""
    f_name = ""
    image_url = ""
    question = ""
    submit_clicked = False
    if request.method == "POST":
        if request.form["action"] == "Upload":
            if request.files["image"].filename != "":
                image = request.files["image"]
                if image.filename[-4:] == ".jpg":
                    ext = ".jpg"
                    # image.save(app.config["IMAGE_UPLOADS"] + "/" + "uploaded_image.jpg")
                    image.save(app.config["IMAGE_UPLOADS"] + "/" + image.filename)
                elif image.filename[-4:] == ".png":
                    ext = ".png"
                    image.save(app.config["IMAGE_UPLOADS"] + "/" + image.filename)
                else:
                    ext = ".jpeg"
                    image.save(app.config["IMAGE_UPLOADS"] + "/" + image.filename)
                f_name = image.filename
                pair["image"] = f_name

            elif request.form["url"] != "":
                image_url = request.form["url"]

                print(image_url)
                if image_url[-4:] == ".jpg":
                    ext = ".jpg"
                    img_response = requests.get(image_url)

                    file = open("static/images/url_image.jpg", "wb")

                    file.write(img_response.content)
                    file.close()
                    pair["image"] = "url_image.jpg"

                elif image_url[-4:] == ".png":
                    ext = ".png"

                    img_response = requests.get(image_url)

                    file = open("static/images/url_image.png", "wb")

                    file.write(img_response.content)
                    file.close()
                    pair["image"] = "url_image.png"

                elif image_url[-5:] == ".jpeg":
                    ext = ".jpeg"
                    img_response = requests.get(image_url)

                    file = open("static/images/url_image.jpeg", "wb")

                    file.write(img_response.content)
                    file.close()
                    pair["image"] = "url_image.jpeg"

                else:
                    ext = ".webp"
                    img_response = requests.get(image_url)

                    file = open("static/images/url_image.webp", "wb")

                    file.write(img_response.content)
                    file.close()
                    pair["image"] = "url_image.webp"

        if request.form["action"] == "Submit":
            question = request.form["question"]
            pair["question"] = question
            submit_clicked = True

    if ext != "":
        time.sleep(0.5)
        path = 'images/' + pair["image"]

        return render_template("index.html", upload=path, url=image_url, question=question, answer1="TOP answer 1",
                               answer2="TOP answer 2", answer3="TOP answer 3",
                               answer4="TOP answer 4", answer5="TOP answer 5")
    time.sleep(0.5)
    path_no_image = 'images/no_image.png'

    if submit_clicked:
        img = pair["image"]
        question = pair["question"]
        print(pair)
        if img != "" and question != "":
            absolute_path_image = app.config["IMAGE_UPLOADS"] + "/" + img
            # answer = inference(image_to_vec, absolute_path_image, question, fasttext_model, vqa_model)
            answer = do_inference(visual_model_name, vqa_model, fasttext_model, img2feat, absolute_path_image, question)
            return render_template("index.html", upload='images/' + img, url=image_url, question=question, answer1=answer[0],
                                   answer2=answer[1], answer3=answer[2], answer4=answer[3], answer5=answer[4])
        else:
            return render_template("index.html", upload=path_no_image, url=image_url, question=question, answer1="TOP answer 1",
                                   answer2="TOP answer 2", answer3="TOP answer 3",
                                   answer4="TOP answer 4", answer5="TOP answer 5")

    else:
        return render_template("index.html", upload=path_no_image, url=image_url, question=question, answer1="TOP answer 1",
                               answer2="TOP answer 2", answer3="TOP answer 3",
                               answer4="TOP answer 4", answer5="TOP answer 5")


if __name__ == '__main__':
    app.run()
