import os
import io
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template, redirect, url_for
from flask_cors import CORS
from model import FAC_CNN
from MY_CNN_MODEL import MY_CNN
from model2 import FAC_CNN2
import torchvision.models as models
import time

import torch.nn as nn

app = Flask(__name__)
CORS(app)  # 解决跨域问题



weights_path = "./fac-cnn19.pth"
class_json_path = "./class_indices19.json"
# assert os.path.exists(weights_path), "weights path does not exist..."
# assert os.path.exists(class_json_path), "class json path does not exist..."

# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# create model
# model = MobileNetV2(num_classes=5)

vgg = models.vgg16(pretrained=False)
model = FAC_CNN(vgg)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

# load class info
json_file = open(class_json_path, 'rb')
class_indict = json.load(json_file)


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        raise ValueError("input file does not RGB image...")
    return my_transforms(image).unsqueeze(0).to(device)


def get_prediction(image_bytes):
    try:
        start = int(round(time.time() * 1000))

        tensor = transform_image(image_bytes=image_bytes)
        outputs = torch.softmax(model.forward(tensor).squeeze(), dim=0)
        prediction = outputs.detach().cpu().numpy()
        template = "class:{:<15} probability:{:.3f}"
        index_pre = [(class_indict[str(index)], float(p)) for index, p in enumerate(prediction)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        index_pre = index_pre[0: 7]
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
        end = int(round(time.time() * 1000))
        print(end - start)
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info


vgg2 = models.vgg16(pretrained=False)
model2 = FAC_CNN2(vgg2)
# load model weights
# model2.load_state_dict(torch.load("./fac-cnn5.pth", map_location=device))
model2.load_state_dict(torch.load("./fac-cnn2_44.pth", map_location=device))
model2.to(device)
model2.eval()

json_file2 = open("./class_indices5.json", 'rb')
class_indict2 = json.load(json_file2)


# vgg = models.vgg16(pretrained=False)
# model2 = FAC_CNN(vgg)
# model2.load_state_dict(torch.load("./fac-cnn44.pth", map_location=device))
# model2.to(device)
# model2.eval()
# json_file2 = open("./class_indices4.json", 'rb')
# class_indict2 = json.load(json_file2)


def get_prediction2(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes)
        outputs = torch.softmax(model2.forward(tensor).squeeze(), dim=0)
        prediction = outputs.detach().cpu().numpy()
        template = "class:{:<15} probability:{:.3f}"
        index_pre = [(class_indict2[str(index)], float(p)) for index, p in enumerate(prediction)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info

# 十九模型--------------------------------
def get_prediction19(image_bytes):
    try:
        start = int(round(time.time() * 1000))

        tensor = transform_image(image_bytes=image_bytes)
        outputs = torch.softmax(model.forward(tensor).squeeze(), dim=0)
        prediction = outputs.detach().cpu().numpy()
        template = "class:{:<15} probability:{:.3f}"
        index_pre = [(class_indict[str(index)], float(p)) for index, p in enumerate(prediction)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        index_pre = index_pre[0: 7]
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
        end = int(round(time.time() * 1000))
        print(end - start)
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info




# 21模型--------------------------------


weights_path = "./fac-cnn21.pth"
class_json_path = "./class_indices21.json"

vgg = models.vgg16(pretrained=False)
model21 = FAC_CNN(vgg)
model21.load_state_dict(torch.load(weights_path, map_location=device))
model21.to(device)
model21.eval()

# load class info
json_file21 = open(class_json_path, 'rb')
class_indict21 = json.load(json_file21)

def get_prediction21(image_bytes):
    try:
        start = int(round(time.time() * 1000))

        tensor = transform_image(image_bytes=image_bytes)
        outputs = torch.softmax(model21.forward(tensor).squeeze(), dim=0)
        prediction = outputs.detach().cpu().numpy()
        template = "class:{:<15} probability:{:.3f}"
        index_pre = [(class_indict21[str(index)], float(p)) for index, p in enumerate(prediction)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        index_pre = index_pre[0: 7]
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
        end = int(round(time.time() * 1000))
        print(end - start)
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info



# 30模型--------------------------------

weights_path = "./aid30_1.pth"
class_json_path = "./class_indices30.json"

vgg = models.vgg16(pretrained=False)
model30 = FAC_CNN(vgg)
model30.load_state_dict(torch.load(weights_path, map_location=device))
model30.to(device)
model30.eval()

# load class info
json_file30 = open(class_json_path, 'rb')
class_indict30 = json.load(json_file30)



def get_prediction30(image_bytes):
    try:
        start = int(round(time.time() * 1000))

        tensor = transform_image(image_bytes=image_bytes)
        outputs = torch.softmax(model30.forward(tensor).squeeze(), dim=0)
        prediction = outputs.detach().cpu().numpy()
        template = "class:{:<15} probability:{:.3f}"
        index_pre = [(class_indict30[str(index)], float(p)) for index, p in enumerate(prediction)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        index_pre = index_pre[0: 7]
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
        end = int(round(time.time() * 1000))
        print(end - start)
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info






















@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction(image_bytes=img_bytes)
    return jsonify(info)


@app.route("/predict2", methods=["POST"])
@torch.no_grad()
def predict2():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction2(image_bytes=img_bytes)
    return jsonify(info)


@app.route("/predict19", methods=["POST"])
@torch.no_grad()
def predict19():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction19(image_bytes=img_bytes)
    return jsonify(info)



@app.route("/predict21", methods=["POST"])
@torch.no_grad()
def predict21():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction21(image_bytes=img_bytes)
    return jsonify(info)



@app.route("/predict30", methods=["POST"])
@torch.no_grad()
def predict30():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction30(image_bytes=img_bytes)
    return jsonify(info)



@app.route("/", methods=["GET", "POST"])
def indexRedirect():
    return redirect(url_for('index'),code=302)

@app.route("/data", methods=["GET", "POST"])
def data():
    return render_template("data.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    return render_template("contact.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    return render_template("login.html")

@app.route("/forecast", methods=["GET", "POST"])
def forecast():
    return render_template("up1.html")


@app.route("/recognize", methods=["GET", "POST"])
def recognize():
    return render_template("recognize.html")

@app.route("/index", methods=["GET", "POST"])
def index():
    return render_template("index.html")

#
# @app.route("/assets/css/main.css", methods=["GET", "POST"])
# def css():
#     return render_template("/assets/css/main.css")



if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=5000)
    app.run()




