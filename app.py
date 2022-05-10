import base64
import io
from flask import Flask, flash, render_template, redirect, request, send_from_directory, url_for, json
import os
import cv2
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_byte_array(image: Image):
    imageByteArr = io.BytesIO()
    image.save(imageByteArr, format('JPEG'))
    imageByteArr = imageByteArr.getvalue()
    return imageByteArr


def makePredictions(path):
    img = cv2.imread(path)

    with open('zam/obj.names', 'r') as f:
        classes = f.read().splitlines()

    net = cv2.dnn.readNetFromDarknet('zam/yolov4-obj.cfg', 'zam/yolov4-obj_last.weights')

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)

    blunt_count = 0
    hooked_count = 0
    bubbled_count = 0
    brokenwithdanglingtip_count = 0

    for (classId, score, box) in zip(classIds, scores, boxes):
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                      color=(0, 255, 0), thickness=2)
        classes_1 = classes[classId]
        if classes_1 == "blunt":
            blunt_count += 1
        elif classes_1 == "hooked":
            hooked_count += 1
        elif classes_1 == "bubbled":
            bubbled_count += 1
        elif classes_1 == "brokenwithdanglingtip":
            brokenwithdanglingtip_count += 1
        text = classes_1 + str(score)
        # text = '%s: %.2f' % (classes[classId[0]], score)
        cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 5,
                    color=(153, 255, 51), thickness=10)
    final_tags = "blunt = " + str(blunt_count) + " - hooked = " + str(hooked_count) + " - bubbled = " + str(
        bubbled_count) + " - brokenwithdanglingtip = " + str(brokenwithdanglingtip_count)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    np_img = Image.fromarray(img)
    img_encoded = image_to_byte_array(np_img)
    base64_bytes = base64.b64encode(img_encoded).decode()
    return base64_bytes, final_tags


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'img' not in request.files:
            return render_template('home.html', filename="unnamed.png", message="Please upload an file")
        f = request.files['img']
        filename = secure_filename(f.filename)
        if f.filename == '':
            return render_template('home.html', filename="unnamed.png", message="No file selected")
        if not ('jpeg' in f.filename or 'png' in f.filename or 'jpg' in f.filename):
            return render_template('home.html', filename="unnamed.png",
                                   message="please upload an image with .png or .jpg/.jpeg extension")
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        if len(files) == 1:
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            files.remove("unnamed.png")
            file_ = files[0]
            os.remove(app.config['UPLOAD_FOLDER'] + '/' + file_)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        predictions, final_tags = makePredictions(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return render_template('predictions.html', user_image=predictions, message=final_tags, show=True)
    return render_template('home.html', filename='unnamed.png')


@app.route('/stats', methods=['GET', 'POST'])
def stats():
    images_output = {}
    if request.method == 'POST':
        for file in request.files.getlist('dir'):
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            else:
                continue
            predictions, final_tags = makePredictions(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            images_output[filename] = final_tags
        return render_template('stats_prediction.html', user_image=predictions, message=images_output, show=True)
    return render_template('stats.html', filename='unnamed.png')


@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    if request.method == 'POST':
        if 'img' not in request.files:
            return render_template('retrain.html', filename="unnamed.png", message="{lease select a valid directory}")
        f = request.files['img']
        filename = secure_filename(f.filename)
        if f.filename == '':
            return render_template('retrain.html', filename="unnamed.png", message="No file selected")
        if not ('jpeg' in f.filename or 'png' in f.filename or 'jpg' in f.filename):
            return render_template('retrain.html', filename="unnamed.png",
                                   message="please upload an image with .png or .jpg/.jpeg extension")
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        if len(files) == 1:
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            files.remove("unnamed.png")
            file_ = files[0]
            os.remove(app.config['UPLOAD_FOLDER'] + '/' + file_)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        predictions = makePredictions(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('retrain.html', filename=f.filename, message=predictions, show=True)
    return render_template('retrain.html', filename='unnamed.png')


# # Route for handling the login page logic
# @app.route('/', methods=['GET', 'POST'])
# def login():
#     error = None
#     if request.method == 'POST':
#         if request.form['username'] != 'admin' or request.form['password'] != 'admin':
#             error = 'Invalid Credentials. Please try again.'
#         else:
#             return redirect(url_for('home'))
#     return render_template('login.html', error=error)

# @app.route("/")
# def first():
#     return render_template('login.html')
#
#
# @app.route("/signin", methods=['POST'])
# def signin():
#     username = request.form['username']
#     password = request.form['password']
#     if username and password:
#         return render_template('home.html')
#     return render_template('login.html')
#
#
# def validateUser(username, password):
#     return True


if __name__ == "__main__":
    app.run()
