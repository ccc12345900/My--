import base64
import io
import os
from io import BytesIO

import numpy as np
import requests as req
from PIL import Image
from flask import Flask, request, render_template, jsonify
from matplotlib import pyplot as plt

import detect as service
import cv2
from flask_cors import CORS
# flask web service
app = Flask(__name__, template_folder="web")
CORS(app, resources=r'/*')	# 注册CORS, "/*" 允许访问所有api

@app.route('/detect')
def detect():
    return render_template('detect.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/detect/imageDetect', methods=['post'])
def upload():
    re_conf = request.form.get('re_conf')
    p_path = os.path.abspath(os.path.abspath(__file__))
    re_conf= float(re_conf)
    father_path = os.path.abspath(os.path.dirname(p_path) + os.path.sep + ".")
    save_path = father_path+"\\mydata\\images\\test"
    # step 1. receive image
    file = request.form.get('imageBase64Code')
    image_link = request.form.get("imageLink")

    if image_link:
        response = req.get(image_link)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(BytesIO(base64.b64decode(file)))
        dirs =father_path+'/mydata/images/test/1.png'
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_BGR2RGB)
        cv2.imwrite(dirs, np.asarray(img))

    # step 2. detect image
    image_array,re_label,num = service.app_detect(re_conf)

    # step 3. convert image_array to byte_array
    # img = Image.fromarray(image_array, 'RGB')
    img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img, 'RGB')
    img_byte_array = io.BytesIO()
    img.save(img_byte_array, format='JPEG')

    # step 4. return image_info to page
    image_info = base64.b64encode(img_byte_array.getvalue()).decode('ascii')
    return jsonify({"image_info": image_info,"re_label":re_label,"num":num})
    # return "image"

if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False, port=8081)


# image = Image.open(BytesIO(base64.b64decode(file)))
#         dirs =father_path+'/mydata/images/test/1.png'
#         cv2.imwrite(dirs, np.asarray(image))