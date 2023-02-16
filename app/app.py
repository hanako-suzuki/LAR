# Flaskとrender_template（HTMLを表示させるための関数）をインポート
from flask import Flask, render_template, request, Response
import cv2
import re
import math

import numpy as np

# 色シンボルの座標
x = [0.65,  0.3,  0.15,  0.275,  0.4,   0.25,  0.4,  0.5]
y = [0.3,   0.6,  0.05,  0.4,    0.45,  0.2,   0.2,  0.35]
# 色シンボルの名前
color_name = ["red", "green", "blue", "lightblue", "yellow", "navy", "purple", "orange"]
# 色シンボルに割り当てたbit列
ave = ["100","011","010","110","111","000","001","101"]
# 読み取り位置座標
x_value = 10
y_value = 10

R_pilot = [160, 55, 70]
G_pilot = [70, 210, 80]
B_pilot = [45, 35, 230]

R_value = 0
G_value = 0
B_value = 0

rgb_message = "tmp"
rgb_color = "color"

# Flaskオブジェクトの生成
app = Flask(__name__)

camera = cv2.VideoCapture(0)

# OpenCVでcamera読み込み
def get_frame():
    flag = 0
    while True:
        success, image = camera.read()
        if not success:
            print("can't load camera")
            break
        
        read_color(image)

        # 差分を取得
        if flag != 0:
            img_diff = cv2.absdiff(image, pre_img)
            ret, jpeg = cv2.imencode('.jpg', img_diff)
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        else:
            flag = 1
        # 画像を保持
        pre_img = image

        # Add OpenCV processing here
        imageOutput = cv2.rectangle(image, (x_value-1, y_value-1), (x_value+1, y_value+1), (255, 0, 0), 1, 4)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # ret, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Encode processed image as JPEG
        # ret, jpeg = cv2.imencode('.jpg', threshold)
        # ret, jpeg = cv2.imencode('.jpg', imageOutput)


        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


def read_color(image):
    global rgb_message
    global R_value
    global G_value
    global B_value
    # 指定した領域の輝度値を取得
    # BGRの順番なので注意
    R_value = 0
    G_value = 0
    B_value = 0
    for i in range(3):
        B_value += image[y_value+i-1, x_value+i-1, 0]
        G_value += image[y_value+i-1, x_value+i-1, 1]
        R_value += image[y_value+i-1, x_value+i-1, 2]
    B_value /= 3
    G_value /= 3
    R_value /= 3
    rgb_message = str(R_value) + ", " + str(G_value) + ", " + str(B_value)
    demodulation()

# チャネル補正
def demodulation():
    H = [[R_pilot[0], G_pilot[0], B_pilot[0]], [R_pilot[1], G_pilot[1], B_pilot[1]], [R_pilot[2], G_pilot[2], B_pilot[2]]]
    H_inv = np.linalg.inv(H)

    re = [[R_value], [G_value], [B_value]]
    ans = np.dot(H_inv, re)
    for i in range(len(ans)):
        if ans[i][0] < 0:
            ans[i][0] = 0
    ans = ans/np.sum(ans)
    x_pos = 0.65*ans[0][0] + 0.3 * ans[1][0] + 0.15 * ans[2][0]
    y_pos = 0.3*ans[0][0] + 0.6 * ans[1][0] + 0.05 * ans[2][0]

    comp = [0 for i in range(len(x))]
    for i in range(len(x)):
        comp[i] = math.sqrt(pow(x_pos-x[i], 2) + pow(y_pos-y[i], 2))
    tmp = comp.index(min(comp))

    global rgb_color
    rgb_color = color_name[tmp]




#「/」へアクセスがあった場合に、「index.html」を返す
@app.route("/")
def index():
    return render_template('index.html')

#「/video_feed」へアクセスがあった場合に、「index.html」を返す
@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ボタンが押下された場合に，輝度値を更新する
@app.route('/update_variable')
def update_variable():
    # Update the value of the variable here
    variable_value = rgb_message + rgb_color
    return variable_value

@app.route('/process', methods=['POST'])
def process():
    file = request.files['image']
    image = np.array(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)

    # Add OpenCV processing here
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Convert processed image back to binary format
    _, image_encoded = cv2.imencode('.jpg', threshold)

    return image_encoded.tobytes()


#おまじない
if __name__ == "__main__":
    app.run(debug=True)