import cv2 as cv2
import numpy as np
import tensorflow as tf
import math
import os
import json
from flask import Flask,request,Response
from flask_cors import CORS
import uuid
#import pyrebase

app = Flask(__name__)
CORS(app)
port = int(os.environ.get("PORT", 5000))

def detect(image):
    interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_thunder_3.tflite')
    interpreter.allocate_tensors()
    #membacagambar
    imagedt=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    width = image.shape[1]
    height = image.shape[0]
    #resizeimage
    img = image.copy()
    img = tf.image.resize_with_pad(np.expand_dims(image, axis=0), 256,256)

    #converttofloat
    input_image=tf.cast(img, dtype=tf.float32)

    #detailinputoutput
    input_details=interpreter.get_input_details()
    output_details=interpreter.get_output_details()

    #penggabungan gambar dengan image data
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    #keypoint 
    keypointsdetection = interpreter.get_tensor(output_details[0]['index'])
    shaped = np.squeeze(np.multiply(interpreter.get_tensor(interpreter.get_output_details()[0]['index']), [height,width,1])).astype(int)
    shaped[0], shaped[1]

    EDGES = {
        (0, 1): 'm',
        (0, 2): 'c',
        (1, 3): 'm',
        (2, 4): 'c',
        (0, 5): 'm',
        (0, 6): 'c',
        (5, 7): 'm',
        (7, 9): 'm',
        (6, 8): 'c',
        (8, 10): 'c',
        (5, 6): 'y',
        (5, 11): 'm',
        (6, 12): 'c',
        (11, 12): 'y',
        (11, 13): 'm',
        (13, 15): 'm',
        (12, 14): 'c',
        (14, 16): 'c'
    }

    #drawkeypoint 
    for kp in shaped:
        ky, kx, kp_conf = kp
        image = cv2.circle(imagedt, (int(kx), int(ky)), 20, (0,255,0), -1)
        continue
    #drawline
    for edge, color in EDGES.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        image = cv2.line(imagedt, (int(x1), int(y1)), (int(x2), int(y2)),(0,255,0), 10)
        continue

    #hitungskala
    #menghitung skala gambar tinggi
    skalaheight=(((height*200)/265)/200)


    #menghitung skala gambar lebar
    skalawidht=(((width*200)/140)/200)

    #perhitungan Euclidean Distance
    #hitung bahu
    bahu1=math.pow(shaped[5][1]-shaped[6][1],2)
    bahu2=math.pow(shaped[5][0]-shaped[6][0],2)
    Panjang_Bahu=math.sqrt(bahu1 + bahu2)
    ld= int(Panjang_Bahu/skalawidht)
    

    #hitung tangan kanan
    tangan1=math.pow(shaped[10][1]-shaped[6][1],2)
    tangan2=math.pow(shaped[10][0]-shaped[6][0],2)
    Panjang_Tangan=math.sqrt(tangan1 + tangan2)
    pt= int(Panjang_Tangan/skalaheight)
    

    #huitung panjang badan kanan
    badan11=math.pow(shaped[16][1]-shaped[6][1],2)
    badan22=math.pow(shaped[16][0]-shaped[6][0],2)
    Panjang_badan1=math.sqrt(badan11 + badan22)
    tb= int (Panjang_badan1/skalaheight)

    #Harusnya di buat Array
    #print ("Panjang Bahu", (int(ld)) ,"cm")
    #print ("Panjang Tangan", (int(pt)) ,"cm")
    #print ("Panjang Badan", (int(tb)) ,"cm")
    #print ("Lingkar Badan", (int(lb)) ,"cm")

    i=0
    while i==0:
        
        if ld <= 36 and tb<=133 and pt<=56:
            ukuran = "S"
            #print("UKURAN GAMIS S")
            break
        if ld <= 38 and tb<=136 and pt<=58:
            ukuran = "M"
            #print("UKURAN GAMIS M")
            break
        if ld <= 40 and tb<=140 and pt<=58:
            ukuran = "L"
           #print("UKURAN GAMIS L")
            break
        if ld <= 42 and tb<=142 and pt<=59:
            ukuran = "XL"
            #print("UKURAN GAMIS XL")
            break

    
    Hasil_Pengukuran=[ ld,pt,
                    tb,ukuran]

    print (Hasil_Pengukuran)
        
    #savefile
    path_file=('static/%s.jpg' %uuid.uuid4().hex)
    cv2.imwrite(path_file,image)

    data = {
        "bahu": ld,
        "tangan": pt,
        "badan": tb,
        "ukuran": ukuran,
        "gambar": path_file
    }

    value = {
        "success" : True,
        "data" : data
    }
    return json.dumps(value)
    
        
@app.route('/api/upload',methods=['POST'])
def upload():
    image = cv2.imdecode(np.fromstring(request.files['imagedata'].read(),np.uint8), cv2.IMREAD_UNCHANGED)
    img_processed = detect(image)
    return Response(response=img_processed, status=200,mimetype="application/json")

@app.route("/")
def index():
    return "<h1>Welcome to our server !!</h1>"

if __name__ == '__main__':
    app.run(threaded=True, port=port)
