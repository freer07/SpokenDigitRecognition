from flask import Flask, request, send_file
from wav2spec import wav2spec
from AlexNetSpec import AlexNetSpec as ANS 
import tensorflow as tf
from tensorflow import keras
import numpy as np

app = Flask(__name__)

def pipline(name):
    wav2spec("", name)
    arr = name.split('.')
    filename = arr[0]
    model = keras.models.load_model("model")

    imgName = filename + ".png"
    img = tf.keras.utils.load_img(
       imgName, target_size=(ANS.HEIGHT, ANS.WIDTH)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    respone = send_file(imgName)
    respone.headers["message"] = "This image most likely belongs to {} with a {:.2f} percent confidence.".format([int(x) for x in range(10)][np.argmax(score)], 100 * np.max(score))
    respone.headers["Access-Control-Allow-Origin"] = "*"
    return respone

@app.route("/", methods=['POST'])
def upload():
    f = request.files['wav_file']
    filename = "uploaded_file.wav"
    #assert(f.content_type == "audio/wave")
    f.save(filename)
    
    return pipline(filename)