from flask import Flask, request
from wav2spec import wav2spec
from AlexNetSpec import AlexNetSpec as ANS 
import tensorflow as tf
from tensorflow import keras
import numpy as np

app = Flask(__name__)

def pipline(name, directory):
    wav2spec(directory, name)
    arr = name.split('.')
    arr = arr[0].split('\\')
    filename = arr[len(arr)-1]
    model = keras.models.load_model("model")

    img = tf.keras.utils.load_img(
       directory + "\\" + filename + ".png", target_size=(ANS.HEIGHT, ANS.WIDTH)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return "This image most likely belongs to {} with a {:.2f} percent confidence.".format([int(x) for x in range(10)][np.argmax(score)], 100 * np.max(score))

@app.route("/", methods=['POST'])
def upload():
    f = request.files['wav_file']
    directory = "testdata"
    filename = directory+"\\uploaded_file.wav"
    assert(f.content_type == "audio/wave")
    f.save(filename)

    return pipline(filename, directory)