from flask import Flask, request, send_file
from wav2spec import wav2spec
from AlexNetSpec import AlexNetSpec as ANS 
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from werkzeug.middleware.proxy_fix import ProxyFix


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

    response = send_file(imgName)
    response.headers["message"] = "This image most likely belongs to {} with a {:.2f} percent confidence.".format([int(x) for x in range(10)][np.argmax(score)], 100 * np.max(score))
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    return response


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        #DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )
    # Tell flask it is behind proxy
    app.wsgi_app = ProxyFix(
        app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'
    
    @app.route("/", methods=['POST'])
    def upload():
        f = request.files['wav_file']
        filename = "uploaded_file.wav"
        #assert(f.content_type == "audio/wave")
        f.save(filename)
        
        return pipline(filename)

    return app