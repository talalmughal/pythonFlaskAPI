import os
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from flask_negotiate import consumes
from tensorflow import keras
from tensorflow.keras.models import load_model

# Provided file extension/format validation
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Configuring flask API
app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/api/": {"origins": ""}})
app.config["DEBUG"] = True

# Defining all of the classes
classes = ['Biryani', 'Burger', 'Fries', 'Pizza', 'Sandwich']

# Loading the already trained model from local storage
pathOfModel = "C:/tmp/model/"
loadedModel = load_model(pathOfModel)

# Path for image upload
uploadFolder = '/path/to/the/uploads'


# Routes
@app.route('/predict/class', methods=['POST'])
@cross_origin(origin='*')
@consumes('multipart/form-data')
# Manipulating the image
def uploadImage():
    try:
        # Requesting the file with the object key as backgroundImage
        # print(request.files)
        backgroundImage = request.files['backgroundImage']
        file_name = 'C:/Users/m3ham/Desktop/API2/uploads/' + backgroundImage.filename
        backgroundImage.save(file_name)

        # Resizing image into standard size
        img = keras.preprocessing.image.load_img(
            file_name, target_size=(180, 180)
        )
        # Converting image into an array
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Making prediction on that image array
        predictions = loadedModel.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        # Deleting the image from the 'uploads' folder, after processing
        os.unlink(file_name)

        # If the probability of success is less than 80%
        if (np.max(score) * 100) < 80:
            return jsonify('Object not identified')

        # If the success rate is above 80%
        else:
            return jsonify(
                {'predicted_class': classes[np.argmax(score)], "score": "{:.2f}".format(100 * np.max(score))}
            )

    # Exception handling
    except Exception as e:
        print("EXCEPTION IN CODE, {}".format(e))
        return jsonify({"Error": "An error Occurred"}), 500


# Driver Code
if __name__ == "__main__":
    app.run(host='0.0.0.0', threaded=True)
