import json
import os
import random

import config
import numpy as np
import tensorflow as tf
from flask import Flask, request
from tensorflow.keras.models import load_model
import model

app = Flask(__name__)

saved_model = load_model(config.model_path)
feature_model = tf.keras.models.Model(saved_model.inputs, [layer.output for layer in saved_model.layers])

_, (x_test, _) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0


def get_prediction():
    index = np.random.choice(x_test.shape[0])
    image = x_test[index, :, :]
    image_arr = np.reshape(image, (1, 784))
    return feature_model.predict(image_arr), image


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        preds, image = get_prediction()
        final_preds = [p.tolist() for p in preds]
        return json.dumps({"prediction": final_preds, "image": image.tolist()})
    return "Welcome to the ml server"

@app.route("/train", methods=["POST"])
def train():
    
    data = request.get_json()
    
    num_layers=data.get('num_layers', 2)
    hidden_units_per_layers=data.get('hidden_units_per_layers', 2)
    batch_size=data.get('num_layers', 2)
    epochs= data.get('epochs', 2)

    model.train(num_layers=num_layers, hidden_units_per_layers=hidden_units_per_layers, epochs=epochs, batch_size=batch_size)

    return json.dumps({'status': 'done'})
if __name__ == "__main__":
    app.run()
