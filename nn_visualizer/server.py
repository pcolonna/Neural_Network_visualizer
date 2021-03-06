import json
import os
import random

import config
import model
import numpy as np
import tensorflow as tf
import utils
from flask import Flask, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

feature_model = None


def load_saved_model():
    global feature_model
    loaded_model = load_model(config.model_path)
    feature_model = tf.keras.models.Model(loaded_model.inputs, [layer.output for layer in loaded_model.layers])


def get_prediction():

    _, (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / 255.0

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

    num_layers = data.get("num_layers", 2)
    hidden_units_per_layers = data.get("hidden_units_per_layers", 2)
    batch_size = data.get("num_layers", 2)
    epochs = data.get("epochs", 2)

    _ = model.train(
        num_layers=num_layers, hidden_units_per_layers=hidden_units_per_layers, epochs=epochs, batch_size=batch_size
    )

    load_saved_model()

    return json.dumps({"status": "done"})


@app.route("/summary", methods=["POST"])
def summary():
    loaded_model = load_model(config.model_path)
    return utils.summarize(loaded_model)


if __name__ == "__main__":
    load_saved_model()
    app.run()
