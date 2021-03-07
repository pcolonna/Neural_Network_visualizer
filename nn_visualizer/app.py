import json

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

URI = "http://127.0.0.1:5000"

st.title("Neural Network Visualizer")

st.sidebar.title("Parameters")

num_layers = st.sidebar.slider("Number of Layers", min_value=2, max_value=10, step=1)
hidden_units_per_layers = st.sidebar.slider("Number of hidden units per layers", min_value=2, max_value=16, step=1)

batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=256, step=16)

if st.sidebar.button("Train the new model"):
    pass


if st.sidebar.button("Get random predictions"):
    response = requests.post(URI, data={})
    # print(response.text)
    response = json.loads(response.text)
    preds = response.get("prediction")
    image = response.get("image")
    image = np.reshape(image, (28, 28))

    st.sidebar.image(image, width=150)

    for layer, p in enumerate(preds):
        numbers = np.squeeze(np.array(p))

        fig = plt.figure(figsize=(32, 4))

        if layer == 2:
            row = 1
            col = 10
        else:
            row = 2
            col = 16

        for i, number in enumerate(numbers):
            plt.subplot(row, col, i + 1)
            plt.imshow((number * np.ones((8, 8, 3))).astype("float32"), cmap="binary")
            plt.xticks([])
            plt.yticks([])
            if layer == 2:
                plt.xlabel(str(i), fontsize=40)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()

        st.text("Layer {}".format(layer + 1))
        st.pyplot(fig)

# st.sidebar.markdown("## Input Image")

# image_canvas = st.sidebar.image(np.zeros((28, 28)), width=150)
