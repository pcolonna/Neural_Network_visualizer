import json
import math

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

st.set_page_config(layout="wide")

URI = "http://127.0.0.1:5000"

st.title("Neural Network Visualizer")

st.sidebar.title("Parameters")

num_layers = st.sidebar.slider("Number of Layers", min_value=2, max_value=10, step=1)
hidden_units_per_layers = st.sidebar.slider("Number of hidden units per layers", min_value=2, max_value=64, step=1)

batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=256, step=16)
epochs = st.sidebar.slider("Epochs", min_value=2, max_value=30, step=1)

if st.sidebar.button("Train the new model"):
    progress_bar = st.sidebar.progress(0)
    response = requests.post(
        URI + "/train",
        json={
            "num_layers": num_layers,
            "hidden_units_per_layers": hidden_units_per_layers,
            "batch_size": batch_size,
            "epochs": epochs,
        },
    )

col_left, col_right = st.beta_columns(2)

with col_left:
    st.markdown("<h2> Model summary </h2>", unsafe_allow_html=True)
with col_right:
    st.markdown("<h2> Hidden units visualisation </h2>", unsafe_allow_html=True)

if st.sidebar.button("Get random predictions"):
    response = requests.post(URI, data={})
    response = json.loads(response.text)
    preds = response.get("prediction")
    image = response.get("image")
    image = np.reshape(image, (28, 28))

    st.sidebar.image(image, width=150)

    with col_left:
        summary = requests.post(URI + "/summary", data={}).text
        st.text(summary)

    with col_right:
        for layer, p in enumerate(preds):
            numbers = np.squeeze(np.array(p))

            fig = plt.figure(figsize=(32, 4))

            row = math.ceil(len(numbers) / 16)
            print(len(numbers) / 16)
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
