import streamlit as st
import numpy as np 

st.title('Neural Network Visualizer')

st.sidebar.title('Parameters')

num_layers = st.sidebar.slider('Number of Layers', min_value=2, max_value=10, step=1)
hidden_units_per_layers = st.sidebar.slider('Number of hidden units per layers', min_value=2, max_value=16, step=1)

batch_size = st.sidebar.slider('Batch Size', min_value=16, max_value=256, step=16)

if st.sidebar.button('Train the new model'):
    pass


if st.sidebar.button('Get random predictions'):
    pass

st.sidebar.markdown('## Input Image')

image_canvas = st.sidebar.image(np.zeros((28,28)), width=150)