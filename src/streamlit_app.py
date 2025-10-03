import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# Cargar modelo
model = tf.keras.models.load_model("../src/notebooks/model_80_sigmoid_softmax_relu_32_128_callback.keras")

# Diccionario de etiquetas
labels = ['A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i', 'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's', 'T', 't', 'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z']

st.title("Clasificador de Letras con CNN")

uploaded_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Leer y preprocesar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img, (32, 32))
    img_normalized = img_resized.astype("float32") / 255.0

    # Predicción
    pred = model.predict(np.expand_dims(img_normalized, axis=0))
    label_idx = np.argmax(pred)
    prob = np.max(pred)

    st.image(img, caption="Imagen cargada", channels="BGR")
    st.write(f"**Predicción:** {labels[label_idx]} (confianza: {prob:.2f})")
