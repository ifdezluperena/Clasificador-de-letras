import streamlit as st
import zipfile, io, numpy as np
from PIL import Image
import tensorflow as tf

# Rutas
ZIP_PATH = "src/streamlit_data/LETTER_IMG_TEST-20251003T150011Z-1-001.zip"
MODEL_PATH = "src/notebooks/model_80_sigmoid_softmax_relu_32_128_callback.keras"



labels = ['A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i', 'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's', 'T', 't', 'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z']


# Cargar modelo una vez
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Cargar imágenes del zip
@st.cache_resource
def load_images_from_zip():
    images = {}
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for filename in z.namelist():
            if filename.endswith((".png", ".jpg", ".jpeg")):
                img_data = z.read(filename)
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                images[filename] = img
    return images

def preprocess(img, size=(32, 32)):
    img = img.resize(size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Cargar modelo e imágenes
model = load_model()
images = load_images_from_zip()
image_names = list(images.keys())

st.title("Demo Clasificación de Letras")

selected = st.selectbox("Elige una imagen de prueba:", image_names)

if selected:
    img = images[selected]
    st.image(img, width=200)

    # Preprocesar
    x = preprocess(img)

    # Predicción
    preds = model.predict(x)
    pred_idx = np.argmax(preds)
    pred_label = labels[pred_idx]

    st.write(f"### Predicción: **{pred_label}**")
