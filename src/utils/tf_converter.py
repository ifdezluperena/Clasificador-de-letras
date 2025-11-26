import tensorflow as tf

# Cargar modelo .keras
model = tf.keras.models.load_model("src/notebooks/model_80_sigmoid_softmax_relu_32_128_callback.keras")

# Convertir a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar archivo tflite
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversión completada: model.tflite creado")
