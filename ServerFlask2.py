from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import io
from PIL import Image
import os
import logging

# Configurar registro
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas y orígenes

# Ruta del modelo
model_path = 'image_classification_model_new.h5'

# Verificar si el modelo existe
if not os.path.exists(model_path):
    logging.error(f"No se encontró el archivo del modelo en {model_path}")
    raise FileNotFoundError(f"No se encontró el archivo del modelo en {model_path}")

logging.info(f"Cargando el modelo desde {model_path}")
model = load_model(model_path)
logging.info("Modelo cargado exitosamente")

# Umbral de confianza para clasificar imágenes no reconocidas
confidence_threshold = 0.6  # Ajusta este valor según tus necesidades

# Función para clasificar una imagen
def classify_image(image_path):
    image = load_img(image_path, target_size=(150, 150))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0

    # Realizar predicción
    prediction = model.predict(image)
    max_confidence = np.max(prediction)  # Mayor probabilidad entre las clases
    predicted_class_indices = np.argmax(prediction, axis=1)

    # Verificar si la confianza es menor al umbral
    if max_confidence < confidence_threshold:
        class_label = "Imagen no detectada en el modelo"
    else:
        # Obtener el nombre de la clase predicha
        class_label = "CANCER" if predicted_class_indices[0] == 0 else "NORMAL"

    return class_label

# Endpoint de predicción de imágenes
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se encontró el archivo"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No se seleccionó ningún archivo"}), 400

        image = Image.open(io.BytesIO(file.read()))
        image_path = 'temp.jpg'  # Guardar temporalmente la imagen en el servidor
        image.save(image_path)

        # Clasificar la imagen
        class_label = classify_image(image_path)

        return jsonify({"result": class_label})
    except Exception as e:
        error_message = f"Error en la predicción: {str(e)}"
        logging.error(error_message)
        return jsonify({"error": error_message}), 500

# Endpoint para incrementar el conocimiento del modelo con nuevas imágenes
@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        images = request.files.getlist("files")
        labels = request.form.getlist("labels")

        if not images or not labels or len(images) != len(labels):
            return jsonify({"error": "Debe enviar imágenes y etiquetas correspondientes"}), 400

        # Preparar los datos para reentrenamiento
        x_train = []
        y_train = []

        for file, label in zip(images, labels):
            image = Image.open(io.BytesIO(file.read()))
            image = image.resize((150, 150))
            image = img_to_array(image) / 255.0  # Normalizar
            x_train.append(image)

            # Convertir la etiqueta a formato categórico (0 para "CANCER", 1 para "NORMAL")
            y_train.append(0 if label.upper() == "CANCER" else 1)

        x_train = np.array(x_train)
        y_train = to_categorical(np.array(y_train), num_classes=2)

        # Cargar el modelo con pesos existentes
        logging.info("Cargando modelo para reentrenamiento incremental.")
        model = load_model(model_path)

        # Compilar el modelo con una tasa de aprendizaje baja para ajuste fino
        model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

        # Ajustar el modelo con los nuevos datos
        model.fit(x_train, y_train, epochs=3, batch_size=8, verbose=1)

        # Guardar el modelo actualizado
        model.save(model_path)
        logging.info("Modelo actualizado y guardado exitosamente con entrenamiento incremental.")

        return jsonify({"message": "Modelo reentrenado exitosamente"})
    except Exception as e:
        error_message = f"Error en el reentrenamiento: {str(e)}"
        logging.error(error_message)
        return jsonify({"error": error_message}), 500

# Endpoint de prueba
@app.route("/test", methods=["GET"])
def test():
    return jsonify({"message": "El servidor está funcionando correctamente"})

if __name__ == "__main__":
    app.run(debug=True)
