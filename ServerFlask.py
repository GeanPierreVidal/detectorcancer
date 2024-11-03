from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import io
from PIL import Image
import os
import logging

# Configurar registro
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas y orígenes

# Asegúrate de usar una ruta absoluta correcta
model_path = 'image_classification_model_new.h5'

# Verificar la existencia del archivo del modelo
if not os.path.exists(model_path):
    logging.error(f"No se encontró el archivo del modelo en {model_path}")
    raise FileNotFoundError(f"No se encontró el archivo del modelo en {model_path}")

logging.info(f"Cargando el modelo desde {model_path}")
model = load_model(model_path)
logging.info("Modelo cargado exitosamente")

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Preprocesar la imagen
def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = preprocess_input(image)  # Aplicar el preprocesamiento específico del modelo
    image = np.expand_dims(image, axis=0)
    return image


# Endpoint para la predicción
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        image = Image.open(io.BytesIO(file.read()))
        prepared_image = prepare_image(image, target_size=(150, 150))  # Ajusta el tamaño según tu modelo

        # Realiza la predicción utilizando el modelo
        predictions = model.predict(prepared_image)

        # Aquí puedes procesar las predicciones según sea necesario
        # Por ejemplo, convertir las predicciones en una lista
        predictions_list = predictions.tolist()

        return jsonify({"predictions": predictions_list})
    except Exception as e:
        error_message = f"Error en la predicción: {str(e)}"
        logging.error(error_message)
        return jsonify({"error": error_message}), 500


# Endpoint de prueba
@app.route("/test", methods=["GET"])
def test():
    return jsonify({"message": "El servidor está funcionando correctamente"})


if __name__ == "__main__":
    app.run(debug=True)
