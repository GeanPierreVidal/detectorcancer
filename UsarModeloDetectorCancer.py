import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model_path = 'D:/Imagenes_Colpo_Modelo/image_classification_model.h5'
try:
    model = load_model(model_path)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

# Función para clasificar una imagen
def classify_image(image_path, model, target_size=(150, 150), confidence_threshold=0.5):
    try:
        # Cargar la imagen y prepararla para la predicción
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image /= 255.0  # Normalizar la imagen

        # Realizar la predicción
        prediction = model.predict(image)
        max_confidence = np.max(prediction)
        class_idx = np.argmax(prediction[0])
        class_label = list(model._get_trainable_state()[0].class_indices.keys())[class_idx]

        if max_confidence >= confidence_threshold:
            return class_label, max_confidence
        else:
            return "No válida", max_confidence

    except Exception as e:
        print(f"Error en la clasificación de la imagen: {e}")
        return None, None

# Ejemplo de uso
image_path = 'D:/Imagenes_Colpo_Modelo/pruebas/some_image.jpg'
class_label, confidence = classify_image(image_path, model, confidence_threshold=0.7)

if class_label and confidence:
    print(f'Clase predicha: {class_label}')
    print(f'Confianza: {confidence}')

    if class_label == "No válida":
        print("La predicción no supera el umbral de confianza establecido.")
else:
    print("Error al clasificar la imagen.")
