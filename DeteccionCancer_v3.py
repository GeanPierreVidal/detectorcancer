import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Ruta absoluta a los directorios
train_dir = 'D:/Imagenes_Colpo_Modelo/train'
validation_dir = 'D:/Imagenes_Colpo_Modelo/validation'

# Crear las carpetas si no existen
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Verificar que las rutas existen
assert os.path.isdir(train_dir), f"El directorio {train_dir} no existe."
assert os.path.isdir(validation_dir), f"El directorio {validation_dir} no existe."

# Verificar la versión de TensorFlow
print(f"TensorFlow version: {tf.__version__}")

# Crear un generador de datos para el entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalizar las imágenes
    rotation_range=40,  # Rotar las imágenes aleatoriamente
    width_shift_range=0.3,  # Desplazar las imágenes horizontalmente
    height_shift_range=0.3,  # Desplazar las imágenes verticalmente
    shear_range=0.3,  # Aplicar una transformación de corte
    zoom_range=0.3,  # Aplicar un zoom aleatorio
    horizontal_flip=True,  # Invertir las imágenes horizontalmente
    fill_mode='nearest'  # Rellenar los píxeles vacíos después de la transformación
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Crear generadores de datos
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Definir el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compilar el modelo
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

# Implementar Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,  # Incrementar el número de épocas
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping]
)

# Guardar el modelo entrenado en el nuevo formato
model.save('D:/Imagenes_Colpo_Modelo/image_classification_model.keras')

# Evaluar el modelo
loss, accuracy = model.evaluate(validation_generator)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Función para clasificar una imagen
# Función para clasificar una imagen
def classify_image(image_path):
    # Cargar la imagen y preprocesarla
    image = load_img(image_path, target_size=(150, 150))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0

    # Predecir la clase
    prediction = model.predict(image)
    class_idx = np.argmax(prediction[0])
    class_label = list(train_generator.class_indices.keys())[class_idx]

    # Verificar si la predicción es válida
    if prediction[0][class_idx] < 0.7:  # Ajusta este umbral según la confianza deseada
        return "No es una imagen del modelo"

    return f'La imagen pertenece a la clase: {class_label}'

# Ejemplo de uso
image_path = 'D:/Imagenes_Colpo_Modelo/pruebas/buscar.jpg'
class_label = classify_image(image_path)
print(f'La imagen pertenece a la clase: {class_label}')
