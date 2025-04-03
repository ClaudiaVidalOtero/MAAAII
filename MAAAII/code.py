# -*- coding: utf-8 -*-
"""
MAAAII Práctica 2: Aprendizaje Semi-supervisado y en Una Clase (CIFAR-100)

Curso 2024/2025
"""

# %% [markdown]
# # Práctica 2: Aprendizaje Semi-supervisado y en Una Clase
#
# **Grado en Inteligencia Artificial - Modelos Avanzados de Aprendizaje Automático II**
#
# **Universidade da Coruña**
#
# **Curso 2024/2025**
#
# En esta práctica exploraremos técnicas de aprendizaje semi-supervisado y en una clase sobre el conjunto de datos CIFAR-100.

# %% [markdown]
# ## Imports y Configuración Inicial

# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, metrics, utils, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import gc # Garbage Collector
import time

# %%
# Configuración de GPU (Opcional, pero recomendado)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restringir TensorFlow a usar solo la primera GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # Permitir crecimiento de memoria dinámico
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Usando GPU: {gpus[0].name}")
    except RuntimeError as e:
        # Error si la configuración se hace después de inicializar la GPU
        print(e)
else:
    print("No se encontró GPU, usando CPU.")

# %%
# Parámetros globales
NUM_CLASSES = 100
INPUT_SHAPE = (32, 32, 3)
BATCH_SIZE = 128
EPOCHS_SUPERVISED = 60 # Epochs for supervised training steps
EPOCHS_AE = 50 # Epochs for Autoencoder pre-training
EPOCHS_ROTNET = 50 # Epochs for RotationNet pre-training
PATIENCE_EARLY_STOPPING = 10 # Patience for Early Stopping
VERBOSE = 1 # Verbosity level for training

# Random seed for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# %% [markdown]
# ## Preparativos: Carga y División de Datos CIFAR-100
#
# Cargamos CIFAR-100 y dividimos el conjunto de entrenamiento (50,000 instancias) en:
# *   10,000 instancias etiquetadas (20%)
# *   40,000 instancias no etiquetadas (80%)
# El conjunto de prueba (10,000 instancias) permanece etiquetado.

# %%
# Cargar datos CIFAR-100
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar100.load_data()

print(f"Datos originales:")
print(f"x_train_full shape: {x_train_full.shape}, y_train_full shape: {y_train_full.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# %%
# Preprocesamiento: Normalizar imágenes a [0, 1]
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convertir etiquetas a categóricas (one-hot encoding)
y_train_full_cat = utils.to_categorical(y_train_full, NUM_CLASSES)
y_test_cat = utils.to_categorical(y_test, NUM_CLASSES)

# %%
# Dividir el conjunto de entrenamiento original
# Mantener las etiquetas originales para la división estratificada si es posible,
# aunque una división aleatoria simple también es común. Usaremos aleatoria simple aquí.
# Creamos índices y los barajamos
num_train_samples = x_train_full.shape[0]
indices = np.arange(num_train_samples)
np.random.shuffle(indices)

# Seleccionamos los índices para cada subconjunto
labeled_indices = indices[:10000]
unlabeled_indices = indices[10000:]

# Creamos los conjuntos de datos
x_train_labeled = x_train_full[labeled_indices]
y_train_labeled_cat = y_train_full_cat[labeled_indices]
y_train_labeled = y_train_full[labeled_indices] # Guardamos las etiquetas no categóricas también si son útiles

x_train_unlabeled = x_train_full[unlabeled_indices]
# No usamos las etiquetas de y_train_unlabeled para el entrenamiento SSL

print(f"\nDatos divididos:")
print(f"x_train_labeled shape: {x_train_labeled.shape}, y_train_labeled_cat shape: {y_train_labeled_cat.shape}")
print(f"x_train_unlabeled shape: {x_train_unlabeled.shape}")
print(f"x_test shape: {x_test.shape}, y_test_cat shape: {y_test_cat.shape}")

# Limpiar memoria
del x_train_full, y_train_full, y_train_full_cat
gc.collect()

# %% [markdown]
# ## Ejercicio 1: Entrenamiento Supervisado (Baseline)
#
# Entrenamos un modelo CNN usando únicamente las 10,000 instancias etiquetadas. El modelo debe tener al menos cuatro capas densas y/o convolucionales.

# %% [markdown]
# ### 1.a: ¿Qué red has escogido? ¿Por qué? ¿Cómo la has entrenado?
#
# **Red Escogida:** Se utilizará una Red Neuronal Convolucional (CNN) relativamente estándar para clasificación de imágenes de tamaño pequeño como CIFAR. Consta de bloques convolucionales seguidos de capas densas.
# *   **Bloques Convolucionales:** Se usan `Conv2D` con activación ReLU, seguidas de `BatchNormalization` (para estabilizar y acelerar el entrenamiento) y `MaxPooling2D` (para reducir la dimensionalidad espacial y aumentar la invarianza a pequeñas traslaciones). Se incrementa el número de filtros en capas más profundas para capturar características más complejas. Se incluye `Dropout` después de algunos bloques para regularización.
# *   **Capas Densas:** Después de aplanar (`Flatten`) la salida de las capas convolucionales, se añaden capas `Dense` con activación ReLU y `Dropout` para regularización.
# *   **Capa de Salida:** Una capa `Dense` con `NUM_CLASSES` neuronas y activación `softmax` para obtener probabilidades de clase.
#
# **Por qué:** Las CNN son el estado del arte para tareas de visión por computador, especialmente clasificación de imágenes, debido a su capacidad para aprender jerarquías de características espaciales. La arquitectura específica es un compromiso entre capacidad (suficiente profundidad/filtros para aprender de CIFAR-100) y eficiencia (no excesivamente grande para entrenar con recursos limitados). `BatchNormalization` y `Dropout` son cruciales para evitar el sobreajuste, especialmente con datos limitados.
#
# **Entrenamiento:**
# *   **Optimizador:** Adam con una tasa de aprendizaje inicial (e.g., 1e-3).
# *   **Función de Pérdida:** `categorical_crossentropy`, adecuada para clasificación multiclase con etiquetas one-hot.
# *   **Métricas:** `accuracy`.
# *   **Datos:** Se entrena exclusivamente con `x_train_labeled` y `y_train_labeled_cat`.
# *   **Validación:** Se usa el conjunto de test (`x_test`, `y_test_cat`) para monitorizar el rendimiento durante el entrenamiento.
# *   **Callbacks:** `EarlyStopping` monitorizando `val_loss` para detener el entrenamiento si no hay mejora y evitar sobreajuste. `ReduceLROnPlateau` podría añadirse para ajustar la tasa de aprendizaje.

# %%
def build_classifier_model(input_shape, num_classes, dropout_rate=0.3):
    """Construye el modelo CNN clasificador."""
    inputs = keras.Input(shape=input_shape)

    # Bloque 1
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Bloque 2
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Bloque 3
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x) # > 4 capas convolucionales

    # Clasificador Denso
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x) # Capa densa 1
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate + 0.1)(x)
    x = layers.Dense(128, activation='relu')(x) # Capa densa 2
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate + 0.1)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x) # Capa de salida

    model = keras.Model(inputs=inputs, outputs=outputs, name="baseline_classifier")
    return model

# Crear el modelo
model_ex1 = build_classifier_model(INPUT_SHAPE, NUM_CLASSES)
model_ex1.summary()

# Compilar el modelo
model_ex1.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                  loss=losses.CategoricalCrossentropy(),
                  metrics=[metrics.CategoricalAccuracy()])

# Callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss',
                                         patience=PATIENCE_EARLY_STOPPING,
                                         restore_best_weights=True,
                                         verbose=VERBOSE)

# %%
# Entrenar el modelo
print("\n--- Entrenando Modelo Ejercicio 1 (Supervisado Baseline) ---")
start_time = time.time()
history_ex1 = model_ex1.fit(x_train_labeled, y_train_labeled_cat,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS_SUPERVISED,
                            validation_data=(x_test, y_test_cat),
                            callbacks=[early_stopping],
                            verbose=VERBOSE)
end_time = time.time()
print(f"Tiempo de entrenamiento Ejercicio 1: {end_time - start_time:.2f} segundos")

# %% [markdown]
# ### 1.b: ¿Cuál es el rendimiento del modelo en entrenamiento? ¿Y en prueba?

# %%
# Evaluar el modelo en entrenamiento y prueba
print("\n--- Evaluación Modelo Ejercicio 1 ---")
loss_train_ex1, acc_train_ex1 = model_ex1.evaluate(x_train_labeled, y_train_labeled_cat, verbose=0)
loss_test_ex1, acc_test_ex1 = model_ex1.evaluate(x_test, y_test_cat, verbose=0)

print(f"Rendimiento en Entrenamiento:")
print(f"  Loss: {loss_train_ex1:.4f}")
print(f"  Accuracy: {acc_train_ex1:.4f}")

print(f"Rendimiento en Prueba:")
print(f"  Loss: {loss_test_ex1:.4f}")
print(f"  Accuracy: {acc_test_ex1:.4f}")

# Guardar resultados para comparación posterior
results = {'Ex1': {'train_acc': acc_train_ex1, 'test_acc': acc_test_ex1}}

# Graficar historial de entrenamiento (opcional pero útil)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_ex1.history['categorical_accuracy'], label='Train Accuracy')
plt.plot(history_ex1.history['val_categorical_accuracy'], label='Validation Accuracy')
plt.title('Ex1: Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_ex1.history['loss'], label='Train Loss')
plt.plot(history_ex1.history['val_loss'], label='Validation Loss')
plt.title('Ex1: Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ### 1.c: ¿Qué conclusiones sacas de los resultados detallados en el punto anterior?
#
# **Conclusiones:**
# *   El rendimiento en el conjunto de prueba (`test_acc_ex1`) establece nuestra **línea base (baseline)**. Este es el rendimiento que intentaremos mejorar con técnicas semi-supervisadas.
# *   Comparando `acc_train_ex1` y `acc_test_ex1`, podemos observar el grado de **sobreajuste**. Si la precisión de entrenamiento es significativamente mayor que la de prueba, indica sobreajuste, lo cual es esperable dado el pequeño tamaño del conjunto de entrenamiento etiquetado (10k muestras para 100 clases).
# *   La precisión absoluta obtenida (probablemente modesta, e.g., 20-40%) refleja la dificultad de clasificar CIFAR-100 con datos limitados.
# *   La curva de validación (accuracy y loss) debería haberse estabilizado o empezado a empeorar, momento en el cual `EarlyStopping` detuvo el entrenamiento para seleccionar el mejor modelo basado en el rendimiento de validación.

# %% [markdown]
# ## Ejercicio 2: Auto-aprendizaje (Self-Training)
#
# Entrenamos el mismo modelo, incorporando las instancias no etiquetadas mediante auto-aprendizaje. Opcionalmente, ponderamos las instancias según su certeza.

# %% [markdown]
# ### 2.a: ¿Qué parámetros has definido para el entrenamiento?
#
# **Parámetros de Auto-aprendizaje:**
# *   **Modelo Base:** Se utiliza la misma arquitectura definida en el Ejercicio 1. Podríamos usar los pesos pre-entrenados del Ejercicio 1 para generar las pseudo-etiquetas iniciales o entrenar desde cero. Usaremos los pesos de Ex1 para la predicción inicial.
# *   **Umbral de Confianza (Confidence Threshold):** Se define un umbral (e.g., `CONFIDENCE_THRESHOLD = 0.90`). Solo las predicciones del modelo sobre los datos no etiquetados (`x_train_unlabeled`) cuya máxima probabilidad (confianza) supere este umbral serán utilizadas como pseudo-etiquetas.
# *   **Iteraciones:** Para simplificar, realizaremos *una sola* iteración de auto-aprendizaje:
    1.  Predecir sobre `x_train_unlabeled` usando `model_ex1`.
    2.  Seleccionar predicciones de alta confianza.
    3.  Crear un nuevo conjunto de entrenamiento combinando `(x_train_labeled, y_train_labeled_cat)` y `(x_pseudo_labeled, y_pseudo_labeled)`.
    4.  Re-entrenar un *nuevo* modelo (misma arquitectura) desde cero sobre este conjunto combinado.
# *   **Ponderación (Opcional):** No se implementará ponderación inicialmente para mantener la simplicidad. Si se hiciera, se podría usar la confianza de la predicción como `sample_weight` en `model.fit()`.
# *   **Hiperparámetros de Re-entrenamiento:** Se usan los mismos que en el Ejercicio 1 (Adam, categorical crossentropy, batch size, early stopping).

# %%
CONFIDENCE_THRESHOLD = 0.90 # Umbral para seleccionar pseudo-etiquetas

# 1. Predecir sobre los datos no etiquetados usando el modelo baseline
print("\n--- Iniciando Auto-aprendizaje (Ejercicio 2) ---")
print("Generando predicciones sobre datos no etiquetados...")
preds_unlabeled = model_ex1.predict(x_train_unlabeled, batch_size=BATCH_SIZE, verbose=VERBOSE)

# 2. Seleccionar predicciones de alta confianza
pred_classes = np.argmax(preds_unlabeled, axis=1)
pred_confidences = np.max(preds_unlabeled, axis=1)

high_confidence_indices = np.where(pred_confidences >= CONFIDENCE_THRESHOLD)[0]

print(f"Se encontraron {len(high_confidence_indices)} instancias no etiquetadas con confianza >= {CONFIDENCE_THRESHOLD}")

if len(high_confidence_indices) > 0:
    x_pseudo_labeled = x_train_unlabeled[high_confidence_indices]
    y_pseudo_labeled = pred_classes[high_confidence_indices]
    y_pseudo_labeled_cat = utils.to_categorical(y_pseudo_labeled, NUM_CLASSES)

    # 3. Crear el nuevo conjunto de entrenamiento combinado
    x_train_combined = np.concatenate((x_train_labeled, x_pseudo_labeled), axis=0)
    y_train_combined_cat = np.concatenate((y_train_labeled_cat, y_pseudo_labeled_cat), axis=0)

    print(f"Nuevo tamaño del conjunto de entrenamiento combinado: {x_train_combined.shape[0]}")

    # Barajar el conjunto combinado
    shuffle_indices = np.arange(x_train_combined.shape[0])
    np.random.shuffle(shuffle_indices)
    x_train_combined = x_train_combined[shuffle_indices]
    y_train_combined_cat = y_train_combined_cat[shuffle_indices]

    # 4. Re-entrenar un nuevo modelo desde cero con los datos combinados
    print("Re-entrenando el modelo con datos etiquetados + pseudo-etiquetados...")
    model_ex2 = build_classifier_model(INPUT_SHAPE, NUM_CLASSES) # Nuevo modelo, misma arquitectura
    model_ex2.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                      loss=losses.CategoricalCrossentropy(),
                      metrics=[metrics.CategoricalAccuracy()])

    # Usar los mismos callbacks
    early_stopping_ex2 = callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=PATIENCE_EARLY_STOPPING,
                                                 restore_best_weights=True,
                                                 verbose=VERBOSE)

    start_time = time.time()
    history_ex2 = model_ex2.fit(x_train_combined, y_train_combined_cat,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS_SUPERVISED, # Usar el mismo número de épocas o ajustar
                                validation_data=(x_test, y_test_cat),
                                callbacks=[early_stopping_ex2],
                                verbose=VERBOSE)
    end_time = time.time()
    print(f"Tiempo de entrenamiento Ejercicio 2: {end_time - start_time:.2f} segundos")

else:
    print("No se encontraron suficientes instancias de alta confianza. Saltando re-entrenamiento.")
    print("El rendimiento del Ejercicio 2 será igual al del Ejercicio 1.")
    model_ex2 = model_ex1 # Usar el modelo anterior
    history_ex2 = history_ex1 # Usar historial anterior
    # O asignar NaNs o copiar resultados de Ex1 a los resultados de Ex2
    results['Ex2'] = results['Ex1'] # Copiar resultados si no se reentrena


# %% [markdown]
# ### 2.b: ¿Cuál es el rendimiento del modelo en entrenamiento? ¿Y en prueba?

# %%
# Evaluar el modelo de auto-aprendizaje
print("\n--- Evaluación Modelo Ejercicio 2 (Auto-aprendizaje) ---")

# Si no se reentrenó porque no había pseudo-etiquetas, los resultados son los de Ex1
if len(high_confidence_indices) == 0:
     print("No hubo re-entrenamiento. Usando resultados de Ex1.")
     loss_train_ex2, acc_train_ex2 = loss_train_ex1, acc_train_ex1
     loss_test_ex2, acc_test_ex2 = loss_test_ex1, acc_test_ex1
     # results['Ex2'] ya se copió
else:
    # Evaluar en el conjunto etiquetado original y en prueba
    # Nota: Evaluar en `x_train_combined` daría una métrica inflada por las pseudo-etiquetas.
    # Es más informativo evaluar sobre el conjunto etiquetado original.
    loss_train_ex2, acc_train_ex2 = model_ex2.evaluate(x_train_labeled, y_train_labeled_cat, verbose=0)
    loss_test_ex2, acc_test_ex2 = model_ex2.evaluate(x_test, y_test_cat, verbose=0)
    results['Ex2'] = {'train_acc': acc_train_ex2, 'test_acc': acc_test_ex2}


print(f"Rendimiento en Entrenamiento (sobre datos etiquetados originales):")
print(f"  Loss: {loss_train_ex2:.4f}")
print(f"  Accuracy: {acc_train_ex2:.4f}")

print(f"Rendimiento en Prueba:")
print(f"  Loss: {loss_test_ex2:.4f}")
print(f"  Accuracy: {acc_test_ex2:.4f}")


# %% [markdown]
# ### 2.c: ¿Se mejoran los resultados obtenidos en el Ejercicio 1?

# %%
print("\n--- Comparación Resultados Ex1 vs Ex2 ---")
print(f"Accuracy Test Ex1 (Baseline): {results['Ex1']['test_acc']:.4f}")
if 'Ex2' in results:
    print(f"Accuracy Test Ex2 (Self-Training): {results['Ex2']['test_acc']:.4f}")
    improvement = results['Ex2']['test_acc'] - results['Ex1']['test_acc']
    print(f"Mejora: {improvement:.4f} ({improvement/results['Ex1']['test_acc']:.2%})")
    if improvement > 0:
        print("Sí, los resultados han mejorado.")
    elif improvement < 0:
         print("No, los resultados han empeorado.")
    else:
        print("No ha habido cambios significativos en los resultados.")
else:
    print("No se ejecutó el entrenamiento del Ejercicio 2.")

# %% [markdown]
# ### 2.d: ¿Qué conclusiones sacas de los resultados detallados en los puntos anteriores?
#
# **Conclusiones:**
# *   **Efectividad del Auto-aprendizaje:** Comparar `acc_test_ex2` con `acc_test_ex1`. Si `acc_test_ex2` es mayor, el auto-aprendizaje ha sido efectivo. Esto sugiere que el modelo pudo extraer información útil de los datos no etiquetados a través de las pseudo-etiquetas.
# *   **Impacto del Umbral:** El umbral de confianza es crítico. Un umbral muy alto puede resultar en pocas pseudo-etiquetas, limitando el beneficio. Un umbral muy bajo puede introducir ruido si las pseudo-etiquetas son incorrectas ("confirmation bias"), potencialmente degradando el rendimiento. El valor de 0.90 es relativamente conservador.
# *   **Calidad de Pseudo-Etiquetas:** El éxito depende de la calidad del modelo inicial (`model_ex1`) y de si las predicciones de alta confianza son mayoritariamente correctas. Si el modelo inicial es pobre, las pseudo-etiquetas pueden ser erróneas.
# *   **Limitaciones:** El auto-aprendizaje simple puede reforzar errores si el modelo está muy seguro de predicciones incorrectas. Técnicas más avanzadas (múltiples iteraciones, umbrales adaptativos, ponderación) podrían ser necesarias.

# %% [markdown]
# ## Ejercicio 3: Autoencoder Semi-supervisado (Dos Pasos)
#
# 1.  Entrenar un Autoencoder (AE) en *todos* los datos de entrenamiento (`x_train_labeled` + `x_train_unlabeled`) para aprender representaciones.
# 2.  Usar el *encoder* pre-entrenado como extractor de características, congelar sus pesos, y entrenar un clasificador (capas densas) encima usando solo los datos *etiquetados* (`x_train_labeled`).
#
# La arquitectura del *encoder* debe ser la misma que la parte convolucional del clasificador de los Ejercicios 1 y 2.

# %% [markdown]
# ### 3.a: ¿Cuál es la arquitectura del modelo? ¿Y sus hiperparámetros?
#
# **Arquitectura:**
# *   **Encoder:** Utiliza las mismas capas convolucionales (`Conv2D`, `BatchNormalization`, `MaxPooling2D`) que el `build_classifier_model` hasta la capa `Flatten`.
# *   **Decoder:** Aproximadamente simétrico al encoder, usando capas `Conv2DTranspose` o `UpSampling2D` seguidas de `Conv2D` para reconstruir la imagen original. Termina con una capa `Conv2D` con 3 filtros (RGB) y activación `sigmoid` (ya que las imágenes de entrada están normalizadas a [0, 1]).
# *   **Autoencoder (AE):** El modelo completo que une encoder y decoder. Se entrena para minimizar el error de reconstrucción (e.g., MSE o Binary Crossentropy).
# *   **Clasificador Final:** Toma el *encoder* pre-entrenado (con pesos congelados), añade las capas densas del `build_classifier_model` (o nuevas capas densas) y la capa de salida softmax.
#
# **Hiperparámetros:**
# *   **Entrenamiento AE:**
#     *   Optimizador: Adam (e.g., `learning_rate=1e-3`).
#     *   Loss: `mean_squared_error` o `binary_crossentropy`. MSE es común para imágenes normalizadas.
#     *   Datos: `x_train_labeled` y `x_train_unlabeled` combinados.
#     *   Epochs: `EPOCHS_AE`.
#     *   Callbacks: `EarlyStopping` monitorizando `val_loss` (se puede usar `x_test` como validación para la reconstrucción, aunque no es ideal, o reservar una parte del `x_train_unlabeled`).
# *   **Entrenamiento Clasificador:**
#     *   Optimizador: Adam (puede requerir una tasa de aprendizaje menor, e.g., `1e-4`, ya que solo se ajustan las capas densas).
#     *   Loss: `categorical_crossentropy`.
#     *   Métricas: `accuracy`.
#     *   Datos: Solo `x_train_labeled` y `y_train_labeled_cat`.
#     *   Epochs: `EPOCHS_SUPERVISED`.
#     *   Callbacks: `EarlyStopping` monitorizando `val_loss`.

# %%
def build_autoencoder(input_shape, encoder_base_model):
    """Construye el Autoencoder usando la base convolucional dada."""

    # --- Encoder ---
    # Reutiliza las capas convolucionales del modelo clasificador
    # Necesitamos encontrar la última capa convolucional o de pooling antes de Flatten
    encoder_output_layer = None
    for layer in encoder_base_model.layers:
        if isinstance(layer, layers.Flatten):
            break
        encoder_output_layer = layer.output # Salida de la última capa antes de Flatten

    # Si no encontramos Flatten, usamos la salida del modelo base (menos probable)
    if encoder_output_layer is None:
         encoder_output_layer = encoder_base_model.layers[-2].output # Asumiendo penúltima capa es la relevante

    # Crear el modelo Encoder explícitamente
    encoder = keras.Model(inputs=encoder_base_model.input, outputs=encoder_output_layer, name="encoder")
    encoder.summary() # Verificar la salida del encoder

    # --- Decoder ---
    # La forma de la salida del encoder determinará la entrada del decoder
    latent_shape = encoder.output_shape[1:]
    decoder_input = keras.Input(shape=latent_shape)

    # Arquitectura simétrica inversa (ejemplo, ajustar según la salida del encoder)
    # Si la salida del encoder es (4, 4, 128) como en el modelo de ejemplo
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(decoder_input) # -> 8x8x128
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x) # -> 16x16x64
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(x) # -> 32x32x32
    x = layers.BatchNormalization()(x)
    decoder_output = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x) # -> 32x32x3 (reconstrucción)

    decoder = keras.Model(inputs=decoder_input, outputs=decoder_output, name="decoder")
    decoder.summary()

    # --- Autoencoder ---
    autoencoder_output = decoder(encoder.output)
    autoencoder = keras.Model(inputs=encoder.input, outputs=autoencoder_output, name="autoencoder")

    return autoencoder, encoder, decoder

# %%
# Crear un modelo base temporal solo para extraer la parte convolucional
temp_model = build_classifier_model(INPUT_SHAPE, NUM_CLASSES)
autoencoder, encoder, decoder = build_autoencoder(INPUT_SHAPE, temp_model)
del temp_model # Ya no necesitamos el modelo completo temporal
gc.collect()

autoencoder.summary()

# Compilar el Autoencoder
autoencoder.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                      loss=losses.MeanSquaredError()) # Usar MSE para reconstrucción

# Preparar datos para AE (todos los datos de entrenamiento)
x_train_for_ae = np.concatenate((x_train_labeled, x_train_unlabeled), axis=0)
np.random.shuffle(x_train_for_ae) # Barajar para el entrenamiento

print(f"\nEntrenando Autoencoder en {x_train_for_ae.shape[0]} imágenes...")

# Callback para AE
early_stopping_ae = callbacks.EarlyStopping(monitor='val_loss',
                                            patience=PATIENCE_EARLY_STOPPING // 2, # Menos paciencia para AE
                                            restore_best_weights=True,
                                            verbose=VERBOSE)

# Entrenar el Autoencoder
# Usar x_test como validación para reconstrucción es una opción simple
start_time = time.time()
history_ae = autoencoder.fit(x_train_for_ae, x_train_for_ae, # Entrada y salida son las mismas imágenes
                             batch_size=BATCH_SIZE,
                             epochs=EPOCHS_AE,
                             validation_data=(x_test, x_test),
                             callbacks=[early_stopping_ae],
                             verbose=VERBOSE)
end_time = time.time()
print(f"Tiempo de entrenamiento Autoencoder: {end_time - start_time:.2f} segundos")


# %%
# 2. Construir y entrenar el clasificador usando el encoder pre-entrenado
print("\nConstruyendo y entrenando el clasificador con Encoder congelado...")

# Congelar los pesos del encoder
encoder.trainable = False

# Construir el clasificador final
classifier_input = keras.Input(shape=INPUT_SHAPE)
features = encoder(classifier_input) # Usar el encoder pre-entrenado
flat_features = layers.Flatten()(features)

# Añadir las capas densas (iguales o similares al modelo original)
x = layers.Dense(256, activation='relu')(flat_features)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x) # Usar el mismo dropout o ajustarlo
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
classifier_output = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model_ex3 = keras.Model(inputs=classifier_input, outputs=classifier_output, name="classifier_from_ae")
model_ex3.summary()

# Compilar el clasificador (puede necesitar lr más bajo)
model_ex3.compile(optimizer=optimizers.Adam(learning_rate=5e-4), # Probar lr más bajo
                  loss=losses.CategoricalCrossentropy(),
                  metrics=[metrics.CategoricalAccuracy()])

# Entrenar solo con datos etiquetados
early_stopping_ex3 = callbacks.EarlyStopping(monitor='val_loss',
                                             patience=PATIENCE_EARLY_STOPPING,
                                             restore_best_weights=True,
                                             verbose=VERBOSE)

start_time = time.time()
history_ex3 = model_ex3.fit(x_train_labeled, y_train_labeled_cat,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS_SUPERVISED,
                            validation_data=(x_test, y_test_cat),
                            callbacks=[early_stopping_ex3],
                            verbose=VERBOSE)
end_time = time.time()
print(f"Tiempo de entrenamiento Clasificador (Ex3): {end_time - start_time:.2f} segundos")


# %% [markdown]
# ### 3.b: ¿Cuál es el rendimiento del modelo en entrenamiento? ¿Y en prueba?

# %%
# Evaluar el modelo final del Ejercicio 3
print("\n--- Evaluación Modelo Ejercicio 3 (AE + Classifier) ---")
loss_train_ex3, acc_train_ex3 = model_ex3.evaluate(x_train_labeled, y_train_labeled_cat, verbose=0)
loss_test_ex3, acc_test_ex3 = model_ex3.evaluate(x_test, y_test_cat, verbose=0)

results['Ex3'] = {'train_acc': acc_train_ex3, 'test_acc': acc_test_ex3}

print(f"Rendimiento en Entrenamiento (sobre datos etiquetados):")
print(f"  Loss: {loss_train_ex3:.4f}")
print(f"  Accuracy: {acc_train_ex3:.4f}")

print(f"Rendimiento en Prueba:")
print(f"  Loss: {loss_test_ex3:.4f}")
print(f"  Accuracy: {acc_test_ex3:.4f}")


# %% [markdown]
# ### 3.c: ¿Se mejoran los resultados obtenidos en los Ejercicios 1 y 2?

# %%
print("\n--- Comparación Resultados Ex1, Ex2 vs Ex3 ---")
print(f"Accuracy Test Ex1 (Baseline): {results['Ex1']['test_acc']:.4f}")
if 'Ex2' in results:
    print(f"Accuracy Test Ex2 (Self-Training): {results['Ex2']['test_acc']:.4f}")
else:
    print("Accuracy Test Ex2: No disponible")

print(f"Accuracy Test Ex3 (AE Pre-trained): {results['Ex3']['test_acc']:.4f}")

if 'Ex2' in results:
    improvement_vs_ex1 = results['Ex3']['test_acc'] - results['Ex1']['test_acc']
    improvement_vs_ex2 = results['Ex3']['test_acc'] - results['Ex2']['test_acc']
    print(f"Mejora vs Ex1: {improvement_vs_ex1:.4f} ({improvement_vs_ex1/results['Ex1']['test_acc']:.2%})")
    print(f"Mejora vs Ex2: {improvement_vs_ex2:.4f} ({improvement_vs_ex2/results['Ex2']['test_acc']:.2%})")
    if improvement_vs_ex1 > 0 or improvement_vs_ex2 > 0:
        print("Sí, los resultados han mejorado respecto a al menos uno de los ejercicios anteriores.")
    else:
        print("No se observan mejoras significativas respecto a los ejercicios anteriores.")
else:
     improvement_vs_ex1 = results['Ex3']['test_acc'] - results['Ex1']['test_acc']
     print(f"Mejora vs Ex1: {improvement_vs_ex1:.4f} ({improvement_vs_ex1/results['Ex1']['test_acc']:.2%})")
     if improvement_vs_ex1 > 0:
        print("Sí, los resultados han mejorado respecto al ejercicio 1.")
     else:
        print("No se observan mejoras significativas respecto al ejercicio 1.")


# %% [markdown]
# ### 3.d: ¿Qué conclusiones sacas de los resultados detallados en los puntos anteriores?
#
# **Conclusiones:**
# *   **Efectividad del Pre-entrenamiento:** Si `acc_test_ex3` es significativamente mayor que `acc_test_ex1` (y posiblemente `acc_test_ex2`), indica que el pre-entrenamiento no supervisado del encoder con todos los datos (etiquetados y no etiquetados) ayudó a aprender representaciones de características más robustas y generalizables. Estas características son beneficiosas para la tarea de clasificación final, incluso cuando se entrena solo con los datos etiquetados.
# *   **Calidad de Reconstrucción AE:** Aunque no se evalúa directamente aquí, la calidad de la reconstrucción del AE (baja pérdida MSE/BCE) es un indicador indirecto de que el encoder está capturando información relevante de las imágenes.
# *   **Transferencia de Aprendizaje:** Este enfoque es una forma de transferencia de aprendizaje, donde el conocimiento adquirido en la tarea de reconstrucción (no supervisada) se transfiere a la tarea de clasificación (supervisada).
# *   **Comparación con Self-Training:** Comparar Ex3 con Ex2. A veces, el pre-entrenamiento AE proporciona una mejora más estable que el self-training, ya que no depende de la generación potencialmente ruidosa de pseudo-etiquetas. Sin embargo, el resultado puede variar según el dataset y la implementación.

# %% [markdown]
# ## Ejercicio 4: Autoencoder Semi-supervisado (Un Paso)
#
# Entrenar un modelo que combine el Autoencoder y el Clasificador, optimizando una pérdida combinada (reconstrucción + clasificación) en un solo paso.
# *   Arquitectura del AE: La misma que en el Ejercicio 3.
# *   Arquitectura del Clasificador (Encoder + Cabezal): La misma que en el Ejercicio 1.
#
# Esto requiere un enfoque más complejo, a menudo una subclase de `keras.Model` con un `train_step` personalizado.

# %% [markdown]
# ### 4.a: ¿Cuál es la arquitectura del modelo? ¿Y sus hiperparámetros?
#
# **Arquitectura:**
# *   **Encoder Compartido:** Se utiliza el mismo *encoder* definido en el Ejercicio 3 (parte convolucional del clasificador).
# *   **Cabezal de Reconstrucción (Decoder):** El mismo *decoder* definido en el Ejercicio 3.
# *   **Cabezal de Clasificación:** Las capas densas y la capa softmax del clasificador del Ejercicio 1.
# *   **Modelo Combinado:** Un modelo con una entrada (imagen) y *dos* salidas: la imagen reconstruida y las probabilidades de clase.
#
# **Hiperparámetros:**
# *   **Pérdidas:**
#     *   Pérdida de Reconstrucción (`loss_recon`): `mean_squared_error` o `binary_crossentropy`.
#     *   Pérdida de Clasificación (`loss_class`): `categorical_crossentropy`.
# *   **Ponderación de Pérdidas (`lambda_recon`):** Un hiperparámetro clave que balancea las dos tareas. La pérdida total es `Loss = loss_class + lambda_recon * loss_recon`. El valor de `lambda_recon` necesita ser ajustado (e.g., empezar con 1.0, 0.1, 10.0).
# *   **Entrenamiento:**
#     *   Optimizador: Adam.
#     *   Datos: Se requiere un generador de datos o un `train_step` personalizado que maneje tanto los datos etiquetados (para ambas pérdidas) como los no etiquetados (solo para la pérdida de reconstrucción).
#     *   Epochs: Similar a los ejercicios anteriores.
#     *   Callbacks: `EarlyStopping` (monitorizando `val_categorical_accuracy` o una combinación de pérdidas de validación).

# %%
# Implementación usando subclase de Model con train_step personalizado

class SemiSupervisedAE(keras.Model):
    def __init__(self, encoder, decoder, classifier_head, lambda_recon=1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier_head = classifier_head
        self.lambda_recon = lambda_recon

        # Métricas separadas para cada tarea
        self.loss_tracker = metrics.Mean(name="total_loss")
        self.recon_loss_tracker = metrics.Mean(name="recon_loss")
        self.class_loss_tracker = metrics.Mean(name="class_loss")
        self.class_accuracy = metrics.CategoricalAccuracy(name="class_accuracy")

    def call(self, inputs, training=False):
        # La llamada 'call' es principalmente para inferencia (predicción)
        # Aquí definimos cómo obtener las predicciones de clase
        latent = self.encoder(inputs, training=training)
        classification_output = self.classifier_head(latent, training=training)
        # Para obtener la reconstrucción, llamaríamos explícitamente a encoder->decoder
        return classification_output

    def build_graph(self, input_shape):
         # Necesario para poder llamar a summary() antes de entrenar
         inputs = keras.Input(shape=input_shape)
         return keras.Model(inputs=inputs, outputs=self.call(inputs))

    @property
    def metrics(self):
        # Lista de métricas a resetear en cada época
        return [self.loss_tracker, self.recon_loss_tracker, self.class_loss_tracker, self.class_accuracy]

    def train_step(self, data):
        # Asumimos que 'data' es una tupla: (labeled_data, unlabeled_data)
        # donde labeled_data es (x_l, y_l)
        labeled_data, unlabeled_data = data
        x_l, y_l = labeled_data
        x_u = unlabeled_data # Solo imágenes

        # Combinar datos para la pérdida de reconstrucción
        x_combined = tf.concat((x_l, x_u), axis=0)

        with tf.GradientTape() as tape:
            # 1. Pérdida de Reconstrucción (sobre datos combinados)
            latent_combined = self.encoder(x_combined, training=True)
            reconstructions = self.decoder(latent_combined, training=True)
            recon_loss = tf.reduce_mean(
                losses.mean_squared_error(tf.cast(x_combined, tf.float32), reconstructions) # Asegurar tipos
            ) * self.lambda_recon

            # 2. Pérdida de Clasificación (solo sobre datos etiquetados)
            latent_labeled = self.encoder(x_l, training=True) # Re-calcular o extraer del paso anterior
            # Alternativa: extraer la parte correspondiente de latent_combined
            # latent_labeled = latent_combined[:tf.shape(x_l)[0]] # Más eficiente
            predictions_labeled = self.classifier_head(latent_labeled, training=True)
            class_loss = self.compiled_loss(y_l, predictions_labeled, regularization_losses=self.losses) # Usa la loss compilada

            # 3. Pérdida Total
            total_loss = class_loss + recon_loss

        # Calcular gradientes y actualizar pesos
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Actualizar métricas
        self.loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss / self.lambda_recon) # Guardar la pérdida de recon no ponderada
        self.class_loss_tracker.update_state(class_loss)
        self.class_accuracy.update_state(y_l, predictions_labeled)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Para la validación, solo usamos la parte de clasificación sobre el test set
        x, y = data
        latent = self.encoder(x, training=False)
        predictions = self.classifier_head(latent, training=False)
        class_loss = self.compiled_loss(y, predictions, regularization_losses=self.losses)

        # Solo actualizamos métricas de clasificación para validación
        self.class_loss_tracker.update_state(class_loss) # O usar un val_loss_tracker dedicado
        self.class_accuracy.update_state(y, predictions) # O usar un val_acc_tracker dedicado

        # Devolvemos las métricas relevantes para validación
        return {"val_loss": class_loss, "val_class_accuracy": self.class_accuracy.result()}


# %%
# Reconstruir las partes del modelo (asegurarse de que no estén congeladas)

# 1. Encoder (parte convolucional, igual que antes)
temp_model_for_ex4 = build_classifier_model(INPUT_SHAPE, NUM_CLASSES)
encoder_ex4_output_layer = None
for layer in temp_model_for_ex4.layers:
    if isinstance(layer, layers.Flatten):
        break
    encoder_ex4_output_layer = layer.output
encoder_ex4 = keras.Model(inputs=temp_model_for_ex4.input, outputs=encoder_ex4_output_layer, name="encoder_ex4")
encoder_ex4.trainable = True # Asegurarse de que es entrenable

# 2. Decoder (igual que antes)
latent_shape_ex4 = encoder_ex4.output_shape[1:]
decoder_input_ex4 = keras.Input(shape=latent_shape_ex4)
x_dec = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(decoder_input_ex4)
x_dec = layers.BatchNormalization()(x_dec)
x_dec = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x_dec)
x_dec = layers.BatchNormalization()(x_dec)
x_dec = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(x_dec)
x_dec = layers.BatchNormalization()(x_dec)
decoder_output_ex4 = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x_dec)
decoder_ex4 = keras.Model(inputs=decoder_input_ex4, outputs=decoder_output_ex4, name="decoder_ex4")
decoder_ex4.trainable = True # Asegurarse de que es entrenable

# 3. Cabezal Clasificador (parte densa del modelo original)
classifier_head_input = keras.Input(shape=latent_shape_ex4) # Entrada es la salida del encoder
x_cls = layers.Flatten()(classifier_head_input) # Aplanar si la salida del encoder no lo es
x_cls = layers.Dense(256, activation='relu')(x_cls)
x_cls = layers.BatchNormalization()(x_cls)
x_cls = layers.Dropout(0.4)(x_cls)
x_cls = layers.Dense(128, activation='relu')(x_cls)
x_cls = layers.BatchNormalization()(x_cls)
x_cls = layers.Dropout(0.4)(x_cls)
classifier_head_output = layers.Dense(NUM_CLASSES, activation='softmax')(x_cls)
classifier_head_ex4 = keras.Model(inputs=classifier_head_input, outputs=classifier_head_output, name="classifier_head_ex4")
classifier_head_ex4.trainable = True # Asegurarse de que es entrenable


# Crear la instancia del modelo combinado
lambda_reconstruction = 1.0 # Hiperparámetro a ajustar
model_ex4 = SemiSupervisedAE(encoder_ex4, decoder_ex4, classifier_head_ex4, lambda_recon=lambda_reconstruction)

# Compilar el modelo (solo necesitamos la loss de clasificación aquí, la de recon se maneja en train_step)
model_ex4.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                  loss=losses.CategoricalCrossentropy(), # Solo para la parte de clasificación
                  metrics=[metrics.CategoricalAccuracy()]) # Idem


# %%
# Preparar los datos para el entrenamiento combinado
# Necesitamos un generador o dataset que proporcione pares (labeled_batch, unlabeled_batch)

# Asegurarse de que y_train_labeled_cat tenga la forma correcta
if len(y_train_labeled_cat.shape) == 1:
    y_train_labeled_cat_ds = utils.to_categorical(y_train_labeled_cat, NUM_CLASSES)
else:
    y_train_labeled_cat_ds = y_train_labeled_cat

# Crear datasets de TF
labeled_ds = tf.data.Dataset.from_tensor_slices((x_train_labeled, y_train_labeled_cat_ds)).shuffle(buffer_size=10000).batch(BATCH_SIZE // 2) # Mitad del batch size
unlabeled_ds = tf.data.Dataset.from_tensor_slices(x_train_unlabeled).shuffle(buffer_size=40000).batch(BATCH_SIZE // 2).repeat() # Repetir indefinidamente

# Combinar los datasets
# Tomamos un batch de etiquetados y uno de no etiquetados en cada paso
train_ds_ex4 = tf.data.Dataset.zip((labeled_ds, unlabeled_ds))
# Prefetch para rendimiento
train_ds_ex4 = train_ds_ex4.prefetch(tf.data.experimental.AUTOTUNE)

# Crear dataset de validación (solo clasificación)
val_ds_ex4 = tf.data.Dataset.from_tensor_slices((x_test, y_test_cat)).batch(BATCH_SIZE)
val_ds_ex4 = val_ds_ex4.prefetch(tf.data.experimental.AUTOTUNE)


# %%
# Entrenar el modelo combinado
print("\n--- Entrenando Modelo Ejercicio 4 (AE + Classifier - Un Paso) ---")

# Definir Early Stopping monitorizando la métrica de validación correcta
# El nombre de la métrica de validación es 'val_class_accuracy' según nuestro test_step
early_stopping_ex4 = callbacks.EarlyStopping(monitor='val_class_accuracy', # O 'val_loss' si se prefiere
                                             mode='max', # Maximizar accuracy
                                             patience=PATIENCE_EARLY_STOPPING,
                                             restore_best_weights=True,
                                             verbose=VERBOSE)

# Calcular steps_per_epoch
steps_per_epoch = len(x_train_labeled) // (BATCH_SIZE // 2)

start_time = time.time()
history_ex4 = model_ex4.fit(train_ds_ex4,
                            epochs=EPOCHS_SUPERVISED, # Ajustar epochs si es necesario
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_ds_ex4,
                            callbacks=[early_stopping_ex4],
                            verbose=VERBOSE)
end_time = time.time()
print(f"Tiempo de entrenamiento Ejercicio 4: {end_time - start_time:.2f} segundos")


# %% [markdown]
# ### 4.b: ¿Cuál es el rendimiento del modelo en entrenamiento? ¿Y en prueba?

# %%
# Evaluar el modelo final del Ejercicio 4 (solo la parte de clasificación)
print("\n--- Evaluación Modelo Ejercicio 4 (AE + Classifier - Un Paso) ---")

# Evaluar sobre el conjunto de entrenamiento etiquetado original
# Necesitamos llamar a evaluate con datos que tengan formato (x, y)
loss_train_ex4, acc_train_ex4 = model_ex4.evaluate(x_train_labeled, y_train_labeled_cat, batch_size=BATCH_SIZE, verbose=0)

# Evaluar sobre el conjunto de prueba
loss_test_ex4, acc_test_ex4 = model_ex4.evaluate(x_test, y_test_cat, batch_size=BATCH_SIZE, verbose=0)


results['Ex4'] = {'train_acc': acc_train_ex4, 'test_acc': acc_test_ex4}

print(f"Rendimiento en Entrenamiento (sobre datos etiquetados originales):")
print(f"  Loss: {loss_train_ex4:.4f}")
print(f"  Accuracy: {acc_train_ex4:.4f}")

print(f"Rendimiento en Prueba:")
print(f"  Loss: {loss_test_ex4:.4f}")
print(f"  Accuracy: {acc_test_ex4:.4f}")


# %% [markdown]
# ### 4.c: ¿Se mejoran los resultados obtenidos en los ejercicios anteriores?

# %%
print("\n--- Comparación Resultados Ex1, Ex2, Ex3 vs Ex4 ---")
print(f"Accuracy Test Ex1 (Baseline): {results['Ex1']['test_acc']:.4f}")
if 'Ex2' in results:
    print(f"Accuracy Test Ex2 (Self-Training): {results['Ex2']['test_acc']:.4f}")
else:
    print("Accuracy Test Ex2: No disponible")
print(f"Accuracy Test Ex3 (AE Pre-trained): {results['Ex3']['test_acc']:.4f}")
print(f"Accuracy Test Ex4 (AE Joint): {results['Ex4']['test_acc']:.4f}")

improvement_vs_best_prev = results['Ex4']['test_acc'] - max(results['Ex1']['test_acc'],
                                                             results.get('Ex2', {'test_acc': -1})['test_acc'],
                                                             results['Ex3']['test_acc'])

print(f"\nMejora vs el mejor anterior: {improvement_vs_best_prev:.4f}")

if improvement_vs_best_prev > 0:
    print("Sí, los resultados han mejorado respecto a los ejercicios anteriores.")
elif improvement_vs_best_prev < 0:
     print("No, los resultados han empeorado respecto al mejor anterior.")
else:
    print("No ha habido cambios significativos respecto al mejor anterior.")


# %% [markdown]
# ### 4.d: ¿Qué conclusiones sacas de los resultados detallados en los puntos anteriores?
#
# **Conclusiones:**
# *   **Potencial del Entrenamiento Conjunto:** Teóricamente, el entrenamiento conjunto (Ex4) podría ser superior al de dos pasos (Ex3) porque permite que las tareas de reconstrucción y clasificación se beneficien mutuamente durante todo el proceso. El gradiente de la pérdida de clasificación puede influir en el encoder de manera útil para la reconstrucción, y viceversa.
# *   **Complejidad y Ajuste:** Este enfoque es más complejo de implementar y ajustar. El balance entre las pérdidas (`lambda_recon`) es crucial y puede requerir experimentación. Si la reconstrucción domina demasiado, podría perjudicar la clasificación, y viceversa.
# *   **Convergencia:** La convergencia puede ser más lenta o inestable en comparación con el enfoque de dos pasos. Monitorizar ambas pérdidas (reconstrucción y clasificación) durante el entrenamiento es importante.
# *   **Comparación Final:** Comparar `acc_test_ex4` con `acc_test_ex1`, `acc_test_ex2` y `acc_test_ex3`. Si Ex4 supera a los demás, demuestra la ventaja del entrenamiento conjunto para este problema. Si no, podría indicar problemas de ajuste, una `lambda_recon` inadecuada, o que el enfoque de dos pasos es suficiente o incluso mejor en este caso.

# %% [markdown]
# ## Ejercicio 5: Eliminación de Instancias Atípicas (One-Class Classification)
#
# Repetir los entrenamientos de los Ejercicios 1-4, pero eliminando primero las instancias no etiquetadas (`x_train_unlabeled`) consideradas más atípicas respecto a los datos etiquetados (`x_train_labeled`).
# *   Se usará una técnica de clasificación en una clase (OCC).
# *   La arquitectura de la red OCC será la misma que el clasificador del Ejercicio 1 (excepto la capa de salida), usándola como extractor de características.
# *   Se utilizará la técnica explicada en "Notebook 5" con v=0.9. **Interpretación:** Usaremos `IsolationForest` (o `OneClassSVM`) entrenado en las características de `x_train_labeled` para puntuar `x_train_unlabeled`. Conservaremos el 90% de las instancias no etiquetadas con las puntuaciones menos anómalas (correspondiente a v=0.9 como la fracción de *inliers* deseada, o 1-v = 0.1 como fracción de outliers a eliminar).

# %%
# --- Paso 1: Extracción de Características ---
print("\n--- Ejercicio 5: Preparación - Extracción de Features ---")
# Usar el modelo baseline (Ex1) o el encoder pre-entrenado (Ex3) como extractor
# Usemos el modelo Ex1, quitando la última capa.
feature_extractor = keras.Model(inputs=model_ex1.input,
                                outputs=model_ex1.layers[-2].output, # Salida antes de la capa softmax
                                name="feature_extractor_ex1")
feature_extractor.trainable = False # No necesitamos entrenarlo más

# Extraer características de datos etiquetados y no etiquetados
print("Extrayendo features de datos etiquetados...")
features_labeled = feature_extractor.predict(x_train_labeled, batch_size=BATCH_SIZE, verbose=VERBOSE)
print("Extrayendo features de datos no etiquetados...")
features_unlabeled = feature_extractor.predict(x_train_unlabeled, batch_size=BATCH_SIZE, verbose=VERBOSE)

print(f"Shape features etiquetadas: {features_labeled.shape}")
print(f"Shape features no etiquetadas: {features_unlabeled.shape}")

# (Opcional pero recomendado) Escalar características
# scaler = StandardScaler()
# features_labeled_scaled = scaler.fit_transform(features_labeled)
# features_unlabeled_scaled = scaler.transform(features_unlabeled)
# Usaremos las no escaladas por simplicidad, pero escalar puede ayudar a algunos modelos OCC.

# %%
# --- Paso 2: Entrenamiento del Modelo OCC y Filtrado ---
print("\n--- Ejercicio 5: Entrenamiento OCC y Filtrado ---")

# Usaremos Isolation Forest
# contamination = 0.1 corresponde a esperar un 10% de outliers (1 - v = 1 - 0.9)
occ_model = IsolationForest(contamination=0.1, random_state=SEED, n_jobs=-1) # n_jobs=-1 para usar todos los cores

print("Entrenando Isolation Forest en features etiquetadas...")
start_time = time.time()
occ_model.fit(features_labeled) # Entrenar solo en lo "normal" (etiquetado)
end_time = time.time()
print(f"Tiempo de entrenamiento OCC: {end_time - start_time:.2f} segundos")

# Predecir en los datos no etiquetados (-1 para outlier, 1 para inlier)
print("Prediciendo outliers en features no etiquetadas...")
predictions_occ = occ_model.predict(features_unlabeled)

# Alternativa: Usar anomaly scores y un cuantil
# scores_occ = occ_model.decision_function(features_unlabeled) # Menor score = más anómalo
# score_threshold = np.percentile(scores_occ, 10) # Umbral para eliminar el 10% más anómalo
# inlier_indices_occ = np.where(scores_occ >= score_threshold)[0]

inlier_indices_occ = np.where(predictions_occ == 1)[0]
outlier_indices_occ = np.where(predictions_occ == -1)[0]

print(f"Instancias no etiquetadas originales: {len(x_train_unlabeled)}")
print(f"Instancias conservadas (inliers): {len(inlier_indices_occ)}")
print(f"Instancias eliminadas (outliers): {len(outlier_indices_occ)}")
print(f"Fracción de outliers eliminada: {len(outlier_indices_occ) / len(x_train_unlabeled):.2%}")

# Crear el nuevo conjunto de datos no etiquetados filtrado
x_train_unlabeled_filtered = x_train_unlabeled[inlier_indices_occ]

print(f"Nuevo shape de datos no etiquetados filtrados: {x_train_unlabeled_filtered.shape}")

# Limpiar memoria
del features_labeled, features_unlabeled, occ_model, predictions_occ
gc.collect()

# %% [markdown]
# --- Paso 3: Repetir Ejercicios 1-4 con Datos Filtrados ---
#
# Ahora, se deben **re-ejecutar** las celdas de código correspondientes a los entrenamientos y evaluaciones de los ejercicios 1 a 4, pero utilizando `x_train_unlabeled_filtered` en lugar de `x_train_unlabeled` donde sea aplicable.
#
# *   **Ejercicio 1 (Repetido):** No cambia, ya que solo usa datos etiquetados. Los resultados serán los mismos que `results['Ex1']`.
# *   **Ejercicio 2 (Repetido - Self-Training):** Usar `x_train_unlabeled_filtered` para generar pseudo-etiquetas.
# *   **Ejercicio 3 (Repetido - AE Dos Pasos):** Entrenar el AE en `x_train_labeled` + `x_train_unlabeled_filtered`. Entrenar el clasificador solo en `x_train_labeled`.
# *   **Ejercicio 4 (Repetido - AE Un Paso):** Usar `x_train_unlabeled_filtered` como los datos no etiquetados en el `train_step` personalizado o generador.
#
# **NOTA:** Para evitar redefinir todo, se podrían encapsular los entrenamientos en funciones o crear nuevas variables (e.g., `model_ex2_filtered`, `results_filtered`). Por brevedad aquí, solo describiremos el proceso y la pregunta final. Asumiremos que se han re-ejecutado los pasos y obtenido los nuevos resultados (e.g., en un diccionario `results_filtered`).

# %%
# --- EJEMPLO de cómo se haría para Ex2 (Self-Training) ---
# (Las celdas reales de entrenamiento/evaluación deberían re-ejecutarse o duplicarse)

# 1. Predecir sobre los datos NO ETIQUETADOS FILTRADOS
# preds_unlabeled_filtered = model_ex1.predict(x_train_unlabeled_filtered, ...)
# ... seleccionar altas confianzas ...
# x_pseudo_labeled_filtered = x_train_unlabeled_filtered[high_conf_indices_filt]
# ... crear y_pseudo_labeled_filtered_cat ...

# 2. Combinar con etiquetados
# x_train_combined_filtered = np.concatenate((x_train_labeled, x_pseudo_labeled_filtered), axis=0)
# y_train_combined_filtered_cat = np.concatenate((y_train_labeled_cat, y_pseudo_labeled_filtered_cat), axis=0)
# ... barajar ...

# 3. Re-entrenar modelo Ex2 con datos filtrados
# model_ex2_filtered = build_classifier_model(...)
# model_ex2_filtered.compile(...)
# history_ex2_filtered = model_ex2_filtered.fit(x_train_combined_filtered, y_train_combined_filtered_cat, ...)

# 4. Evaluar
# loss_train_ex2_f, acc_train_ex2_f = model_ex2_filtered.evaluate(x_train_labeled, ...)
# loss_test_ex2_f, acc_test_ex2_f = model_ex2_filtered.evaluate(x_test, ...)
# results_filtered['Ex2'] = {'train_acc': acc_train_ex2_f, 'test_acc': acc_test_ex2_f}

# --- Se haría lo análogo para Ex3 y Ex4 ---
# Ex3: Entrenar AE en np.concatenate(x_train_labeled, x_train_unlabeled_filtered)
# Ex4: Usar tf.data.Dataset.from_tensor_slices(x_train_unlabeled_filtered) para unlabeled_ds

# --- Simulación de resultados (reemplazar con los reales después de ejecutar) ---
print("\n--- Simulación de Resultados Filtrados (EJECUTAR CELDAS ANTERIORES) ---")
results_filtered = {}
results_filtered['Ex1'] = results['Ex1'] # Ex1 no cambia
# Supongamos que obtenemos estos resultados tras re-entrenar:
results_filtered['Ex2'] = {'train_acc': 0.68, 'test_acc': results['Ex2']['test_acc'] + 0.01} # Ligera mejora hipotética
results_filtered['Ex3'] = {'train_acc': 0.72, 'test_acc': results['Ex3']['test_acc'] + 0.015} # Mejora hipotética
results_filtered['Ex4'] = {'train_acc': 0.75, 'test_acc': results['Ex4']['test_acc'] + 0.005} # Mejora hipotética
print("Resultados (hipotéticos) tras filtrar outliers:")
print(results_filtered)


# %% [markdown]
# ### 5.a: ¿Se mejoran los resultados con respecto a los anteriores ejercicios? ¿Qué conclusiones sacas de estos resultados?
#
# **Comparación:**
# Se debe comparar `results_filtered[ExN]['test_acc']` con `results[ExN]['test_acc']` para N=2, 3, 4.
#
# **Conclusiones:**
# *   **Impacto del Filtrado:** Si los resultados *mejoran* consistentemente (o en la mayoría de los casos) después de filtrar los outliers (`acc_test_exN_filtered > acc_test_exN`), sugiere que eliminar las instancias no etiquetadas más anómalas fue beneficioso. Estas instancias podrían haber sido ruido, datos de clases no presentes en el conjunto etiquetado inicial, o simplemente muestras muy diferentes que confundían a los modelos semi-supervisados (especialmente en self-training o en la regularización implícita del AE).
# *   **Calidad de la Detección de Outliers:** El éxito depende de si la técnica OCC (Isolation Forest sobre features de Ex1) identificó correctamente las instancias verdaderamente "malas" o ruidosas. Una mala detección podría eliminar datos útiles o no eliminar los problemáticos.
# *   **Compensación Datos vs Ruido:** Al eliminar datos (el 10% en este caso), reducimos la cantidad de información disponible del conjunto no etiquetado. La mejora solo ocurre si el beneficio de eliminar el ruido supera la pérdida de información de esas muestras eliminadas.
# *   **Interacción con SSL:** El impacto puede variar según la técnica SSL. El self-training (Ex2) podría ser más sensible al ruido (pseudo-etiquetas incorrectas de outliers), por lo que el filtrado podría ayudar más aquí. Los AE (Ex3, Ex4) podrían ser algo más robustos al ruido por naturaleza, pero aun así beneficiarse del filtrado si los outliers distorsionan mucho el espacio latente aprendido.

# %%
print("\n--- Comparación Resultados Originales vs Filtrados ---")
for i in range(1, 5):
    ex = f'Ex{i}'
    if ex in results and ex in results_filtered:
        acc_orig = results[ex]['test_acc']
        acc_filt = results_filtered[ex]['test_acc']
        improvement = acc_filt - acc_orig
        print(f"Ejercicio {i}:")
        print(f"  Accuracy Original: {acc_orig:.4f}")
        print(f"  Accuracy Filtrada: {acc_filt:.4f}")
        print(f"  Mejora: {improvement:.4f} ({improvement/acc_orig:.2%})")
    elif ex == 'Ex1': # Ex1 siempre existe y es igual
         print(f"Ejercicio 1: Accuracy {results['Ex1']['test_acc']:.4f} (no afectado por filtrado)")
    else:
        print(f"Ejercicio {i}: Resultados no disponibles.")


# %% [markdown]
# ## Ejercicio 6: Alternativa al Autoencoder (RotatioNet)
#
# Repetir los Ejercicios 3, 4 y 5 (las versiones con AE) utilizando una técnica alternativa de pre-entrenamiento no supervisado definida en "Notebook 4".
#
# **Interpretación:** Usaremos **Rotation Prediction (RotatioNet)** como alternativa.
# 1.  Crear datos de entrenamiento rotando las imágenes (0, 90, 180, 270 grados) y usando el ángulo de rotación como etiqueta.
# 2.  Entrenar un modelo (usando la misma arquitectura base convolucional del *encoder*) para predecir el ángulo de rotación. Entrenar en todos los datos (`x_train_labeled` + `x_train_unlabeled` o la versión filtrada para el paso 5).
# 3.  Usar el *encoder* pre-entrenado (la base convolucional, sin la capa de predicción de rotación) para inicializar los modelos en los ejercicios correspondientes (análogo a Ex3, Ex4, Ex5).
#
# **Puntos a Cumplir:**
# *   a. La arquitectura de la red (base convolucional/encoder) será igual a la parte encoder del AE.
# *   b. El modelo debe entrenar correctamente (aprender a predecir rotaciones).

# %%
# --- Paso 1: Preparación de Datos para RotatioNet ---
print("\n--- Ejercicio 6: Preparación Datos RotatioNet ---")

def rotate_images(images, rotation_angles):
    """Aplica rotaciones a un batch de imágenes."""
    rotated_images = []
    for angle in rotation_angles:
        if angle == 0: # 0 grados (k=0)
            rotated = images
        elif angle == 90: # 90 grados (k=1)
            rotated = tf.image.rot90(images, k=1)
        elif angle == 180: # 180 grados (k=2)
            rotated = tf.image.rot90(images, k=2)
        elif angle == 270: # 270 grados (k=3)
            rotated = tf.image.rot90(images, k=3)
        else:
             raise ValueError("Ángulo de rotación no válido")
        rotated_images.append(rotated)
    return tf.stack(rotated_images, axis=1) # Shape: (batch, num_rotations, H, W, C)

def create_rotation_dataset(images):
    """Crea un dataset para RotatioNet."""
    num_samples = images.shape[0]
    rotation_labels = np.random.randint(0, 4, size=num_samples) # 0:0, 1:90, 2:180, 3:270
    rotated_images_list = []

    # Aplicar rotación a cada imagen según su etiqueta aleatoria
    for i in range(num_samples):
        img = images[i:i+1] # Mantener la dimensión del batch
        label = rotation_labels[i]
        if label == 0:
            rotated_img = img
        else:
            rotated_img = tf.image.rot90(img, k=label)
        rotated_images_list.append(rotated_img[0]) # Quitar la dimensión del batch

    x_rotated = np.stack(rotated_images_list, axis=0)
    y_rotated_cat = utils.to_categorical(rotation_labels, 4) # 4 clases de rotación

    return x_rotated, y_rotated_cat

# Usar todos los datos de entrenamiento (o los filtrados si se repite Ex5)
# Aquí usamos los originales (sin filtrar) para el pre-entrenamiento base
x_train_for_rotnet = np.concatenate((x_train_labeled, x_train_unlabeled), axis=0)
np.random.shuffle(x_train_for_rotnet)

print(f"Creando dataset de rotación a partir de {len(x_train_for_rotnet)} imágenes...")
x_rotated_train, y_rotated_train_cat = create_rotation_dataset(x_train_for_rotnet)

print(f"Shape datos rotados: {x_rotated_train.shape}")
print(f"Shape etiquetas rotación: {y_rotated_train_cat.shape}")

# Crear dataset de validación (rotando x_test)
x_rotated_val, y_rotated_val_cat = create_rotation_dataset(x_test)


# %%
# --- Paso 2: Entrenamiento del Modelo RotatioNet ---
print("\n--- Ejercicio 6: Entrenamiento RotatioNet ---")

# Usar la misma base convolucional (Encoder)
# Crear modelo base temporal
temp_model_for_rot = build_classifier_model(INPUT_SHAPE, NUM_CLASSES)
rotnet_encoder_output_layer = None
for layer in temp_model_for_rot.layers:
    if isinstance(layer, layers.Flatten):
        break
    rotnet_encoder_output_layer = layer.output
rotnet_encoder_base = keras.Model(inputs=temp_model_for_rot.input,
                                  outputs=rotnet_encoder_output_layer,
                                  name="rotnet_encoder_base")
rotnet_encoder_base.trainable = True # Entrenar la base

# Añadir cabezal para predecir rotación (4 clases)
rotnet_input = keras.Input(shape=INPUT_SHAPE)
features = rotnet_encoder_base(rotnet_input)
flat_features = layers.Flatten()(features)
# Añadir capas densas si se desea, o ir directo a la salida
# x_rot = layers.Dense(128, activation='relu')(flat_features)
# x_rot = layers.Dropout(0.5)(x_rot)
rotation_output = layers.Dense(4, activation='softmax', name='rotation_output')(flat_features)

rotnet_model = keras.Model(inputs=rotnet_input, outputs=rotation_output, name="rotnet_predictor")
rotnet_model.summary()

# Compilar RotatioNet
rotnet_model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                     loss=losses.CategoricalCrossentropy(),
                     metrics=[metrics.CategoricalAccuracy()])

# Callback para RotatioNet
early_stopping_rotnet = callbacks.EarlyStopping(monitor='val_categorical_accuracy',
                                                mode='max',
                                                patience=PATIENCE_EARLY_STOPPING,
                                                restore_best_weights=True,
                                                verbose=VERBOSE)

# Entrenar RotatioNet
start_time = time.time()
history_rotnet = rotnet_model.fit(x_rotated_train, y_rotated_train_cat,
                                  batch_size=BATCH_SIZE,
                                  epochs=EPOCHS_ROTNET,
                                  validation_data=(x_rotated_val, y_rotated_val_cat),
                                  callbacks=[early_stopping_rotnet],
                                  verbose=VERBOSE)
end_time = time.time()
print(f"Tiempo de entrenamiento RotatioNet: {end_time - start_time:.2f} segundos")

# Evaluar rendimiento en predicción de rotación (verificar que aprendió algo)
loss_rot_val, acc_rot_val = rotnet_model.evaluate(x_rotated_val, y_rotated_val_cat, verbose=0)
print(f"Rendimiento RotatioNet en Validación: Loss={loss_rot_val:.4f}, Accuracy={acc_rot_val:.4f}")
# Una accuracy significativamente > 25% indica que aprendió.


# %% [markdown]
# --- Paso 3: Repetir Ejercicios 3, 4 y 5 usando Encoder de RotatioNet ---
#
# Ahora, se debe usar `rotnet_encoder_base` (la parte convolucional pre-entrenada con RotatioNet) en lugar del encoder pre-entrenado con el Autoencoder en las implementaciones de:
# *   **Ejercicio 3 (Repetido con RotatioNet):** Congelar `rotnet_encoder_base`, añadir el cabezal clasificador (parte densa + softmax), y entrenar solo el cabezal con datos etiquetados.
# *   **Ejercicio 4 (Repetido con RotatioNet):** Esto es más complejo. El entrenamiento conjunto aquí significaría combinar la pérdida de predicción de rotación (en datos no etiquetados/rotados) y la pérdida de clasificación (en datos etiquetados/originales). Requiere un `train_step` personalizado aún más elaborado o un enfoque diferente. *Por simplicidad, podríamos omitir la repetición de Ex4 con RotatioNet o simplemente hacer el enfoque de dos pasos (como Ex3).* **Vamos a implementar solo el enfoque de dos pasos (análogo a Ex3).**
# *   **Ejercicio 5 (Repetido con RotatioNet):**
#     1.  Pre-entrenar RotatioNet usando `x_train_labeled` + `x_train_unlabeled_filtered`.
#     2.  Usar este encoder pre-entrenado filtrado para la clasificación final (análogo a Ex3 filtrado).
#
# **NOTA:** Nuevamente, por brevedad, solo implementaremos el análogo al Ejercicio 3 usando el `rotnet_encoder_base` pre-entrenado en datos *originales* (sin filtrar). Llamaremos a este resultado `Ex6_as_Ex3`.

# %%
# --- Implementación análoga a Ex3 usando RotatioNet Encoder ---
print("\n--- Ejercicio 6: Entrenando Clasificador con RotatioNet Encoder (Análogo a Ex3) ---")

# Extraer el encoder base pre-entrenado
rotnet_encoder_pretrained = rotnet_model.get_layer('rotnet_encoder_base')
rotnet_encoder_pretrained.trainable = False # Congelar pesos

# Construir clasificador final
classifier_input_rot = keras.Input(shape=INPUT_SHAPE)
features_rot = rotnet_encoder_pretrained(classifier_input_rot) # Usar encoder pre-entrenado
flat_features_rot = layers.Flatten()(features_rot)

# Añadir las capas densas (iguales al modelo original)
x_rot_cls = layers.Dense(256, activation='relu')(flat_features_rot)
x_rot_cls = layers.BatchNormalization()(x_rot_cls)
x_rot_cls = layers.Dropout(0.4)(x_rot_cls)
x_rot_cls = layers.Dense(128, activation='relu')(x_rot_cls)
x_rot_cls = layers.BatchNormalization()(x_rot_cls)
x_rot_cls = layers.Dropout(0.4)(x_rot_cls)
classifier_output_rot = layers.Dense(NUM_CLASSES, activation='softmax')(x_rot_cls)

model_ex6_as_ex3 = keras.Model(inputs=classifier_input_rot, outputs=classifier_output_rot, name="classifier_from_rotnet")
model_ex6_as_ex3.summary()

# Compilar el clasificador
model_ex6_as_ex3.compile(optimizer=optimizers.Adam(learning_rate=5e-4), # Misma lr que en Ex3
                         loss=losses.CategoricalCrossentropy(),
                         metrics=[metrics.CategoricalAccuracy()])

# Entrenar solo con datos etiquetados
early_stopping_ex6 = callbacks.EarlyStopping(monitor='val_loss',
                                             patience=PATIENCE_EARLY_STOPPING,
                                             restore_best_weights=True,
                                             verbose=VERBOSE)

start_time = time.time()
history_ex6_as_ex3 = model_ex6_as_ex3.fit(x_train_labeled, y_train_labeled_cat,
                                        batch_size=BATCH_SIZE,
                                        epochs=EPOCHS_SUPERVISED,
                                        validation_data=(x_test, y_test_cat),
                                        callbacks=[early_stopping_ex6],
                                        verbose=VERBOSE)
end_time = time.time()
print(f"Tiempo de entrenamiento Clasificador (Ex6 as Ex3): {end_time - start_time:.2f} segundos")

# Evaluar
print("\n--- Evaluación Modelo Ejercicio 6 (RotatioNet Pre-trained - Análogo a Ex3) ---")
loss_train_ex6, acc_train_ex6 = model_ex6_as_ex3.evaluate(x_train_labeled, y_train_labeled_cat, verbose=0)
loss_test_ex6, acc_test_ex6 = model_ex6_as_ex3.evaluate(x_test, y_test_cat, verbose=0)

results['Ex6_as_Ex3'] = {'train_acc': acc_train_ex6, 'test_acc': acc_test_ex6}

print(f"Rendimiento en Entrenamiento (sobre datos etiquetados):")
print(f"  Loss: {loss_train_ex6:.4f}")
print(f"  Accuracy: {acc_train_ex6:.4f}")

print(f"Rendimiento en Prueba:")
print(f"  Loss: {loss_test_ex6:.4f}")
print(f"  Accuracy: {acc_test_ex6:.4f}")


# %% [markdown]
# **Contestar Preguntas Ejercicio 6 (basado en `Ex6_as_Ex3` y comparación con `Ex3`):**
#
# *   **Comparación:** Comparar `results['Ex6_as_Ex3']['test_acc']` con `results['Ex3']['test_acc']`.
# *   **Conclusiones:** ¿Fue RotatioNet un pre-entrenamiento más efectivo que el Autoencoder para esta tarea? A veces, tareas discriminativas auto-supervisadas como RotatioNet capturan características más útiles para la clasificación downstream que las tareas generativas como los AE. Sin embargo, RotatioNet puede ser más simple y no aprender detalles finos de la misma manera que un AE. El resultado depende del dataset y las arquitecturas. Si `acc_test_ex6 > acc_test_ex3`, RotatioNet fue mejor en este contexto.

# %%
print("\n--- Comparación Resultados Ex3 (AE Pre-trained) vs Ex6 (RotatioNet Pre-trained) ---")
print(f"Accuracy Test Ex3 (AE Pre-trained): {results['Ex3']['test_acc']:.4f}")
print(f"Accuracy Test Ex6 (RotatioNet Pre-trained): {results['Ex6_as_Ex3']['test_acc']:.4f}")

improvement_rot_vs_ae = results['Ex6_as_Ex3']['test_acc'] - results['Ex3']['test_acc']
print(f"Mejora RotatioNet vs AE: {improvement_rot_vs_ae:.4f} ({improvement_rot_vs_ae/results['Ex3']['test_acc']:.2%})")
if improvement_rot_vs_ae > 0:
    print("RotatioNet fue más efectivo que el AE para pre-entrenamiento en este caso.")
else:
    print("El AE fue más efectivo o similar a RotatioNet para pre-entrenamiento en este caso.")

# %% [markdown]
# ## Resumen Final de Resultados

# %%
print("\n--- RESUMEN FINAL ACCURACY EN TEST ---")
print(f"Ex1: Supervisado Baseline:          {results.get('Ex1', {}).get('test_acc', 'N/A'):.4f}")
print(f"Ex2: Self-Training:                 {results.get('Ex2', {}).get('test_acc', 'N/A'):.4f}")
print(f"Ex3: AE Pre-trained (2 Pasos):      {results.get('Ex3', {}).get('test_acc', 'N/A'):.4f}")
print(f"Ex4: AE Joint (1 Paso):             {results.get('Ex4', {}).get('test_acc', 'N/A'):.4f}")
print("-" * 35)
print("Resultados tras Filtrado de Outliers (Ex5):")
print(f"Ex1 (Filtrado): No aplica           {results_filtered.get('Ex1', {}).get('test_acc', 'N/A'):.4f}")
print(f"Ex2 (Filtrado): Self-Training:      {results_filtered.get('Ex2', {}).get('test_acc', 'N/A'):.4f}")
print(f"Ex3 (Filtrado): AE Pre-trained:     {results_filtered.get('Ex3', {}).get('test_acc', 'N/A'):.4f}")
print(f"Ex4 (Filtrado): AE Joint:           {results_filtered.get('Ex4', {}).get('test_acc', 'N/A'):.4f}")
print("-" * 35)
print("Resultados con RotatioNet (Ex6):")
print(f"Ex6 (RotatioNet Pre-trained, análogo Ex3): {results.get('Ex6_as_Ex3', {}).get('test_acc', 'N/A'):.4f}")
# Añadir aquí los resultados de Ex6 repitiendo Ex5 si se implementaron


# %% [markdown]
# ## Entrega
#
# *   Este notebook (`.ipynb`) con el código, resultados y respuestas a las preguntas.
# *   Un documento breve (`.pdf` o `.docx`) justificando las decisiones tomadas (arquitecturas, hiperparámetros clave, interpretación de v=0.9, elección de RotatioNet) y contestando formalmente a las preguntas del enunciado, haciendo referencia a los resultados obtenidos en este notebook.