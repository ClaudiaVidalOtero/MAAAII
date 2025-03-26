# P2 MAAAI viva

#PREPROCESADO DE DATOS:
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
import numpy as np

# Cargar el conjunto de datos CIFAR-100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(f"Entrenamiento: {x_train.shape}, Prueba: {x_test.shape}")

# Eliminamos el 80% de las etiquetas en el conjunto de entrenamiento
np.random.seed(42)
unlabeled_indices = np.random.choice(len(x_train), size=int(0.8 * len(x_train)), replace=False)
labeled_indices = np.setdiff1d(np.arange(len(x_train)), unlabeled_indices)

# Creamos las particiones de datos
x_train_labeled = x_train[labeled_indices]
y_train_labeled = y_train[labeled_indices]
x_train_unlabeled = x_train[unlabeled_indices]
y_train_unlabeled = y_train[unlabeled_indices]  

print(f"Instancias de entrenamiento etiquetadas: {x_train_labeled.shape}, No etiquetadas: {x_train_unlabeled.shape}")
