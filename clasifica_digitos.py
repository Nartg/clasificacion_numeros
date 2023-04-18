import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.utils.vis_utils import plot_model  # necesita matplotlib, pydot, graphviz

# Cargar el conjunto de datos MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los datos de entrada
x_train = x_train / 255.0
x_test = x_test / 255.0

# Crear el modelo
model = Sequential()

# Añadir capa de aplanado (flatten)
model.add(Flatten(input_shape=(28, 28, 1)))

# Añadir capa completamente conectada con 128 unidades y función de activación ReLU
model.add(Dense(128, activation='relu'))

# Añadir capa de salida completamente conectada con 10 unidades (correspondientes a las 10 clases de dígitos)
# y función de activación softmax para obtener probabilidades de clasificación
model.add(Dense(10, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Entrenar el modelo
# model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
model.fit(x_train, y_train, epochs=10)

# Evaluar el modelo en el conjunto de prueba
model.evaluate(x_test, y_test)

# Exportar el modelo
model.save('modelo_mnist.h5')















































































#