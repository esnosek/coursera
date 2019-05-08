import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([1.0, 2.0, 3.0, 4.0])
ys = np.array([1.0, 1.5, 2.0, 2.5])
model.fit(xs, ys, epochs=2000)
print(model.predict([7.0]))