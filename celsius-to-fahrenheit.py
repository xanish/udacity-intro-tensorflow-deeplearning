from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as numpy
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# prepare the data
celsius = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

# tf.keras.Sequential creates a neural net by taking layers from input to output sequetially
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=4, input_shape=[1]),
    tf.keras.layers.Dense(units=4),
    tf.keras.layers.Dense(units=1),
])

# loss: using the simple mean square error metric for loss, it allows for a 
# number of small errors as acceptable and doesn't allow few large errors
# optimizer: using TensorFlow's default Adam optimizer to adjust weights to reduce loss value
# typical optimizer values range from 0.1 to 0.001 (lower the value more accurate the results and higher the training time)
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))

# model_history contains various training performance parameters and their values
# using these values we can plot graphs or view the changing parameter values over time
model_history = model.fit(celsius, fahrenheit, epochs=500)

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(model_history.history['loss'])

# finally, verifying the models prediction power
print(model.predict([100.0]))
