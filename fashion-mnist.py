from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import math
import numpy as numpy
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# prepare the data
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# output labels
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# dataset size
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

# data processing
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# cache the dataset for better loading
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

# verify the images from training dataset by reshaping and plotting them
plt.figure(figsize=(10,10))
i = 0
for (image, label) in test_dataset.take(5):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
    i += 1
plt.show()

# tf.keras.Sequential creates a neural net by taking layers from input to output sequetially
# activation - ReLU: ReLU as an activation function allows us to deal with the non linear dependencies / interaction between inputs values
# as ReLU only outputs the value when the net result produced at the node is positive
# activation - softmax: Softmax is like logistic regression but for multi label classification
# it provides a probability of current input Xi belonging to the class label Lj
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28, 1]),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax),
])

# loss: sparse_categorical_crossentropy is used to measure the dissimilarity between the distribution of observed class labels and the predicted probabilities.
# Categorical refers to the possibility of having more than two classes. 
# Sparse refers to using a single integer from zero to the number of classes minus one for class labels, instead of a dense one-hot encoding of the class label (e.g. { 1,0,0; 0,1,0; or 0,0,1 }).
# Cross-entropy loss increases as the predicted probability diverges from the actual label
# optimizer: using TensorFlow's default Adam optimizer to adjust weights to reduce loss value
# typical optimizer values range from 0.1 to 0.001 (lower the value more accurate the results and higher the training time)
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])

BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# model_history contains various training performance parameters and their values
# using these values we can plot graphs or view the changing parameter values over time
model_history = model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/BATCH_SIZE))
print('Accuracy on test dataset:', test_accuracy)

# finally, verifying the models prediction power
for test_images, test_labels in test_dataset.take(10):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)
    print("Predicted: ", labels[np.argmax(predictions[0])], ", Expected: ", labels[test_labels[0]])
