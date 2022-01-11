import os

import tensorflow as tf
from tensorflow import keras

# load dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# take a subset for fast demostration
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

# reshape to fit a dense layer
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Define a simple sequential model
def create_model():
    model = tf.keras.models.Sequential(
        [
            keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    return model


# Train the model
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model.
model.save('mnist_model')
