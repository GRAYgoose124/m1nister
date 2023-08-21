# Import necessary libraries
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# load model checkpoint
if Path("mnist_classifier.keras").exists():
    model = models.load_model("mnist_classifier.keras")
else:
    # Define the model architecture
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", "mse"]
    )

# Train the model
model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=64,
    validation_data=(test_images, test_labels),
)

# save model checkpoint
model.save("mnist_classifier.keras")

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test accuracy:", test_acc)

# If you want to save the model
# model.save('mnist_classifier.h5')
