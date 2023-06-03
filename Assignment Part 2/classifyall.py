import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

def main():
    # Load the model and optimizer weights
    model = create_model()
    model.load_weights()
    optimizer = keras.optimizers.SGD()
    optimizer.load_weights()

    inputs = np.loadtxt(sys.stdin).reshape(-1, 1)
    outputs = np.loadtxt("outputs.txt")

    correct = 0
    num_points = len(inputs)

    for data_point, label in zip(inputs, outputs):
        # Get output from network
        prediction = run_test_case(model, data_point)

        if label == prediction:
            correct += 1

    print("Accuracy: {}%".format(correct / num_points * 100))


def create_model():
    model = keras.Sequential(
        [
            keras.layers.Conv2D(10, kernel_size=5, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Conv2D(20, kernel_size=5, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dense(50, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model