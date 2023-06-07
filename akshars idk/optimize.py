import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from skopt import BayesSearchCV

# Load the data and labels
data = pd.read_csv("traindata.txt", header=None)
labels = pd.read_csv("trainlabels.txt", header=None)

# One-hot encode the labels
encoder = OneHotEncoder(sparse=False)
labels_encoded = encoder.fit_transform(labels)

# Data preprocessing
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split the data into training and testing sets with shuffling
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels_encoded, test_size=0.30, shuffle=True
)


# Define the MLP model
class CustomKerasClassifier(KerasClassifier):
    def __init__(self, kernel_regularizer=None, **kwargs):
        super(CustomKerasClassifier, self).__init__(**kwargs)
        self.kernel_regularizer = kernel_regularizer

    def get_params(self, **params):
        params = super(CustomKerasClassifier, self).get_params(**params)
        params["kernel_regularizer"] = self.kernel_regularizer
        return params


# Create the original create_model function without any changes
def create_model(
    kernel_regularizer=None, learning_rate=0.001, decay_steps=10000, decay_rate=0.96
):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                784,
                activation="relu",
                input_shape=(data.shape[1],),
                kernel_regularizer=kernel_regularizer,
            ),
            tf.keras.layers.Dense(
                256, activation="relu", kernel_regularizer=kernel_regularizer
            ),
            tf.keras.layers.Dense(
                256, activation="relu", kernel_regularizer=kernel_regularizer
            ),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    global_step = tf.compat.v1.train.get_or_create_global_step()
    learning_rate = tf.compat.v1.train.exponential_decay(
        learning_rate, global_step, decay_steps, decay_rate, staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Define the parameter search space
param_space = {
    "kernel_regularizer": [
        None,
        tf.keras.regularizers.l2(0.0001),
        tf.keras.regularizers.l2(0.001),
        tf.keras.regularizers.l2(0.01),
    ],
    "learning_rate": (0.0001, 0.1, "log-uniform"),
    "decay_steps": (1000, 10000),
    "decay_rate": (0.5, 0.99),
}

# Perform Bayesian optimization search with cross-validation
bayes_search = BayesSearchCV(
    CustomKerasClassifier(build_fn=create_model),
    param_space,
    cv=3,
    n_iter=50,
    n_jobs=-1,
)

bayes_search.fit(data, labels)

# Get the best model and its parameters
best_model = bayes_search.best_estimator_
best_params = bayes_search.best_params_

print("Best Parameters:", best_params)

# Define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=8, restore_best_weights=True
)

# Train the best model with early stopping
best_model.fit(
    data,
    labels,
    epochs=50,
    batch_size=32,
    validation_split=0.3,
    callbacks=[early_stopping],
)

# Save the best model
best_model.model.save("best_model.h5")
print("Best model saved.")

# Evaluate the best model using cross-validation
cv_scores = cross_val_score(best_model, data, labels, cv=3)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Split the data into training and testing sets with shuffling
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.3, shuffle=True
)

# Evaluate the best model on the test data
test_loss, test_accuracy = best_model.model.evaluate(test_data, test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
