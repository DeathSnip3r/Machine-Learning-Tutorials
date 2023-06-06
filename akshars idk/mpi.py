import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd

# Load the data and labels
data = pd.read_csv('traindata.txt', header=None)
labels = pd.read_csv('trainlabels.txt', header=None)

# Data preprocessing
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)

# Define the model architecture
def create_model(learning_rate=0.001, dropout_rate=0.3):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(10, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create the model
model = make_pipeline(StandardScaler(), keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=1))

# Define the hyperparameters search space
param_grid = {
    'kerasclassifier__learning_rate': [0.001, 0.0001],
    'kerasclassifier__dropout_rate': [0.1, 0.3, 0.5],
    'kerasclassifier__epochs': [10, 20],
    'kerasclassifier__batch_size': [32, 64]
}

# Perform hyperparameter optimization using GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=3)
grid_search.fit(train_data, train_labels)

# Print the best parameters and accuracy
print("Best Parameters: ", grid_search.best_params_)
print("Best Accuracy: {:.2%}".format(grid_search.best_score_))

# Evaluate the best model on the test data
best_model = grid_search.best_estimator_
test_score = best_model.score(test_data, test_labels)
print("Test Accuracy: {:.2%}".format(test_score))
