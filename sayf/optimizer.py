import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV

# Load the data and labels
data, labels = np.genfromtxt("traindata.txt", delimiter=","), np.genfromtxt("trainlabels.txt", delimiter=",")

# Data preprocessing
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)

# Define the model architecture
def create_model(learning_rate, dropout_rate):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(10, activation='softmax')
    ])

    optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create the model
model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=1)

# Define the hyperparameters search space
param_space = {
    'learning_rate': (0.0001, 0.1, 'log-uniform'),
    'dropout_rate': (0.1, 0.9, 'uniform'),
    'epochs': (10, 500),
    'batch_size': (32, 512),
}

# Perform hyperparameter optimization using Bayesian optimization
opt = BayesSearchCV(model, param_space, cv=3, n_jobs=-1)
opt.fit(train_data, train_labels)

# Print the best parameters and accuracy
print("Best Parameters: ", opt.best_params_)
print("Best Accuracy: {:.2%}".format(opt.best_score_))

# Evaluate the best model on the test data
best_model = opt.best_estimator_
test_loss, test_acc = best_model.evaluate(test_data, test_labels)
print("Test Accuracy: {:.2%}".format(test_acc))
