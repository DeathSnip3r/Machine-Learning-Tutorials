import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

# Load the data and labels
data, labels = np.genfromtxt("traindata.txt", delimiter=","), np.genfromtxt("trainlabels.txt", delimiter=",")

# Data preprocessing
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Generate rotated data
rotation_angles = [-1, -1]  # Specify the rotation angles to generate new data
rotated_data = []
for vector in data:
    rotated_data.extend([np.roll(vector, random.choice(rotation_angles))])
rotated_data = np.array(rotated_data)

# Add noise to the data
noise_factor = 0.0001  # Specify the noise factor
noisy_data = rotated_data + noise_factor * np.random.randn(*rotated_data.shape)

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(noisy_data)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(normalized_data, labels, test_size=0.2)

# Define the dropout rate
dropout_rate = 0.4460576282054137

# Define the model architecture
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
    keras.layers.Dropout(dropout_rate),  # Adding dropout regularization
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(dropout_rate),  # Adding dropout regularization
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model with RMSprop optimizer
optimizer = keras.optimizers.RMSprop(learning_rate=0.005205178926419261)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=200, batch_size=512, verbose=1)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_data, test_labels)

# Save the model
model.save("trained_model.h5")

# Save the predicted labels to a file as integers
# predicted_probabilities = model.predict(test_data)
# predicted_labels = np.argmax(predicted_probabilities, axis=1)
# np.savetxt("testlabels.txt", predicted_labels.astype(int))

# Print the test accuracy as a percentage
print("Test accuracy: {:.2%}".format(test_acc))
