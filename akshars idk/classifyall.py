import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the h5 model and test data
model = keras.models.load_model("0.86.h5")
test_data = pd.read_csv("testdata.txt", header=None)

# Scale the test data
scaler = StandardScaler()
scaled_test_data = scaler.fit_transform(test_data)

# Run the model on the scaled test data
predictions = model.predict(scaled_test_data)

# Save the predicted labels to a file as integers
predicted_labels = np.argmax(predictions, axis=1)
np.savetxt("testlabels.txt", predicted_labels, fmt="%d")
