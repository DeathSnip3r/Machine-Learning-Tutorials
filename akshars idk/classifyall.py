import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the h5 model and test data
model = keras.models.load_model("best_model.h5")
test_data = pd.read_csv("traindata.txt", header=None)

# Scale the test data
scaler = StandardScaler()
scaled_test_data = scaler.fit_transform(test_data)

# Run the model on the scaled test data
predictions = model.predict(scaled_test_data)

# Save the predicted labels to a file as integers
predicted_labels = np.argmax(predictions, axis=1)
np.savetxt("predicted_labels.txt", predicted_labels, fmt="%d")

############################################################################

# Take in the trainlabels.txt file and the predicted_labels.txt file
# and print the accuracy score as a percentage
true_labels = pd.read_csv("trainlabels.txt", header=None)
predicted_labels = pd.read_csv("predicted_labels.txt", header=None)
print(
    "Model accuracy: {:.2f}%".format(
        accuracy_score(true_labels, predicted_labels) * 100
    )
)
