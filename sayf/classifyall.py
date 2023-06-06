import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow import keras

# Load the h5 model and test data
model = keras.models.load_model("58-1.h5")
test_data = np.genfromtxt("traindata.txt", delimiter=",")

# Run the model on the test data
predictions = model.predict(test_data)

# Save the predicted labels to a file as integers
predicted_labels = np.argmax(predictions, axis=1)
np.savetxt("predicted_labels.txt", predicted_labels, fmt="%d")

############################################################################

# Take in the trainlabels.txt file and the predicted_labels.txt file
# and print the accuracy score as a percentage
true_labels = np.genfromtxt("trainlabels.txt", delimiter=",")
predicted_labels = np.genfromtxt("predicted_labels.txt", delimiter=",")
print(
    "Model accuracy: {:.2f}%".format(
        accuracy_score(true_labels, predicted_labels) * 100
    )
)
