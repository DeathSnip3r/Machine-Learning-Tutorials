import os
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Get a list of all .h5 files in the models folder
model_dir = "models"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".h5")]

# Create an output file to write the accuracy and file name
output_file = open("accuracy_output.txt", "w")

# Iterate over each .h5 file
for model_file in model_files:
    # Load the h5 model
    model_path = os.path.join(model_dir, model_file)
    model = keras.models.load_model(model_path)

    # Load the test data
    test_data = pd.read_csv("traindata.txt", header=None)

    # Scale the test data
    scaler = StandardScaler()
    scaled_test_data = scaler.fit_transform(test_data)

    # Run the model on the scaled test data
    predictions = model.predict(scaled_test_data)

    # Save the predicted labels to a file as integers
    predicted_labels = np.argmax(predictions, axis=1)
    np.savetxt("predicted_labels.txt", predicted_labels, fmt="%d")

    # Load the true labels
    true_labels = pd.read_csv("trainlabels.txt", header=None)

    # Calculate the accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Write the accuracy and file name to the output file
    output_file.write(
        "Model accuracy for {}: {:.2f}%\n".format(model_file, accuracy * 100)
    )

    # Print the accuracy and file name to the terminal
    print("Model accuracy for {}: {:.2f}%".format(model_file, accuracy * 100))

# Close the output file
output_file.close()
