import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv("smalldigits.csv", delimiter=",")  # Importing data

xValues = data.iloc[:, :64]  # Read all rows but only the first 64 columns
yValues = data.iloc[:, 64]  # Read all rows but only the index 64 column

# Use 80% data for testing
# Random state is a seed to randomly shuffle
xTrain, xTest, yTrain, yTest = train_test_split(
    xValues, yValues, test_size=0.8, random_state=0)

# initializaing the NB
classifer = BernoulliNB()

# training the model
classifer.fit(xTrain, yTrain)

# testing the model
yPred = classifer.predict(xTest)

print("Confusion Matrix")
print(confusion_matrix(yTest, yPred))
print(round(accuracy_score(yPred, yTest) * 100, 4), "%\n")
