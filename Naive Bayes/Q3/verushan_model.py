import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv("banknote_authentication.csv", delimiter=";")
xValues = data.iloc[:, :4]
yValues = data.iloc[:, 4]

xTrain, xTest, yTrain, yTest = train_test_split(
    xValues, yValues, test_size=0.2, random_state=9)

# initializaing the NB
classifer = GaussianNB()

# training the model
classifer.fit(xTrain, yTrain)

# testing the model
yPred = classifer.predict(xTest)

print("Confusion Matrix")
print(confusion_matrix(yTest, yPred))
print(round(accuracy_score(yPred, yTest) * 100, 4), "%\n")
