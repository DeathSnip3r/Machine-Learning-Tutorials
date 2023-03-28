#reference

import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

numBooks = 7

for bookIndex in range(numBooks+1):
    bookName = "HP" + str(bookIndex) + ".txt"

    with open(bookName, "r") as tFile:

        for line in tFile.readlines():
            words = np.array(line.split())


bookWords = np.array()

data = pd

xValues = data.iloc[:, :64]
yValues = data.iloc[:, 64]

xTrain, xTest, yTrain, yTest = train_test_split(xValues, yValues, test_size=0.8, random_state=0)

sc_X = StandardScaler()
xTrain = sc_X.fit_transform(xTrain)
xTest = sc_X.fit_transform(xTest)

# initializaing the NB
classifer = BernoulliNB()

# training the model
classifer.fit(xTrain, yTrain)

# testing the model
yPred = classifer.predict(xTest)

print("Confusion Matrix")
print(confusion_matrix(yTest, yPred))

print(round(accuracy_score(yPred, yTest) * 100, 4), "%\n")