import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


def getDistribution(x, mean, variance):  # Using Gaussian distribution formula
    xMeanDifference = x - mean
    exponent = -(xMeanDifference * xMeanDifference) / (2 * variance)
    multiplier = 1 / math.sqrt(2 * math.pi * variance)
    return multiplier * math.exp(exponent)


featureIndex = 1  # Index of the feature we're plotting
classification = 1  # Which classification value we're showing the feature under
labels = ["Variance", "Skewness", "Curtosis", "Entropy"]
data = np.loadtxt("banknote_authentication.csv", delimiter=";",
                  dtype="float", skiprows=1)  # Reading in data
# Only using feature column and classification column
data = data[:, (featureIndex, 4)]
# Getting all values with the same classification
xValues = data[np.where(data[:, 1] == classification), 0].flatten()
xValues = np.sort(xValues)
mean = np.mean(xValues)
variance = np.var(xValues)

x = np.arange(np.min(xValues), np.max(xValues), 0.001)

plt.plot(xValues, norm.pdf(xValues, mean, math.sqrt(variance)))
n, bins, patches = plt.hist(
    xValues, bins=15, density=True, facecolor="r", alpha=0.75)
plt.xlabel(labels[featureIndex])
plt.ylabel("Probability")
plt.title("Probability of all {} values".format(labels[featureIndex]))
plt.show()
