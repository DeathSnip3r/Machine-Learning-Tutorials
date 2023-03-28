import numpy as np

DATA_FILE_NAME = "simple-food-reviews.txt"


class Review:
    def __init__(self, reviewLine):
        spacePos = reviewLine.find(" ")
        # Checks if this is a good or bad review
        self.classification = reviewLine[:spacePos] == "1"
        # Remove endline character and store the review
        self.review = reviewLine[spacePos + 1:].replace("\n", "")


def import_data():
    trainingData = []
    testData = []

    numTrainingData = 12  # Number of reviews to use as training data
    numTestData = 6  # Number of reviews to use as test data

    with open(DATA_FILE_NAME, "r") as tf:  # Reading from data file
        # Store every review as a class object
        allReviews = [Review(review) for review in tf.readlines()]
        # np.random.shuffle(allReviews)

    trainingData = allReviews[:numTrainingData]
    testData = allReviews[numTrainingData: numTrainingData + numTestData]

    return trainingData, testData


def import_custom_data():  # Use all simple reviews as training and two custom reviews as test data
    trainingData = []
    testData = []

    with open(DATA_FILE_NAME, "r") as tf:  # Reading from data file
        # Store every review as a class object
        trainingData = [Review(review) for review in tf.readlines()]

    with open("test-food-reviews.txt", "r") as tf:
        testData = [Review(review) for review in tf.readlines()]

    return trainingData, testData


trainingData, testData = import_data()  # For normal training mode
# trainingData, testData = import_custom_data()  # Custom training model

wordsFound = dict()  # word -> {numHam, numSpam}
numBad, numGood = 0, 0

for review in trainingData:

    if review.classification:
        numGood += 1
    else:
        numBad += 1

    for word in review.review.split():

        if word in wordsFound:

            if review.classification:
                wordsFound[word][0] += 1
            else:
                wordsFound[word][1] += 1
        else:

            if review.classification:
                wordsFound[word] = [1, 0]
            else:
                wordsFound[word] = [0, 1]

numReviews = len(trainingData)
pOverallBadReview = numBad / numReviews
pOverallGoodReview = numGood / numReviews
kValue = 1
numClasses = 2  # A review is either good or bad

for word in wordsFound.keys():  # Store number of good and bad reviews for each word as a fraction
    # Number of good and bad reviews from this class
    classNums = wordsFound[word]

    if classNums[0] == 0 or numGood == 0:
        classNums[0] = (classNums[0] + kValue) / \
            (numGood + numClasses * kValue)
    else:
        classNums[0] /= numGood

    if classNums[1] == 0 or numBad == 0:
        classNums[1] = (classNums[1] + kValue) / (numBad + numClasses * kValue)
    else:
        classNums[1] /= numBad

# Model has been trained now iterate through the test data
confusionMatrix = np.zeros((2, 2))

for review in testData:
    pReviewBad = 1
    pReviewGood = 1

    for word in wordsFound:

        if word in review.review:
            # Multiply by probability of this word being part of a good/bad review
            pReviewBad *= wordsFound[word][1]
            pReviewGood *= wordsFound[word][0]
        else:
            # Multiply by probability of this word not being part of a good/bad review
            pReviewBad *= 1 - wordsFound[word][1]
            pReviewGood *= 1 - wordsFound[word][0]

    for word in review.review.split():

        if word not in wordsFound:
            pReviewBad *= kValue / (kValue * numClasses)

    pReviewGivenBad = (pReviewBad * pOverallBadReview) / (
        pReviewBad * pOverallBadReview + pReviewGood * pOverallGoodReview
    )  # Probability of this being a review given that it's bad

    pReviewGivenGood = 1 - pReviewGivenBad
    # Probability of this being a review given that it's good

    reviewPrediction = True

    if pReviewGivenBad >= pReviewGivenGood:
        reviewPrediction = False

    if reviewPrediction != review.classification:

        if review.classification:
            confusionMatrix[1, 0] += 1
        else:
            confusionMatrix[0, 1] += 1
    else:

        if review.classification:
            confusionMatrix[0, 0] += 1
        else:
            confusionMatrix[1, 1] += 1

# Prints out words and their probabilities
# for key, value in sorted(wordsFound.items()):
#     print(key, end=" ")

#     print(*reversed(value), sep=" ")


print("P(Positive | Review) = {}, P(Negative | Review) = {}".format(
    pReviewGivenGood, pReviewGivenBad))
accuracy = np.trace(confusionMatrix) / len(testData)
print(*confusionMatrix, sep="\n")
print("Accuracy", round(accuracy, 4) * 100)